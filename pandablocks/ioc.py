# Creating EPICS records directly from PandA Blocks and Fields

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from string import digits
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from softioc import asyncio_dispatcher, builder, softioc
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import (
    Arm,
    ChangeGroup,
    Disarm,
    GetBlockInfo,
    GetChanges,
    GetFieldInfo,
    GetMultiline,
    Put,
)
from pandablocks.hdf import write_hdf_files
from pandablocks.responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    BlockInfo,
    Changes,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TableFieldDetails,
    TableFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)

# Define the public API of this module
# TODO!
__all__ = ["create_softioc"]

TIMEOUT = 2

# Constants used in bool records
ZNAM_STR = "0"
ONAM_STR = "1"


@dataclass
class _BlockAndFieldInfo:
    """Contains all available information for a Block, including Fields and all the
    Values for `block_info.number` instances of the Fields."""

    block_info: BlockInfo
    fields: Dict[str, FieldInfo]
    values: Dict[str, Union[str, List[str]]]
    # See `_panda_to_epics_name` to create key for Dict from keys returned
    # from GetChanges


@dataclass
class _RecordInfo:
    """A container for a record and extra information needed to later update
    the record.

    `record`: The PythonSoftIOC RecordWrapper instance
    `data_type_func`: Function to convert string data to form appropriate for the record
    `labels`: List of valid values for the record. If not None, the `record` is
        an enum type.
    `table_updater`: Class instance that managed updating table records. If present
        the `record` is part of a larger table."""

    record: RecordWrapper
    data_type_func: Callable
    labels: Optional[List[str]] = None
    table_updater: Optional["_TableUpdater"] = None


# TODO: Be fancy and turn this into a custom type for Dict keys etc?
def _panda_to_epics_name(field_name: str) -> str:
    """Convert PandA naming convention to EPICS convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return field_name.replace(".", ":")


def _epics_to_panda_name(field_name: str) -> str:
    """Convert EPICS naming convention to PandA convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return field_name.replace(":", ".")


def create_softioc(host: str, record_prefix: str) -> None:
    """Create a PythonSoftIOC from fields and attributes of a PandA.

    This function will introspect a PandA for all defined Blocks, Fields of each Block,
    and Attributes of each Field, and create appropriate EPICS records for each.


    Args:
        host: The address of the PandA, in IP or hostname form. No port number required.
        record_prefix: The string prefix used for creation of all records.
    """
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()

    client = AsyncioClient(host)

    asyncio.run_coroutine_threadsafe(client.connect(), dispatcher.loop).result(TIMEOUT)

    all_records = asyncio.run_coroutine_threadsafe(
        create_records(client, dispatcher, record_prefix), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    asyncio.run_coroutine_threadsafe(update(client, all_records), dispatcher.loop)
    # TODO: Check return from line above periodically to see if there was an exception

    # Temporarily leave this running forever to aid debugging
    # TODO: Delete this
    softioc.interactive_ioc(globals())

    asyncio.run_coroutine_threadsafe(client.close(), dispatcher.loop).result(TIMEOUT)


def _ensure_block_number_present(block_and_field_name: str) -> str:
    """Ensure that the block instance number is always present on the end of the block
    name. If it is not present, add "1" to it.

    This works as PandA alias's the <1> suffix if there is only a single instance of a
    block

    Args:
        block_and_field_name: A string containing the block and the field name,
        e.g. "SYSTEM.TEMP_ZYNQ", or "INENC2.CLK". Must be in PandA format.

    Returns:
        str: The block and field name which will have an instance number.
        e.g. "SYSTEM1.TEMP_ZYNQ", or "INENC2.CLK".
    """
    block_name_number, field_name = block_and_field_name.split(".", maxsplit=1)
    if not block_name_number[-1].isdigit():
        block_name_number += "1"

    return f"{block_name_number}.{field_name}"


async def introspect_panda(client: AsyncioClient) -> Dict[str, _BlockAndFieldInfo]:
    """Query the PandA for all its Blocks, Fields of each Block, and Values of each Field

    Args:
        client (AsyncioClient): Client used for commuication with the PandA

    Returns:
        Dict[str, BlockAndFieldInfo]: Dictionary containing all information on
            the block
    """

    def _store_values(
        block_and_field_name: str,
        value: Union[str, List[str]],
        values: Dict[str, Dict[str, Union[str, List[str]]]],
    ):
        """Parse the data given in `block_and_field_name` and `value` into a new entry
        in the `values` dictionary"""

        block_name_number, field_name = block_and_field_name.split(".", maxsplit=1)

        block_and_field_name = _ensure_block_number_present(block_and_field_name)

        # Parse *METADATA.LABEL_<block><num> into "<block>" key and
        # "<block><num>:LABEL" value
        if block_name_number.startswith("*METADATA") and field_name.startswith(
            "LABEL_"
        ):
            _, block_name_number = field_name.split("_", maxsplit=1)
            block_and_field_name = block_name_number + ":LABEL"
        else:
            block_and_field_name = _panda_to_epics_name(block_and_field_name)

        block_name = block_name_number.rstrip(digits)

        if block_name not in values:
            values[block_name] = {}
        values[block_name][block_and_field_name] = value

    # Get the list of all blocks in the PandA
    block_dict = await client.send(GetBlockInfo())

    # Concurrently request info for all fields of all blocks
    # Note order of requests is important as it is unpacked by index below
    returned_infos = await asyncio.gather(
        *[client.send(GetFieldInfo(block)) for block in block_dict],
        client.send(GetChanges(ChangeGroup.ALL, True)),
    )

    field_infos: List[Dict[str, FieldInfo]] = returned_infos[0:-1]

    changes: Changes = returned_infos[-1]
    if changes.in_error:
        raise Exception("TODO: Some better error handling")

    # Create a dict which maps block name to all values for all instances
    # of that block (e.g. {"TTLIN" : {"TTLIN1:VAL": "1", "TTLIN2:VAL" : "5", ...} })
    values: Dict[str, Dict[str, Union[str, List[str]]]] = {}
    for block_and_field_name, value in changes.values.items():
        _store_values(block_and_field_name, value, values)

    # Parse the multiline data into the same values structure
    for block_and_field_name, multiline_value in changes.multiline_values.items():
        _store_values(block_and_field_name, multiline_value, values)

    panda_dict = {}
    for (block_name, block_info), field_info in zip(block_dict.items(), field_infos):
        panda_dict[block_name] = _BlockAndFieldInfo(
            block_info=block_info, fields=field_info, values=values[block_name]
        )

    return panda_dict


class IgnoreException(Exception):
    """Raised to indicate the current item should not be handled"""


@dataclass
class _RecordUpdater:
    """Handles Put'ing data back to the PandA when an EPICS record is updated"""

    # TODO: Argument docs

    # TODO: Is this better as an inner class inside IocRecordFactory?
    # May depend how GraphQL will handle its records and subsequent PandA updates.

    # TODO: Work out how to do this for records that aren't created
    # through _create_record_info

    record_name: str
    client: AsyncioClient
    data_type_func: Callable
    labels: Optional[List[str]] = None

    # TODO: Add type to new_val
    async def update(self, new_val):
        try:
            # If this is an enum record, retrieve the string value
            if self.labels:
                assert int(new_val) < len(
                    self.labels
                ), f"Invalid label index {new_val}, only {len(self.labels)} labels"
                val = self.labels[int(new_val)]
            else:
                # Necessary to wrap the data_type_func call in str() as we must
                # differentiate between ints and floats - some PandA fields will not
                # accept the wrong number format.
                val = str(self.data_type_func(new_val))
            panda_field = _epics_to_panda_name(self.record_name)
            await self.client.send(Put(panda_field, val))
        except IgnoreException:
            # Some values, e.g. tables, do not use this update mechanism
            logging.debug(f"Ignoring update to record {self.record_name}")
            pass
        except Exception as e:
            logging.error(f"Unable to update record {self.record_name}", exc_info=e)


class TablePacking:
    @staticmethod
    def unpack(
        row_words: int,
        table_fields: Dict[str, TableFieldDetails],
        table_data: List[str],
    ) -> List[np.ndarray]:
        """Unpacks the given `packed` data based on the fields provided.
        Returns the unpacked data in column-indexed format

        Args:
            row_words: The number of 32-bit words per row
            table_fields: The list of fields present in the packed data. Must be ordered
                in bit-ascending order (i.e. lowest bit_low field first)
            table_data: The list of data for this table, from PandA. Each item is
                expected to be the string representation of a uint32.

        Returns:
            numpy array: A list of 1-D numpy arrays, one item per field. Each item
            will have length equal to the PandA table's number of rows.
        """

        data = np.array(table_data, dtype=np.uint32)
        # Convert 1-D array into 2-D, one row element per row in the PandA table
        data = data.reshape(len(data) // row_words, row_words)
        packed = data.T

        unpacked = []
        for name, field_details in table_fields.items():
            offset = field_details.bit_low
            bit_len = field_details.bit_high - field_details.bit_low + 1

            # The word offset indicates which column this field is in
            # (column is exactly one 32-bit word)
            word_offset = offset // 32

            # bit offset is location of field inside the word
            bit_offset = offset & 0x1F

            # Mask to remove every bit that isn't in the range we want
            mask = (1 << bit_len) - 1

            # Can't use proper numpy types, that's only available in 1.21+
            val: Any = (packed[word_offset] >> bit_offset) & mask

            if field_details.subtype == "int":
                # First convert from 2's complement to offset, then add in offset.
                # TODO: Test this with extreme values - int_max, int_min, etc.
                val = np.int32((val ^ (1 << (bit_len - 1))) + (-1 << (bit_len - 1)))
            else:
                # Use shorter types, as these are used in waveform creation
                if bit_len <= 8:
                    val = np.uint8(val)
                elif bit_len <= 16:
                    val = np.uint16(val)

            unpacked.append(val)

        return unpacked

    @staticmethod
    def pack(
        row_words: int,
        table_records: Dict[str, _RecordInfo],
        table_fields: Dict[str, TableFieldDetails],
    ) -> List[str]:
        """Pack the records based on the field definitions into the format PandA expects
        for table writes.
        TODO: parameter documentation

        Returns:
            List[str]: The list of data ready to be sent to PandA
        """

        packed = None

        # Iterate over the zipped fields and their associated records to construct the
        # packed array. Note that the MODE record is at the end of the table_records,
        # and so will be ignored by zip() as it has no associated value in table_fields
        for field_details, record_info in zip(
            table_fields.values(), table_records.values()
        ):
            curr_val = record_info.record.get()
            # PandA always handles tables in uint32 format
            curr_val = np.uint32(curr_val)

            if packed is None:
                # Create 1-D array sufficiently long to exactly hold the entire table
                packed = np.zeros((len(curr_val), row_words), dtype=np.uint32)
            else:
                # TODO: Probably shouldn't be an assert, use warning/error mechanism
                assert len(packed) == len(curr_val), "Table waveform lengths mismatched"

            offset = field_details.bit_low

            # The word offset indicates which column this field is in
            # (each column is one 32-bit word)
            word_offset = offset // 32
            # bit offset is location of field inside the word
            bit_offset = offset & 0x1F

            # Slice to get the column to apply the values to.
            # bit shift the value to the relevant bits of the word
            packed[:, word_offset] |= curr_val << bit_offset

        # I'm surprised I need this assert here...
        assert packed

        # 2-D array -> 1-D array -> list[int] -> list[str]
        return [str(x) for x in packed.flatten().tolist()]


class TableModeEnum(Enum):
    """Operation modes for the MODES record on PandA table fields"""

    VIEW = 0  # Discard all EPICS record updates, process all PandA updates (default)
    EDIT = 1  # Process all EPICS record updates, discard all PandA updates
    SUBMIT = 2  # Push EPICS records to PandA, overriding current PandA data
    DISCARD = 3  # Discard all EPICS records, re-fetch from PandA


@dataclass
class _TableUpdater:
    """Class to handle updating table records, batching PUTs into the smallest number
    possible to avoid unnecessary/unwanted processing on the PandA

    `client`: The client to be used to read/write to the PandA

    `table_name`: The name of the table, in EPICS format, e.g. "SEQ1:TABLE"

    `field_info`: The TableFieldInfo structure for this table

    `table_fields`: The list of all fields in the table. Must be ordered in
    bit-ascending order

    `table_records`: The list of records that comprises this table"""

    client: AsyncioClient
    table_name: str
    field_info: TableFieldInfo
    table_fields: Dict[str, TableFieldDetails]
    table_records: Dict[str, _RecordInfo]

    def __post_init__(self):
        # Called at the end of dataclass's __init__
        # The field order will be whatever was configured in the PandA.
        # Ensure fields in bit order from lowest to highest so we can parse data
        self.table_fields = dict(
            sorted(self.table_fields.items(), key=lambda item: item[1].bit_low)
        )

    def validate(self, record: RecordWrapper, new_val) -> bool:
        """Controls whether updates to the EPICS records are processed, based on the
        value of the MODE record.

        Args:
            record (RecordWrapper): The record currently being validated
            new_val (Any): The new value attempting to be written

        Returns:
            bool: `True` to allow record update, `False` otherwise.
        """

        record_val = self._mode_record_info.record.get()

        if record_val == TableModeEnum.VIEW.value:
            logging.debug(
                f"{self.table_name} MODE record is VIEW, stopping update "
                f"to {record.name}"
            )
            return False
        elif record_val == TableModeEnum.EDIT.value:
            logging.debug(
                f"{self.table_name} MODE record is EDIT, allowing update "
                f"to {record.name}"
            )
            return True
        elif record_val == TableModeEnum.SUBMIT.value:
            # SUBMIT only present when currently writing out data to PandA.
            logging.warning(
                f"Update of record {record.name} attempted while MODE was SUBMIT."
                "New value will be discarded"
            )
            return False
        elif record_val == TableModeEnum.DISCARD.value:
            # DISCARD only present when currently overriding local data with PandA data
            logging.warning(
                f"Update of record {record.name} attempted while MODE was DISCARD."
                "New value will be discarded"
            )
        else:
            raise Exception("MODE record has unrecognised value: " + str(record_val))

        return False

    async def update(self, new_val: int):
        # This is called whenever the MODE record of the table is updated

        assert self._mode_record_info.labels

        new_label = self._mode_record_info.labels[new_val]

        if new_label == TableModeEnum.SUBMIT.name:
            # Send all EPICS data to PandA
            assert self.field_info.row_words
            packed_data = TablePacking.pack(
                self.field_info.row_words, self.table_records, self.table_fields
            )

            panda_field_name = _epics_to_panda_name(self.table_name)
            # TODO: Gotta do something in case this fails
            await self.client.send(Put(panda_field_name, packed_data))
            # Already in on_update of this record, so disable processing to
            # avoid recursion
            self._mode_record_info.record.set(TableModeEnum.VIEW.value, process=False)
        elif new_label == TableModeEnum.DISCARD.name:
            # Recreate EPICS data from PandA data
            panda_field_name = _epics_to_panda_name(self.table_name)
            panda_vals = await self.client.send(GetMultiline(f"{panda_field_name}"))

            # TODO: panda_vals is already a list, but the type system doesn't know that
            assert self.field_info.row_words
            field_data = TablePacking.unpack(
                self.field_info.row_words, self.table_fields, list(panda_vals)
            )

            for record_info, data in zip(self.table_records.values(), field_data):
                record_info.record.set(data, process=False)

            # Already in on_update of this record, so disable processing to
            # avoid recursion
            self._mode_record_info.record.set(TableModeEnum.VIEW.value, process=False)

    def update_table(self, new_values: List[str]) -> None:
        """Update the EPICS records with the given values from the PandA, depending
        on the value of the table's MODE record

        Args:
            new_values: The list of new values from the PandA
        """

        curr_mode = TableModeEnum(self._mode_record_info.record.get())

        if curr_mode == TableModeEnum.VIEW:
            assert self.field_info.row_words
            field_data = TablePacking.unpack(
                self.field_info.row_words, self.table_fields, list(new_values)
            )

            for record_info, data in zip(self.table_records.values(), field_data):
                # Must skip processing as the validate method would reject the update
                record_info.record.set(data, process=False)
        else:
            # No other mode allows PandA updates to EPICS records
            logging.warning(
                f"Update of table {self.table_name} attempted when MODE "
                "was not VIEW. New value will be discarded"
            )

    def set_mode_record(self, record_info: _RecordInfo) -> None:
        """Set the special MODE record that controls the behaviour of this table"""
        self._mode_record_info = record_info


class _HDF5RecordController:
    """Class to create and control the records that handle HDF5 processing"""

    _HDF5_PREFIX = "HDF5"
    _WAVEFORM_LENGTH = 4096  # TODO: This is used for path lengths - get from Python?

    _client: AsyncioClient

    _file_path_record: RecordWrapper
    _file_name_record: RecordWrapper
    _file_number_record: RecordWrapper
    _file_format_record: RecordWrapper
    _num_capture_record: RecordWrapper
    _flush_period_record: RecordWrapper
    _capture_control_record: RecordWrapper  # Turn capture on/off
    _arm_disarm_record: RecordWrapper  # Send Arm/Disarm

    _capture_task: Optional[asyncio.Task] = None

    def __init__(self, client: AsyncioClient):
        if importlib.util.find_spec("h5py") is None:
            logging.warning("No HDF5 support detected - skipping creating HDF5 records")
            return

        self._client = client

        # Create the records
        # Naming convention and settings (mostly) copied from FSCN2 HDF5 records
        self._file_path_record = builder.WaveformOut(
            self._HDF5_PREFIX + ":FilePath",
            length=self._WAVEFORM_LENGTH,
            FTVL="UCHAR",
            DESC="File path for HDF5 files",
            validate=self._parameter_validate,
        )

        self._file_name_record = builder.WaveformOut(
            self._HDF5_PREFIX + ":FileName",
            length=self._WAVEFORM_LENGTH,
            FTVL="UCHAR",
            DESC="File name prefix for HDF5 files",
            validate=self._parameter_validate,
        )

        self._file_number_record = builder.aOut(
            self._HDF5_PREFIX + ":FileNumber",
            validate=self._parameter_validate,
            DESC="Incrementing file number suffix",
        )

        self._file_format_record = builder.WaveformOut(
            self._HDF5_PREFIX + ":FileTemplate",
            length=64,
            FTVL="UCHAR",
            DESC="Format string used for file naming",
            validate=self._template_validate,
        )

        # Add a trailing \0 for C-based tools that may try to read the
        # entire waveform as a string
        # Awkward form of data to work around issue #39
        # https://github.com/dls-controls/pythonSoftIOC/issues/39
        # Done here, rather than inside record set, to work around issue #37
        # https://github.com/dls-controls/pythonSoftIOC/issues/37
        self._file_format_record.set(
            np.frombuffer("%s/%s_%d.h5".encode() + b"\0", dtype=np.uint8)
        )

        self._num_capture_record = builder.aOut(
            self._HDF5_PREFIX + ":NumCapture",
            initial_value=-1,  # Infinite capture
            # TODO: There's no way to stop an infinite capture?
            DESC="Number of captures to make. -1 = forever",
        )

        self._flush_period_record = builder.aOut(
            self._HDF5_PREFIX + ":FlushPeriod",
            initial_value=1.0,
            DESC="Frequency that data is flushed (seconds)",
        )

        self._capture_control_record = builder.boolOut(
            self._HDF5_PREFIX + ":Capture",
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            on_update=self._capture_on_update,
            DESC="Controls HDF5 capture",
        )

        self._arm_disarm_record = builder.boolOut(
            self._HDF5_PREFIX + ":Arm",
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            on_update=self._arm_on_update,
            DESC="Controls PandA arming",
        )

    def _parameter_validate(self, record: RecordWrapper, new_val) -> bool:
        """Control when values can be written to parameter records
        (file name etc.) based on capturing record's value"""

        if self._capture_control_record.get() == ONAM_STR:
            # Currently capturing, discard parameter updates
            logging.warning(
                "Data capture in progress. Update of HDF5 "
                f"record {record.name} discarded."
            )
            return False
        return True

    def _template_validate(self, record: RecordWrapper, new_val: np.ndarray) -> bool:
        """Validate that the FileTemplate record contains exactly the right number of
        format specifiers"""
        string_val = self._numpy_to_string(new_val)
        string_format_count = string_val.count("%s")
        number_format_count = string_val.count("%d")

        if string_format_count != 2 or number_format_count != 1:
            logging.error(
                'FileTemplate record must contain exactly 2 "%s" '
                'and exactly 1 "%d" format specifiers.'
            )
            return False

        return self._parameter_validate(record, new_val)

    async def _arm_on_update(self, new_val: int) -> None:
        """Process an update to the Arm record, to arm/disarm the PandA"""
        # TODO: Report errors here if arming/disarming fails - try-except probably
        if new_val:
            logging.debug("Arming PandA")
            await self._client.send(Arm())
        else:
            logging.debug("Disarming PandA")
            await self._client.send(Disarm())

    async def _capture_on_update(self, new_val: int) -> None:
        """Process an update to the Capture record, to start/stop recording HDF5 data"""
        if new_val:
            format_str: str = self._waveform_record_to_string(self._file_format_record)

            # Mask out the required %d format specifier while we substitute in
            # file directory and file name
            MASK_CHARS = "##########"
            masked_format_str = format_str.replace("%d", MASK_CHARS)

            (substituted_format_str,) = (
                masked_format_str
                % (
                    self._waveform_record_to_string(self._file_path_record),
                    self._waveform_record_to_string(self._file_name_record),
                ),
            )

            scheme = substituted_format_str.replace(MASK_CHARS, "%d")

            # As the capture may run forever, ensure we schedule it separately

            if self._capture_task:
                logging.error(
                    "Capture task was already running when Capture record enabled. "
                    "Killing existing task and starting new one."
                )
                self._capture_task.cancel()

            # todo: second invocation of this task never works - no new files are written...

            self._capture_task = asyncio.create_task(
                write_hdf_files(
                    self._client,
                    scheme,
                    num=self._num_capture_record.get(),
                    arm=False,  # We handle our own arming/disarming
                    flush_period=self._flush_period_record.get(),
                )
            )
        else:
            if self._capture_task:
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    logging.info(
                        "Capture task successfully cancelled when Capture "
                        "record disabled."
                    )

                self._capture_task = None

    def _waveform_record_to_string(self, record: RecordWrapper) -> str:
        """Handle converting WaveformOut record data into python string"""
        return self._numpy_to_string(record.get())

    def _numpy_to_string(self, val: np.ndarray) -> str:
        """Handle converting a numpy array of dtype=uint8 to a python string"""
        assert val.dtype == np.uint8
        # numpy gives byte stream, which we decode, and then remove the trailing \0.
        # Many tools are C-based and so will add (and expect) a null trailing byte
        return val.tobytes().decode()[:-1]


class IocRecordFactory:
    """Class to handle creating PythonSoftIOC records for a given field defined in
    a PandA"""

    _record_prefix: str
    # TODO: Not sure whether I like passing the client in, purely for child classes...
    _client: AsyncioClient

    _pos_out_row_counter: int = 0

    # List of methods in builder, used for parameter validation
    _builder_methods = [
        method
        for _, method in inspect.getmembers(builder, predicate=inspect.isfunction)
    ]

    def __init__(self, record_prefix: str, client: AsyncioClient):
        self._record_prefix = record_prefix
        self._client = client

        # Set the record prefix
        builder.SetDeviceName(self._record_prefix)

    def _process_labels(
        self, labels: Optional[List[str]], record_value: str
    ) -> Tuple[List[str], int]:
        """Find the index of `record_value` in the `labels` list, suitable for
        use in an `initial_value=` argument during record creation.
        Secondly, return a new list from the given labels that are all short
        enough to fit within EPICS 25 character label limit.

        Raises ValueError if `record_value` not found in `labels`."""
        assert labels
        assert len(labels) > 0

        if not all(len(label) < 25 for label in labels):
            logging.warning(
                "One or more labels do not fit EPICS maximum length of "
                f"25 characters. Long labels will be truncated. Labels: {labels}"
            )

        return ([label[:25] for label in labels], labels.index(record_value))

    def _check_num_values(self, values: Dict[str, str], num: int) -> None:
        """Function to check that the number of values is at least the expected amount.
        Allow extra values for future-proofing, if PandA has new fields/attributes the
        client does not know about.
        Raises AssertionError if too few values."""
        assert len(values) >= num, (
            f"Incorrect number of values, {len(values)}, expected at least {num}.\n"
            + "{values}"
        )

    def _create_record_info(
        self,
        record_name: str,
        description: Optional[str],
        record_creation_func: Callable,
        data_type_func: Callable,
        labels: List[str] = [],
        *args,
        **kwargs,
    ) -> _RecordInfo:
        """Create the record, using the given function and passing all optional
        arguments and keyword arguments, and then set the description field for the
        record.
        If the record is an Out type, a default on_update mechanism will be added. This
        can be overriden by specifying a custom "on_update=..." keyword.

        Args:
            record_name: The name this record will be created with
            description: The description for this field. This will be truncated
                to 40 characters due to EPICS limitations.
            record_creation_func: The function that will be used to create
                this record. Must be one of the builder.* functions.
            data_type_func: The function to use to convert the value returned
                from GetChanges, which will always be a string, into a type appropriate
                for the record e.g. int, float.
            labels: If the record type being created is a mbbi or mbbo
                record, provide the list of valid labels here.

        Returns:
            _RecordInfo: Class containing the created record and anything needed for
                updating the record.
        """
        assert (
            record_creation_func in self._builder_methods
        ), "Unrecognised record creation function passed to _create_record_info"

        if (
            record_creation_func == builder.mbbIn
            or record_creation_func == builder.mbbOut
        ):
            assert (
                len(labels) <= 16
            ), f"Too many labels ({len(labels)}) to create record {record_name}"

        # If there is no on_update, and the record type allows one, create it
        if "on_update" not in kwargs and record_creation_func in [
            builder.aOut,
            builder.boolOut,
            builder.mbbOut,
            builder.longOut,
            builder.stringOut,
            builder.WaveformOut,
        ]:
            # TODO: See how this interacts with the update on PandA changes thread
            # If the caller hasn't provided an update method, create it now
            record_updater = _RecordUpdater(
                record_name,
                self._client,
                data_type_func,
                labels=labels if labels else None,
            )
            update_kwarg = {"on_update": record_updater.update}
        else:
            update_kwarg = {}

        # Record description field is a maximum of 40 characters long. Ensure any string
        # is shorter than that before setting it.
        if description and len(description) > 40:
            logging.warning(
                f"Description for {record_name} longer than EPICS limit of "
                f"40 characters. It will be truncated. Description: {description}"
            )
            description = description[:40]

        kwargs.update({"DESC": description})

        record = record_creation_func(
            record_name, *labels, *args, **update_kwarg, **kwargs
        )

        record_info = _RecordInfo(
            record, data_type_func=data_type_func, labels=labels if labels else None
        )

        return record_info

    def _make_time(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[str, _RecordInfo]:
        """Make one record for the timer itself, and a sub-record for its units"""
        assert isinstance(field_info, (TimeFieldInfo, SubtypeTimeFieldInfo))
        assert field_info.units_labels

        record_dict: Dict[str, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            float,
            **kwargs,
        )

        units_record = record_name + ":UNITS"
        labels, initial_index = self._process_labels(
            field_info.units_labels, values[units_record]
        )
        record_dict[units_record] = self._create_record_info(
            units_record,
            "Units of time setting",
            builder.mbbIn,
            type(initial_index),
            labels=labels,
            initial_value=initial_index,
        )

        return record_dict

    def _make_type_time(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        """Make the records for a field of type "time" - one for the time itself, one
        for units, and one for the MIN value.
        """
        # RAW attribute ignored - EPICS should never care about it
        self._check_num_values(values, 2)
        assert isinstance(field_info, TimeFieldInfo)
        record_dict = self._make_time(
            record_name,
            field_info,
            values,
            builder.aOut,
            initial_value=float(values[record_name]),
        )

        min_record = record_name + ":MIN"
        record_dict[min_record] = self._create_record_info(
            min_record,
            "Minimum programmable time",
            builder.aIn,
            type(field_info.min),
            initial_value=field_info.min,
        )

        return record_dict

    def _make_subtype_time_param(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 2)
        return self._make_time(
            record_name,
            field_info,
            values,
            builder.aOut,
            initial_value=float(values[record_name]),
        )

    def _make_subtype_time_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 2)
        return self._make_time(
            record_name,
            field_info,
            values,
            builder.aIn,
            initial_value=float(values[record_name]),
        )

    def _make_subtype_time_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_time(record_name, field_info, values, builder.aOut)

    def _make_bit_out(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, BitOutFieldInfo)

        record_dict = {}
        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.boolIn,
            int,
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            initial_value=int(values[record_name]),
        )

        cw_rec_name = record_name + ":CAPTURE_WORD"
        record_dict[cw_rec_name] = self._create_record_info(
            cw_rec_name,
            "Name of field containing this bit",
            builder.stringIn,
            type(field_info.capture_word),
            initial_value=field_info.capture_word,
        )

        offset_rec_name = record_name + ":OFFSET"
        record_dict[offset_rec_name] = self._create_record_info(
            offset_rec_name,
            "Position of this bit in captured word",
            builder.aIn,
            type(field_info.offset),
            initial_value=field_info.offset,
        )

        return record_dict

    def _make_pos_out(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 5)
        assert isinstance(field_info, PosOutFieldInfo)
        assert field_info.capture_labels
        record_dict: Dict[str, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.aIn,
            float,
            initial_value=float(values[record_name]),
        )

        capture_rec = record_name + ":CAPTURE"
        labels, capture_index = self._process_labels(
            field_info.capture_labels, values[capture_rec]
        )
        record_dict[capture_rec] = self._create_record_info(
            capture_rec,
            "Capture options",
            builder.mbbOut,
            int,
            labels=labels,
            initial_value=capture_index,
        )

        offset_rec = record_name + ":OFFSET"
        record_dict[offset_rec] = self._create_record_info(
            offset_rec,
            "Offset",
            builder.aOut,
            int,
            initial_value=int(values[offset_rec]),
        )

        scale_rec = record_name + ":SCALE"
        record_dict[scale_rec] = self._create_record_info(
            scale_rec,
            "Scale factor",
            builder.aOut,
            float,
            initial_value=float(values[scale_rec]),
        )

        units_rec = record_name + ":UNITS"
        record_dict[units_rec] = self._create_record_info(
            units_rec,
            "Units string",
            builder.stringOut,
            str,
            initial_value=values[units_rec],
        )

        # TODO: Work out how to test SCALED, as well as all tabular,
        # records as they aren't returned at all

        # TODO: Descriptions of records created inline below

        # SCALED attribute doesn't get returned from GetChanges. Instead
        # of trying to dynamically query for it we'll just recalculate it
        scaled_rec = record_name + ":SCALED"
        scaled_calc_rec = builder.records.calc(
            scaled_rec,
            CALC="A*B + C",
            INPA=builder.CP(record_dict[record_name].record),
            INPB=builder.CP(record_dict[scale_rec].record),
            INPC=builder.CP(record_dict[offset_rec].record),
        )

        # Create the POSITIONS "table" of records. Most are aliases of the records
        # created above.
        positions_str = f"POSITIONS:{self._pos_out_row_counter}"
        builder.records.stringin(positions_str + ":NAME", VAL=record_name)

        scaled_calc_rec.add_alias(self._record_prefix + ":" + positions_str + ":VAL")

        record_dict[capture_rec].record.add_alias(
            self._record_prefix + ":" + positions_str + ":" + capture_rec.split(":")[-1]
        )
        record_dict[offset_rec].record.add_alias(
            self._record_prefix + ":" + positions_str + ":" + offset_rec.split(":")[-1]
        )
        record_dict[scale_rec].record.add_alias(
            self._record_prefix + ":" + positions_str + ":" + scale_rec.split(":")[-1]
        )
        record_dict[units_rec].record.add_alias(
            self._record_prefix + ":" + positions_str + ":" + units_rec.split(":")[-1]
        )

        self._pos_out_row_counter += 1

        return record_dict

    def _make_ext_out(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, ExtOutFieldInfo)
        assert field_info.capture_labels
        record_dict = {}
        # TODO: Check if initial_value should be set- this field appears
        # to be write only though
        record_dict[record_name] = self._create_record_info(
            record_name, field_info.description, builder.aIn, int
        )

        capture_rec = record_name + ":CAPTURE"
        labels, capture_index = self._process_labels(
            field_info.capture_labels, values[capture_rec]
        )
        record_dict[capture_rec] = self._create_record_info(
            capture_rec,
            field_info.description,
            builder.mbbOut,
            int,
            labels=labels,
            initial_value=capture_index,
        )

        return record_dict

    def _make_ext_out_bits(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, ExtOutBitsFieldInfo)
        assert field_info.bits

        record_dict = self._make_ext_out(record_name, field_info, values)

        # Create a "table" out of the items present in the list of bits

        # Identify which BITS field this is and calculate its offset - we want BITS0
        # through BITS3 to look like one continuous table from the outside, indexed
        # 0 through 127
        bits_index_str = record_name[-1]
        assert bits_index_str.isdigit()
        bits_index = int(bits_index_str)
        offset = bits_index * 32

        capture_rec = record_name + ":CAPTURE"
        capture_record_info = record_dict[capture_rec]

        # There is a single CAPTURE record which is alias'd to appear in each row.
        # This is because you can only capture a whole field's worth of bits at a time,
        # and not bits individually. When one is captured, they all are.
        for i in range(offset, offset + 32):
            capture_record_info.record.add_alias(
                f"{self._record_prefix}:BITS:{i}:CAPTURE"
            )

        # Each row of the table has a VAL and a NAME.
        for i, label in enumerate(field_info.bits):
            link = self._record_prefix + ":" + label.replace(".", ":") + " CP"
            enumerated_bits_prefix = f"BITS:{offset + i}"
            builder.records.bi(f"{enumerated_bits_prefix}:VAL", INP=link)
            # TODO: Description?

            builder.records.stringin(f"{enumerated_bits_prefix}:NAME", VAL=label)
            # TODO: Description?

        return record_dict

    def _make_bit_mux(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 2)
        assert isinstance(field_info, BitMuxFieldInfo)
        record_dict: Dict[str, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.stringOut,
            str,
            initial_value=values[record_name],
        )

        delay_rec = record_name + ":DELAY"
        record_dict[delay_rec] = self._create_record_info(
            delay_rec,
            "Clock delay on input",
            builder.aOut,
            int,
            initial_value=int(values[delay_rec]),
        )

        max_delay_rec = record_name + ":MAX_DELAY"
        record_dict[max_delay_rec] = self._create_record_info(
            max_delay_rec,
            "Maximum valid input delay",
            builder.aIn,
            type(field_info.max_delay),
            initial_value=field_info.max_delay,
        )

        return record_dict

    def _make_pos_mux(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, PosMuxFieldInfo)
        assert field_info.labels
        record_dict: Dict[str, _RecordInfo] = {}

        # This should be an mbbOut record, but there are too many posssible labels
        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.stringOut,
            str,
            initial_value=values[record_name],
        )

        return record_dict

    def _make_table(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, List[str]]
    ) -> Dict[str, _RecordInfo]:
        assert isinstance(field_info, TableFieldInfo)
        assert field_info.fields
        assert field_info.row_words

        record_dict: Dict[str, _RecordInfo] = {}

        # A mechanism to stop the EPICS on_update processing from occurring
        # for table records - their updates are handled through the _TableUpdater
        def _raiseIgnoreException(ignored):
            raise IgnoreException("This item should be ignored")

        # Create the updater
        table_updater = _TableUpdater(
            self._client,
            record_name,
            field_info,
            field_info.fields,
            record_dict,
        )

        field_data = TablePacking.unpack(
            field_info.row_words, table_updater.table_fields, values[record_name]
        )

        for (field_name, field_details), data in zip(
            table_updater.table_fields.items(), field_data
        ):
            full_name = record_name + ":" + field_name
            record_dict[full_name] = self._create_record_info(
                full_name,
                field_details.description,
                builder.WaveformOut,
                _raiseIgnoreException,
                validate=table_updater.validate,
                # FTVL keyword is inferred from dtype of the data array by pythonSoftIOC
                # Lines below work around issue #37 in PythonSoftIOC.
                # Commented out lined should be reinstated, and length + datatype lines
                # deleted, when that issue is fixed
                # NELM=field_info.max_length,
                # initial_value=data,
                length=field_info.max_length,
                datatype=data.dtype,
            )

            # This line is a workaround for issue #37 in PythonSoftIOC
            record_dict[full_name].record.set(data, process=False)

        # Create the mode record that controls when to Put back to PandA
        labels, index_value = self._process_labels(
            [
                TableModeEnum.VIEW.name,
                TableModeEnum.EDIT.name,
                TableModeEnum.SUBMIT.name,
                TableModeEnum.DISCARD.name,
            ],
            TableModeEnum.VIEW.name,  # Default state is VIEW mode
        )
        mode_record_name = record_name + ":" + "MODE"
        record_dict[mode_record_name] = self._create_record_info(
            mode_record_name,
            "Controls PandA <-> EPICS data interface",
            builder.mbbOut,
            _raiseIgnoreException,
            labels=labels,
            initial_value=index_value,
            on_update=table_updater.update,
        )

        record_dict[mode_record_name].table_updater = table_updater

        table_updater.set_mode_record(record_dict[mode_record_name])

        return record_dict

    def _make_uint(
        self,
        record_name: str,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[str, _RecordInfo]:
        assert isinstance(field_info, UintFieldInfo)

        record_dict: Dict[str, _RecordInfo] = {}
        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            int,
            **kwargs,
        )

        max_record = record_name + ":MAX"
        record_dict[max_record] = self._create_record_info(
            max_record,
            "Maximum valid value for this field",
            builder.aIn,
            type(field_info.max),
            initial_value=field_info.max,
        )

        return record_dict

    def _make_uint_param(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_uint(
            record_name,
            field_info,
            builder.aOut,
            initial_value=int(values[record_name]),
        )

    def _make_uint_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_uint(
            record_name,
            field_info,
            builder.aIn,
            initial_value=int(values[record_name]),
        )

    def _make_uint_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_uint(
            record_name, field_info, builder.aOut, always_update=True
        )

    def _make_int_param(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.aOut,
                int,
                initial_value=int(values[record_name]),
            )
        }

    def _make_int_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.aIn,
                int,
                initial_value=int(values[record_name]),
            )
        }

    def _make_int_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 0)
        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.aOut,
                int,
                always_update=True,
            )
        }

    def _make_scalar(
        self,
        record_name: str,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[str, _RecordInfo]:
        # RAW attribute ignored - EPICS should never care about it
        assert isinstance(field_info, ScalarFieldInfo)
        assert field_info.offset is not None  # offset may be 0
        record_dict: Dict[str, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name, field_info.description, record_creation_func, float, **kwargs
        )

        offset_rec = record_name + ":OFFSET"
        record_dict[offset_rec] = self._create_record_info(
            offset_rec,
            "Offset from scaled data to value",
            builder.aIn,
            type(field_info.offset),
            initial_value=field_info.offset,
        )

        scale_rec = record_name + ":SCALE"
        record_dict[scale_rec] = self._create_record_info(
            scale_rec,
            "Scaling from raw data to value",
            builder.aIn,
            type(field_info.scale),
            initial_value=field_info.scale,
        )

        units_rec = record_name + ":UNITS"
        record_dict[units_rec] = self._create_record_info(
            units_rec,
            "Units associated with value",
            builder.stringIn,
            type(field_info.units),
            initial_value=field_info.units,
        )

        return record_dict

    def _make_scalar_param(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_scalar(
            record_name,
            field_info,
            builder.aOut,
            initial_value=float(values[record_name]),
        )

    def _make_scalar_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_scalar(
            record_name,
            field_info,
            builder.aIn,
            initial_value=float(values[record_name]),
        )

    def _make_scalar_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_scalar(
            record_name, field_info, builder.aOut, always_update=True
        )

    def _make_bit(
        self,
        record_name: str,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[str, _RecordInfo]:

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                record_creation_func,
                int,
                ZNAM=ZNAM_STR,
                ONAM=ONAM_STR,
                **kwargs,
            )
        }

    def _make_bit_param(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_bit(
            record_name,
            field_info,
            builder.boolOut,
            initial_value=int(values[record_name]),
        )

    def _make_bit_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_bit(
            record_name,
            field_info,
            builder.boolIn,
            initial_value=int(values[record_name]),
        )

    def _make_bit_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_bit(
            record_name, field_info, builder.boolOut, always_update=True
        )

    def _make_action_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        raise Exception(
            "Documentation says this field isn't useful for non-write types"
        )  # TODO: Maybe just a warning? Should I still create a record here?
        # TODO: Ask whether param - action is a valid type and whether it needs a record

    def _make_action_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 0)
        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.boolOut,
                int,  # not bool, as that'll treat string "0" as true
                # TODO: See if this one even gets reported as a change
                # we might just be able to ignore it?
                ZNAM=ZNAM_STR,
                ONAM=ONAM_STR,
                always_update=True,
            )
        }

    def _make_lut(
        self,
        record_name: str,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[str, _RecordInfo]:
        # RAW attribute ignored - EPICS should never care about it
        return {
            record_name: self._create_record_info(
                record_name, field_info.description, record_creation_func, str, **kwargs
            ),
        }

    def _make_lut_param(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_lut(
            record_name,
            field_info,
            builder.stringOut,
            initial_value=values[record_name],
        )

    def _make_lut_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_lut(
            record_name,
            field_info,
            builder.stringIn,
            initial_value=values[record_name],
        )

    def _make_lut_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_lut(
            record_name, field_info, builder.stringOut, always_update=True
        )

    def _make_enum(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, EnumFieldInfo)

        labels, index_value = self._process_labels(
            field_info.labels, values[record_name]
        )

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                record_creation_func,
                int,
                labels=labels,
                initial_value=index_value,
                **kwargs,
            )
        }

    def _make_enum_param(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_enum(record_name, field_info, values, builder.mbbOut)

    def _make_enum_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_enum(record_name, field_info, values, builder.mbbIn)

    def _make_enum_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        assert isinstance(field_info, EnumFieldInfo)
        assert field_info.labels
        assert record_name not in values
        # Fake data for the default label value
        fake_vals = {record_name: field_info.labels[0]}
        return self._make_enum(
            record_name, field_info, fake_vals, builder.mbbOut, always_update=True
        )

    def create_record(
        self,
        record_name: str,
        field_info: FieldInfo,
        field_values: Dict[str, Union[str, List[str]]],
    ) -> Dict[str, _RecordInfo]:
        """Create the record (and any child records) for the PandA field specified in
        the parameters.

        Args:
            record_name: The name of the record to create, with colons separating
                words in EPICS style.
            field_info: The field info for the record being created
            field_values: The dictionary of values for the record and
                all child records. The keys are in EPICS style.

        Returns:
            Dict[str, _RecordInfo]: A dictionary of all records created and their
                associated _RecordInfo object
                #TODO: Update this once decided whether need to return POSITION, BITS,
                # and TABLE records (and any other records created in unusual ways)
        """

        try:
            key = (field_info.type, field_info.subtype)
            if key == ("table", None):
                # Table expects vals in Dict[str, List[str]]
                # TODO: Can I do this without creating a new dictionary?
                list_vals = {
                    k: v for (k, v) in field_values.items() if isinstance(v, list)
                }

                return self._make_table(record_name, field_info, list_vals)

            # All fields expect field_values to be Dict[str,str]
            # TODO: Can I do this without creating a new dictionary?
            str_vals = {k: v for (k, v) in field_values.items() if isinstance(v, str)}

            return self._field_record_mapping[key](
                self, record_name, field_info, str_vals
            )

        except KeyError as e:
            # Unrecognised type-subtype key, ignore this item. This allows the server
            # to define new types without breaking the client.
            # TODO: This catches exceptions from within mapping functions too! probably
            # need another try-except block inside this one?
            logging.warning(
                f"Exception while creating record for {record_name}, type {key}",
                exc_info=e,
            )
            return {}

    # Map a field's (type, subtype) to a function that creates and returns record(s)
    _field_record_mapping: Dict[
        Tuple[str, Optional[str]],
        Callable[
            ["IocRecordFactory", str, FieldInfo, Dict[str, str]],
            Dict[str, _RecordInfo],
        ],
    ] = {
        # Order matches that of PandA server's Field Types docs
        ("time", None): _make_type_time,
        ("bit_out", None): _make_bit_out,
        ("pos_out", None): _make_pos_out,
        ("ext_out", "timestamp"): _make_ext_out,
        ("ext_out", "samples"): _make_ext_out,
        ("ext_out", "bits"): _make_ext_out_bits,
        ("bit_mux", None): _make_bit_mux,
        ("pos_mux", None): _make_pos_mux,
        # ("table", None): _make_table, TABLE handled separately
        ("param", "uint"): _make_uint_param,
        ("read", "uint"): _make_uint_read,
        ("write", "uint"): _make_uint_write,
        ("param", "int"): _make_int_param,
        ("read", "int"): _make_int_read,
        ("write", "int"): _make_int_write,
        ("param", "scalar"): _make_scalar_param,
        ("read", "scalar"): _make_scalar_read,
        ("write", "scalar"): _make_scalar_write,
        ("param", "bit"): _make_bit_param,
        ("read", "bit"): _make_bit_read,
        ("write", "bit"): _make_bit_write,
        ("param", "action"): _make_action_write,
        ("read", "action"): _make_action_read,
        ("write", "action"): _make_action_write,
        ("param", "lut"): _make_lut_param,
        ("read", "lut"): _make_lut_read,
        ("write", "lut"): _make_lut_write,
        ("param", "enum"): _make_enum_param,
        ("read", "enum"): _make_enum_read,
        ("write", "enum"): _make_enum_write,
        ("param", "time"): _make_subtype_time_param,
        ("read", "time"): _make_subtype_time_read,
        ("write", "time"): _make_subtype_time_write,
    }

    def create_block_records(
        self, block: str, block_info: BlockInfo, block_values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        """Create the block-level records. Currently this is just the LABEL record
        for each block"""

        record_dict = {}
        for key, value in block_values.items():
            # LABEL will either get its value from the block_values if present,
            # or fall back to the block_info description field.
            if (value == "" or value is None) and block_info.description:
                value = block_info.description

            record_dict[key] = self._create_record_info(
                key, None, builder.stringIn, str, initial_value=value
            )

        if block == "PCAP":
            _HDF5RecordController(self._client)

        return record_dict

    def initialise(self, dispatcher: asyncio_dispatcher.AsyncioDispatcher) -> None:
        """Perform any final initialisation code to create the records. No new
        records may be created after this method is called.

        Args:
            dispatcher (asyncio_dispatcher.AsyncioDispatcher): The dispatcher used in
                IOC initialisation
        """
        builder.LoadDatabase()
        softioc.iocInit(dispatcher)


async def create_records(
    client: AsyncioClient,
    dispatcher: asyncio_dispatcher.AsyncioDispatcher,
    record_prefix: str,
) -> Dict[str, _RecordInfo]:
    """Query the PandA and create the relevant records based on the information
    returned"""

    panda_dict = await introspect_panda(client)

    # Dictionary containing every record of every type
    all_records: Dict[str, _RecordInfo] = {}

    record_factory = IocRecordFactory(record_prefix, client)

    # For each field in each block, create block_num records of each field
    for block, panda_info in panda_dict.items():
        block_info = panda_info.block_info
        values = panda_info.values

        # Create block-level records
        block_vals = {
            key: value
            for key, value in values.items()
            if key.endswith(":LABEL") and isinstance(value, str)
        }
        block_records = record_factory.create_block_records(
            block, block_info, block_vals
        )

        for new_record in block_records:
            if new_record in all_records:
                raise Exception(f"Duplicate record name {new_record} detected.")

        all_records.update(block_records)

        for field, field_info in panda_info.fields.items():

            for block_num in range(block_info.number):
                # For consistency in this module, always suffix the block with its
                # number. This means all records will have the block number.
                block_number = block + str(block_num + 1)

                # ":" separator for EPICS Record names, unlike PandA's "."
                record_name = block_number + ":" + field

                # Get the value of the field and all its sub-fields
                # Watch for cases where the record name is a prefix to multiple
                # unrelated fields. e.g. for record_name "INENC1:CLK",
                # values for keys "INENC1:CLK" "INENC1:CLK:DELAY" should match
                # but "INENC1:CLK_PERIOD" should not
                field_values = {
                    field: value
                    for field, value in values.items()
                    if field == record_name or field.startswith(record_name + ":")
                }

                records = record_factory.create_record(
                    record_name, field_info, field_values
                )

                for new_record in records:
                    if new_record in all_records:
                        raise Exception(f"Duplicate record name {new_record} detected.")

                all_records.update(records)

    record_factory.initialise(dispatcher)

    return all_records


async def update(client: AsyncioClient, all_records: Dict[str, _RecordInfo]):
    """Query the PandA at regular intervals for any changed fields, and update
    the records accordingly"""
    while True:
        try:
            changes = await client.send(GetChanges(ChangeGroup.ALL, True))
            if changes.in_error:
                logging.error(
                    "The following fields are reported as being in error: "
                    f"{changes.in_error}"
                )
                # TODO: Combine with getChanges error handling in introspect_panda?

            for field, value in changes.values.items():

                field = _ensure_block_number_present(field)
                # Convert PandA field name to EPICS name
                field = _panda_to_epics_name(field)
                if field not in all_records:
                    raise Exception("Unknown record returned from GetChanges")

                record_info = all_records[field]
                record = record_info.record
                if record_info.labels:
                    # Record is enum, convert string the PandA returns into an int index
                    record.set(record_info.labels.index(value))
                else:
                    record.set(record_info.data_type_func(value))

            for table_field, value_list in changes.multiline_values.items():
                # Convert PandA field name to EPICS name
                table_field = _panda_to_epics_name(table_field)
                # Tables must have a MODE record defined - use it to update the table
                table_updater = all_records[table_field + ":MODE"].table_updater
                assert (
                    table_updater
                ), f"No table updater found for {table_field} MODE record"
                table_updater.update_table(value_list)

            await asyncio.sleep(1)
        except Exception as e:
            logging.error("Exception while processing updates from PandA", exc_info=e)
            pass
