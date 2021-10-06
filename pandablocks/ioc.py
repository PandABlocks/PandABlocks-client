# Creating EPICS records directly from PandA Blocks and Fields

import asyncio
import importlib
import inspect
import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from string import digits
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
from softioc import alarm, asyncio_dispatcher, builder, softioc
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import (
    Arm,
    ChangeGroup,
    CommandException,
    Disarm,
    GetBlockInfo,
    GetChanges,
    GetFieldInfo,
    GetMultiline,
    Put,
)
from pandablocks.hdf import Pipeline, create_default_pipeline, stop_pipeline
from pandablocks.responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    BlockInfo,
    Changes,
    EndData,
    EndReason,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    FrameData,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    StartData,
    SubtypeTimeFieldInfo,
    TableFieldDetails,
    TableFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)

# Define the public API of this module
__all__ = ["create_softioc"]

TIMEOUT = 2

# Constants used in bool records
ZNAM_STR = "0"
ONAM_STR = "1"


class _InErrorException(Exception):
    """Placeholder exception to mark a field as being InError"""


# Custom type aliases and new types
ScalarRecordValue = Union[str, _InErrorException]
TableRecordValue = List[str]
RecordValue = Union[ScalarRecordValue, TableRecordValue]

EpicsName = NewType("EpicsName", str)
PandAName = NewType("PandAName", str)


@dataclass
class _BlockAndFieldInfo:
    """Contains all available information for a Block, including Fields and all the
    Values for `block_info.number` instances of the Fields."""

    block_info: BlockInfo
    fields: Dict[str, FieldInfo]
    values: Dict[EpicsName, RecordValue]
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
    `table_updater`: Class instance that managed updating table records. Only present
        on the MODE record of a table."""

    record: RecordWrapper
    data_type_func: Callable
    labels: Optional[List[str]] = None
    table_updater: Optional["_TableUpdater"] = None


def _panda_to_epics_name(field_name: PandAName) -> EpicsName:
    """Convert PandA naming convention to EPICS convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return EpicsName(field_name.replace(".", ":"))


def _epics_to_panda_name(field_name: EpicsName) -> PandAName:
    """Convert EPICS naming convention to PandA convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return PandAName(field_name.replace(":", "."))


async def _create_softioc(
    client: AsyncioClient,
    record_prefix: str,
    dispatcher: asyncio_dispatcher.AsyncioDispatcher,
):
    """Asynchronous wrapper for IOC creation"""
    await client.connect()
    all_records = await create_records(client, dispatcher, record_prefix)
    asyncio.create_task(update(client, all_records, 1))


# TODO: Consider https://github.com/dls-controls/pythonSoftIOC/issues/43
# which means we might need to ensure ALL Out records have an initial_value...
def create_softioc(host: str, record_prefix: str) -> None:
    """Create a PythonSoftIOC from fields and attributes of a PandA.

    This function will introspect a PandA for all defined Blocks, Fields of each Block,
    and Attributes of each Field, and create appropriate EPICS records for each.


    Args:
        host: The address of the PandA, in IP or hostname form. No port number required.
        record_prefix: The string prefix used for creation of all records.
    """
    # TODO: This needs to read/take in a YAML configuration file, for various aspects
    # e.g. the update() wait time between calling GetChanges
    try:
        dispatcher = asyncio_dispatcher.AsyncioDispatcher()
        client = AsyncioClient(host)
        asyncio.run_coroutine_threadsafe(
            _create_softioc(client, record_prefix, dispatcher), dispatcher.loop
        ).result()

        # Must leave this blocking line here, in the main thread, not in the
        # dispatcher's loop or it'll block every async process in this module
        softioc.interactive_ioc(globals())
    except Exception:
        logging.exception("Exception while initializing softioc")
    finally:
        # Client was connected in the _create_softioc method
        asyncio.run_coroutine_threadsafe(client.close(), dispatcher.loop).result()


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
        value: RecordValue,
        values: Dict[str, Dict[EpicsName, RecordValue]],
    ) -> None:
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
            block_and_field_name = EpicsName(block_name_number + ":LABEL")
        else:
            block_and_field_name = _panda_to_epics_name(PandAName(block_and_field_name))

        block_name = block_name_number.rstrip(digits)

        if block_name not in values:
            values[block_name] = {}
        if block_and_field_name in values[block_name]:
            logging.error(
                f"Duplicate values for {block_and_field_name} detected."
                " Overriding existing value."
            )
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

    # Create a dict which maps block name to all values for all instances
    # of that block (e.g. {"TTLIN" : {"TTLIN1:VAL": "1", "TTLIN2:VAL" : "5", ...} })
    values: Dict[str, Dict[EpicsName, RecordValue]] = {}
    for block_and_field_name, value in changes.values.items():
        _store_values(block_and_field_name, value, values)

    # Parse the multiline data into the same values structure
    for block_and_field_name, multiline_value in changes.multiline_values.items():
        _store_values(block_and_field_name, multiline_value, values)

    # Note any in_error fields so we can later set their records to a non-zero severity
    for block_and_field_name in changes.in_error:
        logging.error(f"PandA reports fields in error: {changes.in_error}")
        _store_values(
            block_and_field_name, _InErrorException(block_and_field_name), values
        )

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
    """Handles Put'ing data back to the PandA when an EPICS record is updated

    Args:
        record_name: The name of the record, without the namespace
        client: The client used to send data to PandA
        data_type_func: Function to convert the new value to the format PandA expects
        labels: If the record is an enum type, provide the list of labels
        previous_value: The initial value of the record. During record updates this
            will be used to restore the previous value of the record if a Put fails.
    """

    record_name: EpicsName
    client: AsyncioClient
    data_type_func: Callable
    labels: Optional[List[str]] = None
    previous_value: Any = None

    # The incoming value's type depends on the record. Ensure you always cast it.
    async def update(self, new_val: Any):
        logging.debug(f"Updating record {self.record_name} with value {new_val}")
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
            self.previous_value = val

        except IgnoreException:
            # Thrown by some data_type_func calls.
            # Some values, e.g. tables, do not use this update mechanism
            logging.debug(f"Ignoring update to record {self.record_name}")
            pass
        except Exception as e:
            logging.error(
                f"Unable to Put record {self.record_name}, value {new_val}, to PandA",
                exc_info=e,
            )
            if self._record:
                logging.debug(f"Restoring previous value to record {self.record_name}")
                self._record.set(self.previous_value, process=False)
            else:
                logging.warning(
                    f"No record found when updating {self.record_name}, "
                    "unable to roll back value"
                )

    def add_record(self, record: RecordWrapper) -> None:
        """Provide the record, used for rolling back data if a Put fails."""
        self._record = record


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
        table_records: Dict[EpicsName, _RecordInfo],
        table_fields: Dict[str, TableFieldDetails],
    ) -> List[str]:
        """Pack the records based on the field definitions into the format PandA expects
        for table writes.
        Args:
            row_words: The number of 32-bit words per row
            table_records: The list of fields and their associated _RecordInfo
                structure, used to access the value of each record.
            table_fields: The list of fields present in the packed data. Must be ordered
                in bit-ascending order (i.e. lowest bit_low field first)

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
                assert len(packed) == len(
                    curr_val
                ), "Table waveform lengths mismatched, cannot pack data"

            offset = field_details.bit_low

            # The word offset indicates which column this field is in
            # (each column is one 32-bit word)
            word_offset = offset // 32
            # bit offset is location of field inside the word
            bit_offset = offset & 0x1F

            # Slice to get the column to apply the values to.
            # bit shift the value to the relevant bits of the word
            packed[:, word_offset] |= curr_val << bit_offset

        assert isinstance(packed, np.ndarray)  # Squash mypy warning

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

    Args:
        client: The client to be used to read/write to the PandA
        table_name: The name of the table, in EPICS format, e.g. "SEQ1:TABLE"
        field_info: The TableFieldInfo structure for this table
        table_fields: The list of all fields in the table. During initialization they
            will be sorted into bit-ascending order
        table_records: The list of records that comprises this table
        previous_value: The initial value of the table. During record updates this
            will be used to restore the previous value of the record if a Put fails."""

    client: AsyncioClient
    table_name: EpicsName
    field_info: TableFieldInfo
    table_fields: Dict[str, TableFieldDetails]
    table_records: Dict[EpicsName, _RecordInfo]
    previous_value: List[str]

    def __post_init__(self):
        # Called at the end of dataclass's __init__
        # The field order will be whatever was configured in the PandA.
        # Ensure fields in bit order from lowest to highest so we can parse data
        self.table_fields = dict(
            sorted(self.table_fields.items(), key=lambda item: item[1].bit_low)
        )

    def validate_waveform(self, record: RecordWrapper, new_val) -> bool:
        """Controls whether updates to the waveform records are processed, based on the
        value of the MODE record.

        Args:
            record: The record currently being validated
            new_val: The new value attempting to be written

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
            return False
        else:
            logging.error("MODE record has unknown value: " + str(record_val))
            # In case it isn't already, set an alarm state on the record
            self._mode_record_info.record.set_alarm(
                alarm.INVALID_ALARM, alarm.UDF_ALARM
            )
            return False

    async def update_waveform(self, new_val: int, record_name: str) -> None:
        """Handles updates to a specific waveform record to update its associated
        scalar value record"""
        self._update_scalar(record_name)

    async def update_mode(self, new_val: int):
        """Controls the behaviour when the MODE record is updated.
        Controls Put'ting data back to PandA, or re-Get'ting data from Panda
        and replacing record data."""

        assert self._mode_record_info.labels

        new_label = self._mode_record_info.labels[new_val]

        if new_label == TableModeEnum.SUBMIT.name:
            try:
                # Send all EPICS data to PandA
                assert self.field_info.row_words
                packed_data = TablePacking.pack(
                    self.field_info.row_words, self.table_records, self.table_fields
                )

                panda_field_name = _epics_to_panda_name(self.table_name)
                await self.client.send(Put(panda_field_name, packed_data))
                self.previous_value = packed_data

            except Exception as e:
                logging.error(
                    f"Unable to Put record {self.table_name}, value {packed_data},"
                    "to PandA",
                    exc_info=e,
                )

                # Reset value of all table records to last known good values
                assert self.field_info.row_words
                field_data = TablePacking.unpack(
                    self.field_info.row_words, self.table_fields, self.previous_value
                )
                for record_info, data in zip(self.table_records.values(), field_data):
                    record_info.record.set(data, process=False)

            # Already in on_update of this record, so disable processing to
            # avoid recursion
            self._mode_record_info.record.set(TableModeEnum.VIEW.value, process=False)

        elif new_label == TableModeEnum.DISCARD.name:
            # Recreate EPICS data from PandA data
            panda_field_name = _epics_to_panda_name(self.table_name)
            panda_vals = await self.client.send(GetMultiline(f"{panda_field_name}"))

            assert self.field_info.row_words
            field_data = TablePacking.unpack(
                self.field_info.row_words, self.table_fields, panda_vals
            )

            for record_info, data in zip(self.table_records.values(), field_data):
                record_info.record.set(data, process=False)

            # Already in on_update of this record, so disable processing to
            # avoid recursion
            self._mode_record_info.record.set(TableModeEnum.VIEW.value, process=False)

    def update_table(self, new_values: List[str]) -> None:
        """Update the waveform records with the given values from the PandA, depending
        on the value of the table's MODE record.
        Note: This is NOT a method called through a record's `on_update`.

        Args:
            new_values: The list of new values from the PandA
        """

        curr_mode = TableModeEnum(self._mode_record_info.record.get())

        if curr_mode == TableModeEnum.VIEW:
            assert self.field_info.row_words
            field_data = TablePacking.unpack(
                self.field_info.row_words, self.table_fields, list(new_values)
            )

            table_field_records = self._get_table_field_records()
            for record_info, data in zip(table_field_records.values(), field_data):
                # Must skip processing as the validate method would reject the update
                record_info.record.set(data, process=False)
                self._update_scalar(record_info.record.name)
        else:
            # No other mode allows PandA updates to EPICS records
            logging.warning(
                f"Update of table {self.table_name} attempted when MODE "
                "was not VIEW. New value will be discarded"
            )

    async def update_index(self, new_val) -> None:
        """Update the SCALAR record for every column in the table based on the new
        index and/or new waveform data."""
        for record_info in self._get_table_field_records().values():
            self._update_scalar(record_info.record.name)

    def _update_scalar(self, waveform_record_name: str) -> None:
        """Update the column's SCALAR record based on the new index and/or new waveform
        data.

        Args:
            waveform_record_name: The name of the column record including leading
            namespace, e.g. "<namespace>:SEQ1:TABLE:POSITION"
        """

        # Remove namespace from record name
        _, waveform_record_name = waveform_record_name.split(":", maxsplit=1)

        waveform_record = self.table_records[EpicsName(waveform_record_name)].record
        waveform_data = waveform_record.get()

        scalar_record = self.table_records[
            EpicsName(waveform_record_name + ":SCALAR")
        ].record

        index_record_name = waveform_record_name.rsplit(":", maxsplit=1)[0] + ":INDEX"
        index_record = self.table_records[EpicsName(index_record_name)].record
        index = index_record.get()

        try:
            scalar_val = waveform_data[index]
            sev = alarm.NO_ALARM
        except IndexError as e:
            logging.warning(
                f"Index {index} of record {waveform_record_name} is out of bounds.",
                exc_info=e,
            )
            scalar_val = 0
            sev = alarm.INVALID_ALARM

        # alarm value is ignored if severity = NO_ALARM. Softioc also defaults
        # alarm value to UDF_ALARM, but I'm specifying it for clarity.
        scalar_record.set(scalar_val, severity=sev, alarm=alarm.UDF_ALARM)

    def _get_table_field_records(self) -> Dict[EpicsName, _RecordInfo]:
        """Filter the list of all table records to only return those of the table
        fields, removing all SCALAR, INDEX, and MODE records."""
        return {
            EpicsName(k): v
            for k, v in self.table_records.items()
            if not k.endswith(("SCALAR", "INDEX", "MODE"))
        }

    def set_mode_record_info(self, record_info: _RecordInfo) -> None:
        """Set the special MODE record that controls the behaviour of this table"""
        self._mode_record_info = record_info


class _HDF5RecordController:
    """Class to create and control the records that handle HDF5 processing"""

    _HDF5_PREFIX = "HDF5"

    _client: AsyncioClient

    _file_path_record: RecordWrapper
    _file_name_record: RecordWrapper
    _file_number_record: RecordWrapper
    _file_format_record: RecordWrapper
    _num_capture_record: RecordWrapper
    _flush_period_record: RecordWrapper
    _capture_control_record: RecordWrapper  # Turn capture on/off
    _arm_disarm_record: RecordWrapper  # Send Arm/Disarm
    _status_message_record: RecordWrapper  # Reports status and error messages
    _currently_capturing_record: RecordWrapper  # If HDF5 file currently being written

    _capture_enabled_event: asyncio.Event = asyncio.Event()
    _write_hdf5_file_task: Optional[asyncio.Task] = None

    def __init__(self, client: AsyncioClient, record_prefix: str):
        if importlib.util.find_spec("h5py") is None:
            logging.warning("No HDF5 support detected - skipping creating HDF5 records")
            return

        self._client = client

        path_length = os.pathconf("/", "PC_PATH_MAX")
        filename_length = os.pathconf("/", "PC_NAME_MAX")

        # Create the records, including an uppercase alias for each
        # Naming convention and settings (mostly) copied from FSCN2 HDF5 records
        file_path_record_name = self._HDF5_PREFIX + ":FilePath"
        self._file_path_record = builder.WaveformOut(
            file_path_record_name,
            length=path_length,
            FTVL="UCHAR",
            DESC="File path for HDF5 files",
            validate=self._parameter_validate,
        )
        self._file_path_record.add_alias(
            record_prefix + ":" + file_path_record_name.upper()
        )

        file_name_record_name = self._HDF5_PREFIX + ":FileName"
        self._file_name_record = builder.WaveformOut(
            file_name_record_name,
            length=filename_length,
            FTVL="UCHAR",
            DESC="File name prefix for HDF5 files",
            validate=self._parameter_validate,
        )
        self._file_name_record.add_alias(
            record_prefix + ":" + file_name_record_name.upper()
        )

        file_number_record_name = self._HDF5_PREFIX + ":FileNumber"
        self._file_number_record = builder.aOut(
            file_number_record_name,
            DESC="Incrementing file number suffix",
            validate=self._parameter_validate,
        )
        self._file_number_record.add_alias(
            record_prefix + ":" + file_number_record_name.upper()
        )

        file_format_record_name = self._HDF5_PREFIX + ":FileTemplate"
        self._file_format_record = builder.WaveformOut(
            file_format_record_name,
            length=64,
            FTVL="UCHAR",
            DESC="Format string used for file naming",
            validate=self._template_validate,
        )
        self._file_format_record.add_alias(
            record_prefix + ":" + file_format_record_name.upper()
        )

        # Add a trailing \0 for C-based tools that may try to read the
        # entire waveform as a string
        # Awkward form of data to work around issue #39
        # https://github.com/dls-controls/pythonSoftIOC/issues/39
        # Done here, rather than inside record creation above, to work around issue #37
        # https://github.com/dls-controls/pythonSoftIOC/issues/37
        self._file_format_record.set(
            np.frombuffer("%s/%s_%d.h5".encode() + b"\0", dtype=np.uint8)
        )

        num_capture_record_name = self._HDF5_PREFIX + ":NumCapture"
        self._num_capture_record = builder.longOut(
            num_capture_record_name,
            initial_value=0,  # Infinite capture
            DESC="Num frames to capture. 0 = infinite",
            DRVL=0,
            validate=self._parameter_validate,
        )
        self._num_capture_record.add_alias(
            record_prefix + ":" + num_capture_record_name.upper()
        )

        flush_period_record_name = self._HDF5_PREFIX + ":FlushPeriod"
        self._flush_period_record = builder.aOut(
            flush_period_record_name,
            initial_value=1.0,
            DESC="Frequency that data is flushed (seconds)",
        )
        self._flush_period_record.add_alias(
            record_prefix + ":" + flush_period_record_name.upper()
        )

        capture_control_record_name = self._HDF5_PREFIX + ":Capture"
        self._capture_control_record = builder.boolOut(
            capture_control_record_name,
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            initial_value=0,  # PythonSoftIOC issue #43
            on_update=self._capture_on_update,
            validate=self._capture_validate,
            DESC="Start/stop HDF5 capture",
        )
        self._capture_control_record.add_alias(
            record_prefix + ":" + capture_control_record_name.upper()
        )

        arm_disarm_record_name = self._HDF5_PREFIX + ":Arm"
        self._arm_disarm_record = builder.boolOut(
            arm_disarm_record_name,
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            initial_value=0,  # PythonSoftIOC issue #43
            on_update=self._arm_on_update,
            DESC="Arm/Disarm the PandA",
        )
        self._arm_disarm_record.add_alias(
            record_prefix + ":" + arm_disarm_record_name.upper()
        )

        # TODO: Set this record somewhere
        status_message_record_name = self._HDF5_PREFIX + ":Status"
        self._status_message_record = builder.stringIn(
            status_message_record_name,
            initial_value="OK",
            DESC="Reports current status of HDF5 capture",
        )
        self._status_message_record.add_alias(
            record_prefix + ":" + status_message_record_name.upper()
        )

        # TODO: Set this record somewhere
        currently_capturing_record_name = self._HDF5_PREFIX + ":Capturing"
        self._currently_capturing_record = builder.boolIn(
            currently_capturing_record_name,
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            initial_value=0,  # PythonSoftIOC issue #43
            DESC="If HDF5 file is currently being written",
        )
        self._capture_control_record.add_alias(
            record_prefix + ":" + currently_capturing_record_name.upper()
        )

        # Spawn the task that will handle HDF5 file writing
        # loop = asyncio.new_event_loop()
        self._write_hdf5_file_task = threading.Thread(target=self._handle_hdf5_data)
        self._write_hdf5_file_task.start()
        # self._write_hdf5_file_task = loop.call_soon_threadsafe(self._handle_hdf5_data())
        # self._write_hdf5_file_task = asyncio.create_task(self._handle_hdf5_data())
        print("HERE")

    def _parameter_validate(self, record: RecordWrapper, new_val) -> bool:
        """Control when values can be written to parameter records
        (file name etc.) based on capturing record's value"""
        logging.debug(f"Validating record {record.name} value {new_val}")
        if self._capture_control_record.get():
            # Currently capturing, discard parameter updates
            logging.warning(
                "Data capture in progress. Update of HDF5 "
                f"record {record.name} with new value {new_val} discarded."
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
        logging.debug(f"Entering HDF5:Arm record on_update method, value {new_val}")
        try:
            if new_val:
                logging.debug("Arming PandA")
                await self._client.send(Arm())
                # if self._write_hdf5_file_task:
                #     logging.info(
                #         "Capture task was already running when Arm record enabled."
                #     )
                # else:
                #     # Start a task on a different thread to process the data
                #     logging.info("Starting capture task in response to Arm request")

            else:
                logging.debug("Disarming PandA")
                await self._client.send(Disarm())

        except CommandException:
            logging.exception("Failure arming/disarming PandA")

        except Exception:
            logging.exception("Unexpected exception when arming/disarming PandA")

    async def _handle_hdf5_data(self):
        """Handles writing HDF5 data from the PandA to file, based on configuration
        in the various HDF5 records.
        This task should exist for the lifetime of the program."""

        while True:
            # Wait for Capture to be enabled
            await self._capture_enabled_event.wait()

            # Keep the start data around to compare against, if capture is
            # enabled/disabled to know if we can keep using the same file
            start_data: Optional[StartData] = None
            captured_frames: int = 0

            try:
                # Store the max number
                # TODO: Could just re-get this number on each loop? Means users could
                # edit it whenever they want
                num_to_capture: int = self._num_capture_record.get()
                pipeline: List[Pipeline] = create_default_pipeline(self._get_scheme())
                flush_period: float = self._flush_period_record.get()

                async for data in self._client.data(
                    scaled=False, flush_period=flush_period
                ):

                    if isinstance(data, StartData):
                        if start_data and data != start_data:
                            # PandA was disarmed, had config changed, and rearmed.
                            # Cannot process to the same file with different start data.
                            logging.error(
                                "New start data detected, differs from previous start "
                                "data for this file. Aborting HDF5 data capture."
                            )
                            self._status_message_record.set(
                                "Mismatched StartData packet for file"
                            )
                            # Disable this task's processing
                            # TODO: This feels weird but we cannot rely on on_update()
                            # as it's not guaranteed to run before this thread gets
                            # to the Event check at the top
                            self._capture_control_record.set(0, process=False)
                            self._capture_enabled_event.clear()
                            pipeline[0].queue.put_nowait(
                                EndData(captured_frames, EndReason.START_DATA_MISMATCH)
                            )
                            break
                        if start_data is None:
                            start_data = data
                            # Only pass StartData to pipeline if it wasn't already open
                            # - if it was there will already be an in-progress HDF file
                            pipeline[0].queue.put_nowait(data)

                    elif isinstance(data, FrameData):
                        pipeline[0].queue.put_nowait(data)
                        captured_frames += 1

                        if captured_frames == num_to_capture:
                            # Reached configured capture limit, stop the file
                            pipeline[0].queue.put_nowait(
                                EndData(captured_frames, EndReason.OK)
                            )
                            # Disable this task's processing
                            # TODO: This feels weird but we cannot rely on on_update()
                            # as it's not guaranteed to run before this thread gets
                            # to the Event check at the top
                            self._capture_control_record.set(0, process=False)
                            self._capture_enabled_event.clear()
                            pipeline[0].queue.put_nowait(
                                EndData(captured_frames, EndReason.START_DATA_MISMATCH)
                            )
                            break
                    # Ignore EndData - handle terminating capture with the Capture
                    # record or when we capture the requested number of frames

                    # Check whether to keep running. At end of loop so we finish
                    # processing the current data packet before exiting.
                    if not self._capture_enabled_event.is_set():
                        # Capture has been disabled, stop processing.
                        logging.info("Capture disabled, closing HDF5 file")
                        self._status_message_record.set(
                            "Capturing finished, file closed"
                        )
                        pipeline[0].queue.put_nowait(
                            EndData(captured_frames, EndReason.OK)
                        )
                        break

            except asyncio.CancelledError:
                logging.info("HDF 5 data capture told to finish")
                pipeline[0].queue.put_nowait(EndData(captured_frames, EndReason.OK))
                raise

            except Exception:
                logging.exception(
                    "HDF5 data capture terminated due to unexpected error"
                )
                pipeline[0].queue.put_nowait(
                    EndData(captured_frames, EndReason.UNKNOWN_EXCEPTION)
                )
            finally:
                stop_pipeline(pipeline)
                pass

    def _get_scheme(self) -> str:
        """Create the scheme for the HDF5 file from the relevant record in the format
        expected by the HDF module"""
        format_str = self._waveform_record_to_string(self._file_format_record)

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

        return substituted_format_str.replace(MASK_CHARS, "%d")

    async def _capture_on_update(self, new_val: int) -> None:
        """Process an update to the Capture record, to start/stop recording HDF5 data"""
        logging.debug(f"Entering HDF5:Capture record on_update method, value {new_val}")
        if new_val:
            self._capture_enabled_event.set()  # Start the writing task
        else:
            self._capture_enabled_event.clear()  # Abort any HDF5 file writing

    def _capture_validate(self, record: RecordWrapper, new_val: int) -> bool:
        """Check the required records have been set before allowing Capture=1"""
        if new_val:
            try:
                self._get_scheme()
            except ValueError:
                logging.exception("At least 1 required record had no value")
                return False

        return True

    def _waveform_record_to_string(self, record: RecordWrapper) -> str:
        """Handle converting WaveformOut record data into python string"""
        val = record.get()
        if val is None:
            raise ValueError(f"Record {record.name} had no value when one is required.")
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
        self, labels: Optional[List[str]], record_value: ScalarRecordValue
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

        # Most likely time we'll see an error is when PandA hardware has set invalid
        # enum value. No logging as already logged when seen from GetChanges.
        if isinstance(record_value, _InErrorException):
            index = 0
        else:
            index = labels.index(record_value)

        return ([label[:25] for label in labels], index)

    def _check_num_values(
        self, values: Dict[EpicsName, ScalarRecordValue], num: int
    ) -> None:
        """Function to check that the number of values is at least the expected amount.
        Allow extra values for future-proofing, if PandA has new fields/attributes the
        client does not know about.
        Raises AssertionError if too few values."""
        assert len(values) >= num, (
            f"Incorrect number of values, {len(values)}, expected at least {num}.\n"
            + f"{values}"
        )

    def _create_record_info(
        self,
        record_name: EpicsName,
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
            *args: Additional arguments that will be passed through to the
                `record_creation_func`
            **kwargs: Additional keyword arguments that will be examined and possibly
                modified for various reasons, and then passed to the
                `record_creation_func`

        Returns:
            _RecordInfo: Class containing the created record and anything needed for
                updating the record.
        """
        extra_kwargs: Dict[str, Any] = {}

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

        # Check the initial value is valid. If it is, apply data type conversion
        # otherwise mark the record as in error
        # Note that many already have correct type, this mostly applies to values
        # that are being sent as strings to analog or long records as the values are
        # returned as strings from Changes command.
        if "initial_value" in kwargs:
            initial_value = kwargs["initial_value"]
            if isinstance(initial_value, _InErrorException):
                logging.warning(
                    f"Marking record {record_name} as invalid due to error from PandA"
                )
                extra_kwargs.update(
                    {"severity": alarm.INVALID_ALARM, "alarm": alarm.UDF_ALARM}
                )
            elif isinstance(initial_value, str):
                kwargs["initial_value"] = data_type_func(initial_value)

        # If there is no on_update, and the record type allows one, create it
        if (
            "on_update" not in kwargs
            and "on_update_name" not in kwargs
            and record_creation_func
            in [
                builder.aOut,
                builder.boolOut,
                builder.mbbOut,
                builder.longOut,
                builder.stringOut,
                builder.WaveformOut,
            ]
        ):
            # If the caller hasn't provided an update method, create it now
            record_updater = _RecordUpdater(
                record_name,
                self._client,
                data_type_func,
                labels if labels else None,
                kwargs["initial_value"] if "initial_value" in kwargs else None,
            )
            extra_kwargs.update({"on_update": record_updater.update})

        # Disable Puts to all In records (unless explicilty enabled)
        if "DISP" not in kwargs and record_creation_func in [
            builder.aIn,
            builder.boolIn,
            builder.mbbIn,
            builder.longIn,
            builder.stringIn,
            builder.WaveformIn,
        ]:
            extra_kwargs.update({"DISP": 1})

        # Record description field is a maximum of 40 characters long. Ensure any string
        # is shorter than that before setting it.
        if description and len(description) > 40:
            # As per Tom Cobb, it's unlikely the descriptions will ever be truncated so
            # we'll hide this message in debug level logging only
            logging.info(
                f"Description for {record_name} longer than EPICS limit of "
                f"40 characters. It will be truncated. Description: {description}"
            )
            description = description[:40]

        extra_kwargs.update({"DESC": description})

        record = record_creation_func(
            record_name, *labels, *args, **extra_kwargs, **kwargs
        )

        record_info = _RecordInfo(
            record, data_type_func=data_type_func, labels=labels if labels else None
        )

        return record_info

    def _make_time(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[EpicsName, _RecordInfo]:
        """Make one record for the timer itself, and a sub-record for its units"""
        assert isinstance(field_info, (TimeFieldInfo, SubtypeTimeFieldInfo))
        assert field_info.units_labels

        record_dict: Dict[EpicsName, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            float,
            **kwargs,
        )

        units_record = EpicsName(record_name + ":UNITS")
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
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
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
            initial_value=values[record_name],
        )

        min_record = EpicsName(record_name + ":MIN")
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
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 2)
        return self._make_time(
            record_name,
            field_info,
            values,
            builder.aOut,
            initial_value=values[record_name],
        )

    def _make_subtype_time_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 2)
        return self._make_time(
            record_name,
            field_info,
            values,
            builder.aIn,
            initial_value=values[record_name],
        )

    def _make_subtype_time_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_time(record_name, field_info, values, builder.aOut)

    def _make_bit_out(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
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
            initial_value=values[record_name],
        )

        cw_rec_name = EpicsName(record_name + ":CAPTURE_WORD")
        record_dict[cw_rec_name] = self._create_record_info(
            cw_rec_name,
            "Name of field containing this bit",
            builder.stringIn,
            type(field_info.capture_word),
            initial_value=field_info.capture_word,
        )

        offset_rec_name = EpicsName(record_name + ":OFFSET")
        record_dict[offset_rec_name] = self._create_record_info(
            offset_rec_name,
            "Position of this bit in captured word",
            builder.longIn,
            type(field_info.offset),
            initial_value=field_info.offset,
        )

        return record_dict

    def _make_pos_out(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 5)
        assert isinstance(field_info, PosOutFieldInfo)
        assert field_info.capture_labels
        record_dict: Dict[EpicsName, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.longIn,
            int,
            initial_value=values[record_name],
        )

        capture_rec = EpicsName(record_name + ":CAPTURE")
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

        offset_rec = EpicsName(record_name + ":OFFSET")
        record_dict[offset_rec] = self._create_record_info(
            offset_rec,
            "Offset",
            builder.aOut,
            float,
            initial_value=values[offset_rec],
        )

        scale_rec = EpicsName(record_name + ":SCALE")
        record_dict[scale_rec] = self._create_record_info(
            scale_rec,
            "Scale factor",
            builder.aOut,
            float,
            initial_value=values[scale_rec],
        )

        units_rec = EpicsName(record_name + ":UNITS")
        record_dict[units_rec] = self._create_record_info(
            units_rec,
            "Units string",
            builder.stringOut,
            str,
            initial_value=values[units_rec],
        )

        # TODO: Work out how to test SCALED, as well as all tabular,
        # records as they aren't returned at all

        # SCALED attribute doesn't get returned from GetChanges. Instead
        # of trying to dynamically query for it we'll just recalculate it
        scaled_rec = record_name + ":SCALED"
        scaled_calc_rec = builder.records.calc(
            scaled_rec,
            CALC="A*B + C",
            INPA=builder.CP(record_dict[record_name].record),
            INPB=builder.CP(record_dict[scale_rec].record),
            INPC=builder.CP(record_dict[offset_rec].record),
            DESC="Value with scaling applied",
            DISP=1,
        )

        # Create the POSITIONS "table" of records. Most are aliases of the records
        # created above.
        positions_str = f"POSITIONS:{self._pos_out_row_counter}"
        builder.records.stringin(
            positions_str + ":NAME",
            VAL=record_name,
            DESC="Table of configured positional outputs",
            DISP=1,
        ),

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
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, ExtOutFieldInfo)
        assert field_info.capture_labels
        record_dict = {}
        # TODO: Check if initial_value should be set- this field appears
        # to be write only though
        record_dict[record_name] = self._create_record_info(
            record_name, field_info.description, builder.aIn, int
        )

        capture_rec = EpicsName(record_name + ":CAPTURE")
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
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
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

        capture_rec = EpicsName(record_name + ":CAPTURE")
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
            # TODO: This field doesn't seem to have any value...
            builder.records.bi(
                f"{enumerated_bits_prefix}:VAL",
                INP=link,
                DESC="Value of field connected to this BIT",
                DISP=1,
            )

            builder.records.stringin(
                f"{enumerated_bits_prefix}:NAME",
                VAL=label,
                DESC="Name of field connected to this BIT",
                DISP=1,
            )

        return record_dict

    def _make_bit_mux(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 2)
        assert isinstance(field_info, BitMuxFieldInfo)
        record_dict: Dict[EpicsName, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.stringOut,
            str,
            initial_value=values[record_name],
        )

        delay_rec = EpicsName(record_name + ":DELAY")
        record_dict[delay_rec] = self._create_record_info(
            delay_rec,
            "Clock delay on input",
            builder.longOut,
            int,
            initial_value=values[delay_rec],
        )

        max_delay_rec = EpicsName(record_name + ":MAX_DELAY")
        record_dict[max_delay_rec] = self._create_record_info(
            max_delay_rec,
            "Maximum valid input delay",
            builder.longIn,
            type(field_info.max_delay),
            initial_value=field_info.max_delay,
        )

        return record_dict

    def _make_pos_mux(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        @dataclass
        class PosMuxValidator:
            """Validate that a given string is a valid label for a PosMux field"""

            labels: List[str]

            def validate(self, record: RecordWrapper, new_val: str):
                if new_val in self.labels:
                    return True
                logging.warning(f"Value {new_val} not valid for record {record.name}")
                return False

        self._check_num_values(values, 1)
        assert isinstance(field_info, PosMuxFieldInfo)
        assert field_info.labels

        record_dict: Dict[EpicsName, _RecordInfo] = {}

        # This should be an mbbOut record, but there are too many posssible labels
        # TODO: Add a :LABELS record so users can get valid values?
        validator = PosMuxValidator(field_info.labels)

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            builder.stringOut,
            str,
            initial_value=values[record_name],
            validate=validator.validate,
        )

        return record_dict

    def _make_table(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, List[str]],
    ) -> Dict[EpicsName, _RecordInfo]:
        assert isinstance(field_info, TableFieldInfo)
        assert field_info.fields
        assert field_info.row_words
        assert field_info.max_length

        record_dict: Dict[EpicsName, _RecordInfo] = {}

        # The INDEX record's starting value
        DEFAULT_INDEX = 0

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
            record_dict,  # Dict is filled throughout this method
            values[record_name],
        )

        # Note that the table_updater's table_fields are guaranteed sorted in bit order,
        # unlike field_info's fields.
        field_data = TablePacking.unpack(
            field_info.row_words, table_updater.table_fields, values[record_name]
        )

        for (field_name, field_details), data in zip(
            table_updater.table_fields.items(), field_data
        ):
            full_name = record_name + ":" + field_name
            full_name = EpicsName(full_name)
            record_dict[full_name] = self._create_record_info(
                full_name,
                field_details.description,
                builder.WaveformOut,
                _raiseIgnoreException,
                validate=table_updater.validate_waveform,
                on_update_name=table_updater.update_waveform,
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

            # Scalar record gives access to individual cell in a column,
            # in combination with the INDEX record defined below
            scalar_record_name = EpicsName(full_name + ":SCALAR")

            # Three possible field types, do per-type config
            record_creation_func: Callable
            lopr = {}
            scalar_labels: List[str] = []
            # No better default than zero, despite the fact it could be a valid value
            initial_value = data[DEFAULT_INDEX] if data.size > 0 else 0
            if field_details.subtype == "int":
                record_creation_func = builder.longIn

            elif field_details.subtype == "uint":
                record_creation_func = builder.longIn
                assert initial_value >= 0, (
                    f"initial value {initial_value} for uint record "
                    f"{scalar_record_name} was negative"
                )
                lopr.update({"LOPR": 0})  # Clamp record to positive values only

            elif field_details.subtype == "enum":
                assert field_details.labels
                record_creation_func = builder.mbbIn
                # Only calling process_labels for label length checking
                scalar_labels, initial_value = self._process_labels(
                    field_details.labels, field_details.labels[initial_value]
                )

            record_dict[scalar_record_name] = self._create_record_info(
                scalar_record_name,
                "Scalar val (set by INDEX rec) of column",
                record_creation_func,
                _raiseIgnoreException,
                initial_value=initial_value,
                labels=scalar_labels,
                **lopr,
            )

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
        mode_record_name = EpicsName(record_name + ":" + "MODE")
        record_dict[mode_record_name] = self._create_record_info(
            mode_record_name,
            "Controls PandA <-> EPICS data interface",
            builder.mbbOut,
            _raiseIgnoreException,
            labels=labels,
            initial_value=index_value,
            on_update=table_updater.update_mode,
        )

        record_dict[mode_record_name].table_updater = table_updater
        table_updater.set_mode_record_info(record_dict[mode_record_name])

        # Index record specifies which element the scalar records should access
        index_record_name = EpicsName(record_name + ":INDEX")
        record_dict[index_record_name] = self._create_record_info(
            index_record_name,
            "Index for all SCALAR records on table",
            builder.longOut,
            _raiseIgnoreException,
            initial_value=DEFAULT_INDEX,
            on_update=table_updater.update_index,
            LOPR=0,
            HOPR=field_info.max_length - 1,  # zero indexing
        )

        return record_dict

    def _make_uint(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[EpicsName, _RecordInfo]:
        assert isinstance(field_info, UintFieldInfo)

        record_dict: Dict[EpicsName, _RecordInfo] = {}
        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            int,
            LOPR=0,  # Uint cannot be below 0
            **kwargs,
        )

        max_record = EpicsName(record_name + ":MAX")
        record_dict[max_record] = self._create_record_info(
            max_record,
            "Maximum valid value for this field",
            builder.longIn,
            type(field_info.max),
            initial_value=field_info.max,
            LOPR=0,  # Uint cannot be below 0
        )

        return record_dict

    def _make_uint_param(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_uint(
            record_name,
            field_info,
            builder.longOut,
            initial_value=values[record_name],
        )

    def _make_uint_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_uint(
            record_name,
            field_info,
            builder.longIn,
            initial_value=values[record_name],
        )

    def _make_uint_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_uint(
            record_name, field_info, builder.longOut, always_update=True
        )

    def _make_int_param(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.longOut,
                int,
                initial_value=values[record_name],
            )
        }

    def _make_int_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.longIn,
                int,
                initial_value=values[record_name],
            )
        }

    def _make_int_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 0)
        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.longOut,
                int,
                always_update=True,
            )
        }

    def _make_scalar(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[EpicsName, _RecordInfo]:
        # RAW attribute ignored - EPICS should never care about it
        assert isinstance(field_info, ScalarFieldInfo)
        assert field_info.offset is not None  # offset may be 0
        record_dict: Dict[EpicsName, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name, field_info.description, record_creation_func, float, **kwargs
        )

        offset_rec = EpicsName(record_name + ":OFFSET")
        record_dict[offset_rec] = self._create_record_info(
            offset_rec,
            "Offset from scaled data to value",
            builder.aIn,
            type(field_info.offset),
            initial_value=field_info.offset,
        )

        scale_rec = EpicsName(record_name + ":SCALE")
        record_dict[scale_rec] = self._create_record_info(
            scale_rec,
            "Scaling from raw data to value",
            builder.aIn,
            type(field_info.scale),
            initial_value=field_info.scale,
        )

        units_rec = EpicsName(record_name + ":UNITS")
        record_dict[units_rec] = self._create_record_info(
            units_rec,
            "Units associated with value",
            builder.stringIn,
            type(field_info.units),
            initial_value=field_info.units,
        )

        return record_dict

    def _make_scalar_param(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_scalar(
            record_name,
            field_info,
            builder.aOut,
            initial_value=values[record_name],
        )

    def _make_scalar_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_scalar(
            record_name,
            field_info,
            builder.aIn,
            initial_value=values[record_name],
        )

    def _make_scalar_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_scalar(
            record_name, field_info, builder.aOut, always_update=True
        )

    def _make_bit(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[EpicsName, _RecordInfo]:

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
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_bit(
            record_name,
            field_info,
            builder.boolOut,
            initial_value=values[record_name],
        )

    def _make_bit_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_bit(
            record_name,
            field_info,
            builder.boolIn,
            initial_value=values[record_name],
        )

    def _make_bit_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_bit(
            record_name, field_info, builder.boolOut, always_update=True
        )

    def _make_action_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        logging.warning(
            f"Field of type {field_info.type} - {field_info.subtype} defined. Ignoring."
        )
        return {}

    def _make_action_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:

        self._check_num_values(values, 0)
        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                builder.boolOut,
                int,  # not bool, as that'll treat string "0" as true
                # TODO: Special on_update to not send the 1 to PandA!
                ZNAM=ZNAM_STR,
                ONAM=ONAM_STR,
                always_update=True,
            )
        }

    def _make_lut(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[EpicsName, _RecordInfo]:
        # RAW attribute ignored - EPICS should never care about it
        return {
            record_name: self._create_record_info(
                record_name, field_info.description, record_creation_func, str, **kwargs
            ),
        }

    def _make_lut_param(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_lut(
            record_name,
            field_info,
            builder.stringOut,
            initial_value=values[record_name],
        )

    def _make_lut_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 1)
        return self._make_lut(
            record_name,
            field_info,
            builder.stringIn,
            initial_value=values[record_name],
        )

    def _make_lut_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        self._check_num_values(values, 0)
        return self._make_lut(
            record_name, field_info, builder.stringOut, always_update=True
        )

    def _make_enum(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
        record_creation_func: Callable,
        **kwargs,
    ) -> Dict[EpicsName, _RecordInfo]:
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
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        return self._make_enum(record_name, field_info, values, builder.mbbOut)

    def _make_enum_read(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        return self._make_enum(record_name, field_info, values, builder.mbbIn)

    def _make_enum_write(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        values: Dict[EpicsName, ScalarRecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        assert isinstance(field_info, EnumFieldInfo)
        assert field_info.labels
        assert record_name not in values
        # Values are not returned for write fields. Create data for label parsing.
        values = {record_name: field_info.labels[0]}
        return self._make_enum(
            record_name, field_info, values, builder.mbbOut, always_update=True
        )

    def create_record(
        self,
        record_name: EpicsName,
        field_info: FieldInfo,
        field_values: Dict[EpicsName, RecordValue],
    ) -> Dict[EpicsName, _RecordInfo]:
        """Create the record (and any child records) for the PandA field specified in
        the parameters.

        Args:
            record_name: The name of the record to create, with colons separating
                words in EPICS style.
            field_info: The field info for the record being created
            field_values: The dictionary of values for the record and
                all child records. The keys are in EPICS style.

        Returns:
            Dict[str, _RecordInfo]: A dictionary of created _RecordInfo objects that
                will need updating later. Not all created records are returned.
        """

        try:
            key = (field_info.type, field_info.subtype)
            if key == ("table", None):
                # Table expects vals in Dict[str, List[str]]
                list_vals = {
                    EpicsName(k): v
                    for (k, v) in field_values.items()
                    if isinstance(v, list)
                }

                return self._make_table(record_name, field_info, list_vals)

            # PandA can never report a table field as in error, only scalar fields
            str_vals = {
                EpicsName(k): v
                for (k, v) in field_values.items()
                if isinstance(v, (str, _InErrorException))
            }

            return self._field_record_mapping[key](
                self, record_name, field_info, str_vals
            )

        except KeyError as e:
            # Unrecognised type-subtype key, ignore this item. This allows the server
            # to define new types without breaking the client.
            logging.warning(
                f"Exception while creating record for {record_name}, type {key}",
                exc_info=e,
            )
            return {}

    # Map a field's (type, subtype) to a function that creates and returns record(s)
    _field_record_mapping: Dict[
        Tuple[str, Optional[str]],
        Callable[
            [
                "IocRecordFactory",
                EpicsName,
                FieldInfo,
                Dict[EpicsName, ScalarRecordValue],
            ],
            Dict[EpicsName, _RecordInfo],
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
        ("param", "action"): _make_action_read,
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
        self, block: str, block_info: BlockInfo, block_values: Dict[EpicsName, str]
    ) -> Dict[EpicsName, _RecordInfo]:
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
            _HDF5RecordController(self._client, self._record_prefix)

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
) -> Dict[EpicsName, _RecordInfo]:
    """Query the PandA and create the relevant records based on the information
    returned"""

    panda_dict = await introspect_panda(client)

    # Dictionary containing every record of every type
    all_records: Dict[EpicsName, _RecordInfo] = {}

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
                record_name = EpicsName(block_number + ":" + field)

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


async def update(
    client: AsyncioClient, all_records: Dict[EpicsName, _RecordInfo], poll_period: int
):
    """Query the PandA at regular intervals for any changed fields, and update
    the records accordingly

    Args:
        client: The AsyncioClient that will be used to get the Changes from the PandA
        all_records: The dictionary of all records that are expected to be updated when
            PandA reports changes. This is NOT all records in the IOC.
        poll_period: The wait time, in seconds, before the next GetChanges is called.
            TODO: This will eventually need to become a millisecond wait, so we can poll
            at 10HZ"""
    while True:
        try:
            changes = await client.send(GetChanges(ChangeGroup.ALL, True))
            if changes.in_error:
                logging.error(
                    "The following fields are reported as being in error: "
                    f"{changes.in_error}"
                )

            for field in changes.in_error:
                field = _ensure_block_number_present(field)
                field = PandAName(field)
                field = _panda_to_epics_name(field)

                if field not in all_records:
                    logging.error(
                        f"Unknown field {field} returned from GetChanges in_error"
                    )

                record = all_records[field].record
                record.set_alarm(alarm.INVALID_ALARM, alarm.UDF_ALARM)

            for field, value in changes.values.items():
                field = _ensure_block_number_present(field)
                field = PandAName(field)
                field = _panda_to_epics_name(field)

                if field not in all_records:
                    logging.error(
                        f"Unknown field {field} returned from GetChanges values"
                    )

                record_info = all_records[field]
                record = record_info.record
                if record_info.labels:
                    # Record is enum, convert string the PandA returns into an int index
                    # TODO: Needs Process=False to not call on_updates which end up
                    #   putting data back to PandA!
                    # TODO: Create GetChanges dict that is passed everywhere, and can be
                    #   used as the "previous value" in the updaters for when Puts fail.
                    record.set(record_info.labels.index(value))
                else:
                    record.set(record_info.data_type_func(value))

            for table_field, value_list in changes.multiline_values.items():
                table_field = PandAName(table_field)
                table_field = _panda_to_epics_name(table_field)
                # Tables must have a MODE record defined - use it to update the table
                mode_rec_name = EpicsName(table_field + ":MODE")
                if mode_rec_name not in all_records:
                    logging.error(
                        f"Table MODE record {mode_rec_name} not found in known records"
                    )
                    break

                table_updater = all_records[mode_rec_name].table_updater
                assert table_updater
                table_updater.update_table(value_list)

            await asyncio.sleep(poll_period)
        except Exception:
            logging.exception("Exception while processing updates from PandA")
            continue
