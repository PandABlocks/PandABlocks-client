# Creating PythonSoftIOCs directly from PandA Blocks and Fields

import asyncio
import inspect
import sys
from dataclasses import dataclass
from string import digits
from typing import Callable, Dict, List, Optional, Tuple

from softioc import asyncio_dispatcher, builder, softioc
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import ChangeGroup, GetBlockInfo, GetChanges, GetFieldInfo
from pandablocks.responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    BlockInfo,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)

# Define the public API of this module
# TODO!
# __all__ = [""]

TIMEOUT = 2


@dataclass
class _BlockAndFieldInfo:
    """Contains all available information for a Block, including Fields and all the
    Values for `block_info.number` instances of the Fields."""

    block_info: BlockInfo
    fields: Dict[str, FieldInfo]
    values: Dict[str, str]
    # See `_create_values_key` to create key for Dict from keys returned from GetChanges


@dataclass
class _RecordInfo:
    """A container for a record and extra information needed to later update
    the record"""

    record: RecordWrapper
    data_type_func: Callable
    labels: Optional[List[str]] = None


def _create_values_key(field_name: str) -> str:
    """Convert the dictionary key style used in `GetChanges.values` to that used
    throughout this application"""
    # EPICS record names use ":" as separators rather than PandA's ".".
    # GraphQL doesn't care, so we'll standardise on EPICS version.
    return field_name.replace(".", ":")


def create_softioc():
    """Main function of this file. Queries the PandA and creates records from it"""
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()

    client = AsyncioClient(sys.argv[1])

    asyncio.run_coroutine_threadsafe(client.connect(), dispatcher.loop).result(TIMEOUT)

    all_records = asyncio.run_coroutine_threadsafe(
        create_records(client, dispatcher), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    asyncio.run_coroutine_threadsafe(update(client, all_records), dispatcher.loop)
    # TODO: Check return from line above periodically to see if there was an exception

    # Temporarily leave this running forever to aid debugging
    softioc.interactive_ioc(globals())

    asyncio.run_coroutine_threadsafe(client.close(), dispatcher.loop).result(TIMEOUT)


async def introspect_panda(client: AsyncioClient) -> Dict[str, _BlockAndFieldInfo]:
    """Query the PandA for all its Blocks, Fields of each Block, and Values of each Field

    Args:
        client (AsyncioClient): Client used for commuication with the PandA

    Raises:
        Exception: TODO

    Returns:
        Dict[str, BlockAndFieldInfo]: Dictionary containing all information on
            the block
    """

    # Get the list of all blocks in the PandA
    block_dict = await client.send(GetBlockInfo())

    # Concurrently request info for all fields of all blocks
    # Note order of requests is important as it is unpacked below
    returned_infos = await asyncio.gather(
        *[client.send(GetFieldInfo(block)) for block in block_dict],
        client.send(GetChanges(ChangeGroup.ALL)),
    )

    field_infos = returned_infos[0:-1]

    changes = returned_infos[-1]
    if changes.in_error:
        raise Exception("TODO: Some better error handling")
    # TODO: Do something with changes.no_value ?

    values: Dict[str, Dict[str, str]] = {}
    for field_name, value in changes.values.items():
        block_name_number, subfield_name = field_name.split(".", maxsplit=1)
        block_name = block_name_number.rstrip(digits)

        field_name = _create_values_key(field_name)

        if block_name not in values:
            values[block_name] = {}
        values[block_name][field_name] = value

    panda_dict = {}
    for (block_name, block_info), field_info in zip(block_dict.items(), field_infos):
        panda_dict[block_name] = _BlockAndFieldInfo(
            block_info=block_info, fields=field_info, values=values[block_name]
        )

    return panda_dict


class IocRecordFactory:
    """Class to handle creating PythonSoftIOC records for a given field defined in
    a PandA"""

    _record_prefix: str

    _pos_out_row_counter: int = 0

    # List of methods in builder, used for parameter validation
    _builder_methods = [
        method
        for _, method in inspect.getmembers(builder, predicate=inspect.isfunction)
    ]

    # Constants used in multiple records
    ZNAM_STR: str = "0"
    ONAM_STR: str = "1"

    def __init__(self, record_prefix: str):
        self._record_prefix = record_prefix

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
        # TODO: Warn on truncation? Similar to warn on description truncation.
        return ([label[:25] for label in labels], labels.index(record_value))

    def _check_num_values(self, values: Dict[str, str], num: int) -> None:
        """Function to check that the number of values is the expected amount.
        Raises AssertionError if incorrect number of values."""
        assert (
            len(values) == num
        ), f"Incorrect number of values, {len(values)}, expected {num}.\n {values}"

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

        Args:
            record_name (str): The name this record will be created with
            description (str): The description for this field. This will be truncated
                to 40 characters due to EPICS limitations.
            record_creation_func (Callable): The function that will be used to create
                this record. Expects to be one of the builder.* functions.
            data_type_func (Callable): The function to use to convert the value returned
                from GetChanges, which will always be a string, into a type appropriate
                for the record e.g. int, float.
            labels (List[str]): If the record type being created is a mbbi or mbbo
                record, provide the list of valid labels here.

        Returns:
            _RecordInfo: Class containing the created record and anything needed for
                updating the record.
        """
        assert (
            record_creation_func in self._builder_methods
        ), "Unrecognised record creation function passed to _create_record_info"

        # TODO: Take another go at validating the data_type_func. Can't use
        # inspect.isbuiltin,could do something like
        # isinstance(data_type_func, type(float)) but that seems hacky

        if (
            record_creation_func == builder.mbbIn
            or record_creation_func == builder.mbbOut
        ):
            assert len(labels) <= 16, f"Too many labels to create record {record_name}"

        record = record_creation_func(record_name, *labels, *args, **kwargs)

        # Record description field is a maximum of 40 characters long. Ensure any string
        # is shorter than that before setting it.
        if description and len(description) > 40:
            # TODO: Add logging/some kind of warning when this happens
            description = description[:40]

        record.DESC = description

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
    ) -> Dict[str, _RecordInfo]:
        """Make one record for the timer itself, and a sub-record for its units"""
        self._check_num_values(values, 2)
        assert isinstance(field_info, (TimeFieldInfo, SubtypeTimeFieldInfo))
        assert field_info.units_labels

        record_dict: Dict[str, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            float,
            initial_value=float(values[record_name]),
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

    def _make_type_time_write(
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
        record_dict = self._make_time(record_name, field_info, values, builder.aOut)

        min_record = record_name + ":MIN"
        record_dict[min_record] = self._create_record_info(
            min_record,
            "Minimum programmable time",
            builder.aIn,
            type(field_info.min),
            initial_value=field_info.min,
        )

        return record_dict

    def _make_subtype_time_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        return self._make_time(record_name, field_info, values, builder.aOut)

    def _make_subtype_time_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        return self._make_time(record_name, field_info, values, builder.aIn)

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
            ZNAM=self.ZNAM_STR,
            ONAM=self.ONAM_STR,
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
            builder.aOut,
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
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        # TODO: Implement this!
        return {}

    def _make_uint(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)
        assert isinstance(field_info, UintFieldInfo)

        record_dict: Dict[str, _RecordInfo] = {}
        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            int,
            initial_value=int(values[record_name]),
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

    def _make_uint_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        return self._make_uint(record_name, field_info, values, builder.aIn)

    def _make_uint_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        return self._make_uint(record_name, field_info, values, builder.aOut)

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
            )
        }

    def _make_scalar(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, _RecordInfo]:
        # RAW attribute ignored - EPICS should never care about it
        self._check_num_values(values, 1)
        assert isinstance(field_info, ScalarFieldInfo)
        assert field_info.offset is not None
        record_dict: Dict[str, _RecordInfo] = {}

        record_dict[record_name] = self._create_record_info(
            record_name,
            field_info.description,
            record_creation_func,
            float,
            initial_value=float(values[record_name]),
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

    def _make_scalar_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_scalar(record_name, field_info, values, builder.aIn)

    def _make_scalar_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_scalar(record_name, field_info, values, builder.aOut)

    def _make_bit(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, _RecordInfo]:
        self._check_num_values(values, 1)

        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                record_creation_func,
                int,
                ZNAM=self.ZNAM_STR,
                ONAM=self.ONAM_STR,
                initial_value=int(values[record_name]),
            )
        }

    def _make_bit_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_bit(record_name, field_info, values, builder.boolIn)

    def _make_bit_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_bit(record_name, field_info, values, builder.boolOut)

    def _make_action_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        raise Exception(
            "Documentation says this field isn't useful for non-write types"
        )  # TODO: Should I still create a record here?

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
                ZNAM=self.ZNAM_STR,
                ONAM=self.ONAM_STR,
            )
        }

    def _make_lut(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, _RecordInfo]:
        # RAW attribute ignored - EPICS should never care about it
        self._check_num_values(values, 1)
        return {
            record_name: self._create_record_info(
                record_name,
                field_info.description,
                record_creation_func,
                str,
                initial_value=values[record_name],
            ),
        }

    def _make_lut_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        return self._make_lut(record_name, field_info, values, builder.stringIn)

    def _make_lut_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, _RecordInfo]:
        return self._make_lut(record_name, field_info, values, builder.stringOut)

    def _make_enum(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
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
            )
        }

    def _make_enum_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_enum(record_name, field_info, values, builder.mbbIn)

    def _make_enum_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        return self._make_enum(record_name, field_info, values, builder.mbbOut)

    def create_record(
        self, record_name: str, field_info: FieldInfo, field_values: Dict[str, str]
    ) -> Dict[str, _RecordInfo]:
        """Create the record (and any child records) for the PandA field specified in
        the parameters.

        Args:
            record_name (str): The name of the record to create, with colons separating
                words in EPICS style.
            field_info (FieldInfo): The field info for the record being created
            field_values (Dict[str, str]): The dictionary of values for the record and
                all child records. The keys are in EPICS style.

        Raises:
            Exception: Raised when the field's type and subtype are unrecognised

        Returns:
            Dict[str, _RecordInfo]: A dictionary of all records created and their
                associated _RecordInfo object
        """
        key = (field_info.type, field_info.subtype)
        if key not in self._field_record_mapping:
            raise Exception(
                f'Unrecognised type-subtype pair "{key}" when creating {record_name}.'
                + "Could not create record."
            )
        return self._field_record_mapping[key](
            self, record_name, field_info, field_values
        )

    # Map a field's (type, subtype) to a function that creates and returns record(s)
    _field_record_mapping: Dict[
        Tuple[str, Optional[str]],
        Callable[
            ["IocRecordFactory", str, FieldInfo, Dict[str, str]],
            Dict[str, _RecordInfo],
        ],
    ] = {
        # Order matches that of PandA server's Field Types docs
        ("time", None): _make_type_time_write,
        ("bit_out", None): _make_bit_out,
        ("pos_out", None): _make_pos_out,
        ("ext_out", "timestamp"): _make_ext_out,
        ("ext_out", "samples"): _make_ext_out,
        ("ext_out", "bits"): _make_ext_out_bits,
        ("bit_mux", None): _make_bit_mux,
        ("pos_mux", None): _make_pos_mux,
        ("table", None): _make_table,
        ("param", "uint"): _make_uint_read,
        ("read", "uint"): _make_uint_read,
        ("write", "uint"): _make_uint_write,
        ("param", "int"): _make_int_read,
        ("read", "int"): _make_int_read,
        ("write", "int"): _make_int_write,
        ("param", "scalar"): _make_scalar_read,
        ("read", "scalar"): _make_scalar_read,
        ("write", "scalar"): _make_scalar_write,
        ("param", "bit"): _make_bit_read,
        ("read", "bit"): _make_bit_read,
        ("write", "bit"): _make_bit_write,
        ("param", "action"): _make_action_read,
        ("read", "action"): _make_action_read,
        ("write", "action"): _make_action_write,
        ("param", "lut"): _make_lut_read,
        ("read", "lut"): _make_lut_read,
        ("write", "lut"): _make_lut_write,
        ("param", "enum"): _make_enum_read,
        ("read", "enum"): _make_enum_read,
        ("write", "enum"): _make_enum_write,
        ("param", "time"): _make_subtype_time_read,
        ("read", "time"): _make_subtype_time_read,
        ("write", "time"): _make_subtype_time_write,
    }

    def initialise(self, dispatcher: asyncio_dispatcher.AsyncioDispatcher) -> None:
        """Perform any final initialisation code to create the records. No new
        records may be created after this method is called.

        Args:
            dispatcher (asyncio_dispatcher.AsyncioDispatcher): The dispatcher used in
                IOC initialisation
        """
        # TODO: Delete WriteRecords when I'm done debugging
        builder.WriteRecords("records.db")
        builder.LoadDatabase()
        softioc.iocInit(dispatcher)


async def create_records(
    client: AsyncioClient,
    dispatcher: asyncio_dispatcher.AsyncioDispatcher,
) -> Dict[str, _RecordInfo]:
    """Query the PandA and create the relevant records based on the information
    returned"""

    panda_dict = await introspect_panda(client)

    # Dictionary containing every record of every type
    all_records: Dict[str, _RecordInfo] = {}

    record_factory = IocRecordFactory("ABC")

    # For each field in each block, create block_num records of each field
    for block, panda_info in panda_dict.items():
        block_info = panda_info.block_info
        values = panda_info.values

        for field, field_info in panda_info.fields.items():

            for block_num in range(block_info.number):
                # ":" separator for EPICS Record names, unlike PandA's "."
                record_name = block + ":" + field
                if block_info.number > 1:
                    # If more than 1 block, the block number becomes part of the PandA
                    # block name and hence should become part of the record.
                    # Note PandA block counter is 1-indexed, hence +1
                    # Only replace first instance to avoid problems with awkward field
                    # names like "DIV1:DIVISOR"
                    record_name = record_name.replace(
                        block, block + str(block_num + 1), 1
                    )

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
        changes = await client.send(GetChanges(ChangeGroup.ALL))
        if changes.in_error:
            pass
            # TODO: We should continue processing here, but log an error somehow
            # TODO: Combine with getChanges error handling in introspect_panda?
            # TODO: Use changes.no_value?

        for field, value in changes.values.items():
            field = _create_values_key(field)
            if field not in all_records:
                raise Exception("Unknown record returned from GetChanges")
            record_info = all_records[field]
            record = record_info.record
            if record_info.labels:
                # Record is enum, convert string the PandA returns into an int index
                record.set(record_info.labels.index(value))
            else:
                record.set(record_info.data_type_func(value))
        await asyncio.sleep(1)


if __name__ == "__main__":
    create_softioc()
