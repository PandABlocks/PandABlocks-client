# Creating PythonSoftIOCs directly from PandA Blocks and Fields

import asyncio
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
class BlockAndFieldInfo:
    """Contains all available information for a Block, including Fields and all the
    Values for `block_info.number` instances of the Fields."""

    block_info: BlockInfo
    fields: Dict[str, FieldInfo]
    values: Dict[str, str]
    # See `_create_values_key` to create key for Dict from keys returned from GetChanges


# TODO: Be really fancy and create a custom type for the key
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
    # TODO: Clean up client when we're done with it
    asyncio.run_coroutine_threadsafe(client.connect(), dispatcher.loop).result(TIMEOUT)

    all_records = asyncio.run_coroutine_threadsafe(
        create_records(client, dispatcher), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    asyncio.run_coroutine_threadsafe(update(client, all_records), dispatcher.loop)
    # TODO: Check return from line above periodically to see if there was an exception

    # Temporarily leave this running forever to aid debugging
    softioc.interactive_ioc(globals())


async def introspect_panda(client: AsyncioClient) -> Dict[str, BlockAndFieldInfo]:
    """Query the PandA for all its Blocks, Fields, and Values of fields

    Args:
        client (AsyncioClient): Client used for commuication with the PandA

    Raises:
        Exception: TODO

    Returns:
        Dict[str, BlockAndFieldInfo]: Dictionary containing all information on
            the block
    """

    # Get the list of all blocks in the PandA
    # TODO: Do we care about the decription of the Block itself?
    block_dict = await client.send(GetBlockInfo())

    # Concurrently request info for all fields of all blocks
    field_infos = await asyncio.gather(
        *[client.send(GetFieldInfo(block)) for block in block_dict]
    )
    # TODO: I should reduce round trips and merge this with previous I/O
    # Request initial value of all fields
    changes = await client.send(GetChanges(ChangeGroup.ALL))

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
        panda_dict[block_name] = BlockAndFieldInfo(
            block_info=block_info, fields=field_info, values=values[block_name]
        )

    return panda_dict


# TODO: Is this actually a Factory? Might be more of a Builder...
class IocRecordFactory:
    """Class to handle creating PythonSoftIOC records for a given field defined in
    a PandA"""

    _client: AsyncioClient

    # Constants used in multiple records
    ZNAM_STR: str = "0"
    ONAM_STR: str = "1"

    def __init__(self, client: AsyncioClient):
        self._client = client

    def _get_enum_index_value(self, labels: Optional[List[str]], record_value: str):
        """Find the index of `record_value` in the `labels` list, suitable for
        use in an `initial_value=` argument during record creation.
        Raises ValueError if `record_value` not found in `labels`."""
        assert labels
        assert len(labels) > 0
        return labels.index(record_value)

    def _create_record(
        self,
        record_name: str,
        description: Optional[str],
        record_creation_func: Callable,
        *args,
        **kwargs,
    ) -> RecordWrapper:
        """Create the record, using the given function and passing all optional
        arguments and keyword arguments, and then set the description field for the
        record.

        Args:
            record_name (str): The name this record will be created with
            description (str): The description for this field. This will be truncated
                to 40 characters due to EPICS limitations.
            record_creation_func (Callable): The function that will be used to create
                this record. Expects to be one of the builder.* functions.

        Returns:
            RecordWrapper: The newly created record.
        """

        record = record_creation_func(record_name, *args, **kwargs)

        # Record description field is a maximum of 40 characters long. Ensure any string
        # is shorter than that before setting it.
        if description and len(description) > 40:
            # TODO: Add logging and some kind of warning when this happens
            # TODO: Some of the hard-coded descriptions break this limit. Re-word them.
            description = description[:40]

        record.DESC = description

        return record

    def _make_time(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, RecordWrapper]:
        """Make one record for the timer itself, and a sub-record for its units"""
        assert len(values) == 2, "Incorrect number of values passed, expected 2"
        assert isinstance(field_info, (TimeFieldInfo, SubtypeTimeFieldInfo))
        assert field_info.units_labels
        # TODO: add more info?
        # TODO: Add similar asserts to every function?

        record_dict: Dict[str, RecordWrapper] = {}

        record_dict[record_name] = self._create_record(
            record_name,
            field_info.description,
            record_creation_func,
            initial_value=float(values[record_name]),
        )

        units_record = record_name + ":UNITS"
        initial_index = self._get_enum_index_value(
            field_info.units_labels, values[units_record]
        )
        record_dict[units_record] = self._create_record(
            units_record,
            "Units of time setting",
            builder.mbbIn,
            *field_info.units_labels,
            initial_value=initial_index,
        )

        return record_dict

    def _make_type_time_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        """Make the records for a field of type "time" - one for the time itself, one
        for units, and one for the MIN value.
        """
        # RAW attribute ignored - EPICS should never care about it
        record_dict = self._make_time(record_name, field_info, values, builder.aOut)
        assert isinstance(field_info, TimeFieldInfo)

        min_record = record_name + ":MIN"
        record_dict[min_record] = self._create_record(
            min_record,
            "Minimum programmable time",
            builder.aIn,
            initial_value=field_info.min,
        )

        return record_dict

    def _make_subtype_time_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_time(record_name, field_info, values, builder.aOut)

    def _make_subtype_time_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_time(record_name, field_info, values, builder.aIn)

    def _make_bit_out(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, BitOutFieldInfo)

        record_dict = {}
        record_dict[record_name] = self._create_record(
            record_name,
            field_info.description,
            builder.boolIn,
            ZNAM=self.ZNAM_STR,
            ONAM=self.ONAM_STR,
            initial_value=int(values[record_name]),
        )

        cw_rec_name = record_name + ":CAPTURE_WORD"
        record_dict[cw_rec_name] = self._create_record(
            cw_rec_name,
            "Name of field containing this bit",
            builder.stringIn,
            initial_value=field_info.capture_word,
        )

        offset_rec_name = record_name + ":OFFSET"
        record_dict[offset_rec_name] = self._create_record(
            offset_rec_name,
            "Position of this bit in captured word",
            builder.aIn,
            initial_value=field_info.offset,
        )

        return record_dict

    def _make_pos_out(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, PosOutFieldInfo)
        assert field_info.labels
        record_dict = {}

        record_dict[record_name] = self._create_record(
            record_name,
            field_info.description,
            builder.aOut,
            initial_value=float(values[record_name]),
        )

        capture_rec = record_name + ":CAPTURE"
        capture_index = self._get_enum_index_value(
            field_info.labels, values[capture_rec]
        )
        record_dict[capture_rec] = self._create_record(
            capture_rec,
            "Capture options",
            builder.mbbOut,
            *field_info.labels,
            initial_value=capture_index,
        )

        offset_rec = record_name + ":OFFSET"
        record_dict[offset_rec] = self._create_record(
            offset_rec, "Offset", builder.aOut, initial_value=int(values[offset_rec])
        )

        scale_rec = record_name + ":SCALE"
        record_dict[scale_rec] = self._create_record(
            scale_rec,
            "Scale factor",
            builder.aOut,
            initial_value=float(values[scale_rec]),
        )

        units_rec = record_name + ":UNITS"
        record_dict[units_rec] = self._create_record(
            units_rec,
            "Units string",
            builder.stringOut,
            initial_value=values[units_rec],
        )

        # SCALED attribute doesn't get returned from GetChanges. Instead
        # of trying to dynamically query for it we'll just recalculate it
        scaled_rec = record_name + ":SCALED"
        builder.records.calc(
            scaled_rec,
            CALC="A*B + C",
            INPA=builder.CP(record_dict[record_name]),
            INPB=builder.CP(record_dict[scale_rec]),
            INPC=builder.CP(record_dict[offset_rec]),
        )

        return record_dict

    def _make_ext_out(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, ExtOutFieldInfo)
        assert field_info.labels
        record_dict = {}
        # TODO: Check if initial_value should be set- this field appears
        # to be write only though
        record_dict[record_name] = self._create_record(
            record_name, field_info.description, builder.aIn
        )

        capture_rec = record_name + ":CAPTURE"
        capture_index = self._get_enum_index_value(
            field_info.labels, values[capture_rec]
        )
        record_dict[capture_rec] = self._create_record(
            capture_rec,
            field_info.description,
            builder.mbbOut,
            *field_info.labels,
            initial_value=capture_index,
        )

        return record_dict

    def _make_ext_out_bits(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        record_dict = self._make_ext_out(record_name, field_info, values)

        # bits_rec = record_name + ":BITS"
        # TODO add BITS record after talking to Tom about exploding it into many
        # capture records.

        return record_dict

    def _make_bit_mux(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, BitMuxFieldInfo)
        record_dict = {}

        record_dict[record_name] = self._create_record(
            record_name,
            field_info.description,
            builder.stringOut,
            initial_value=values[record_name],
        )

        delay_rec = record_name + ":DELAY"
        record_dict[delay_rec] = self._create_record(
            delay_rec,
            "Clock delay on input",
            builder.aOut,
            initial_value=int(values[delay_rec]),
        )

        max_delay_rec = record_name + ":MAX_DELAY"
        record_dict[max_delay_rec] = self._create_record(
            max_delay_rec,
            "Maximum valid input delay",
            builder.aIn,
            initial_value=field_info.max_delay,
        )

        return record_dict

    def _make_pos_mux(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, PosMuxFieldInfo)
        assert field_info.labels
        record_dict = {}

        initial_index = self._get_enum_index_value(
            field_info.labels, values[record_name]
        )
        record_dict[record_name] = self._create_record(
            record_name,
            field_info.description,
            builder.mbbOut,
            *field_info.labels,
            initial_index,
        )

        return record_dict

    def _make_uint(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, UintFieldInfo)

        record_dict = {}
        record_dict[record_name] = self._create_record(
            record_name,
            field_info.description,
            record_creation_func,
            initial_value=int(values[record_name]),
        )

        max_record = record_name + ":MAX"
        record_dict[max_record] = self._create_record(
            max_record,
            "Number of places to right shift calculation result before output",
            builder.aIn,
            initial_value=field_info.max,
        )

        return record_dict

    def _make_uint_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_uint(record_name, field_info, values, builder.aIn)

    def _make_uint_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_uint(record_name, field_info, values, builder.aOut)

    def _make_int(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, RecordWrapper]:

        return {
            record_name: self._create_record(
                record_name,
                field_info.description,
                record_creation_func,
                initial_value=int(values[record_name]),
            )
        }

    def _make_int_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_int(record_name, field_info, values, builder.aIn)

    def _make_int_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        assert record_name not in values
        # Write fields don't have values to return from GetChanges, so set default
        # TODO: Extend this logic to all other _write functions?
        values[record_name] = "0"
        return self._make_int(record_name, field_info, values, builder.aOut)

    def _make_scalar(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, RecordWrapper]:
        # RAW attribute ignored - EPICS should never care about it
        assert isinstance(field_info, ScalarFieldInfo)
        assert field_info.offset is not None
        record_dict = {}

        # print(float(values[record_name]), type(float(values[record_name])))
        # record_dict[record_name] = self._create_record(
        #     record_name,
        #     field_info.description,
        #     record_creation_func,
        #     initial_value=float(values[record_name]),
        #     # TODO: Line above, when record_creation_func is aIn,
        #     # gives "TypeError: must be real number, not str"
        #     # initial_value=values[record_name],
        # )

        offset_rec = record_name + ":OFFSET"
        record_dict[offset_rec] = self._create_record(
            offset_rec,
            "Offset from scaled data to value",
            builder.aIn,
            initial_value=int(field_info.offset),
        )

        scale_rec = record_name + ":SCALE"
        record_dict[scale_rec] = self._create_record(
            scale_rec,
            "Scaling from raw data to value",
            builder.aIn,
            initial_value=field_info.scale,
        )

        units_rec = record_name + ":UNITS"
        record_dict[units_rec] = self._create_record(
            units_rec,
            "Units associated with value",
            builder.stringIn,
            initial_value=field_info.units,
        )

        return record_dict

    def _make_scalar_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return self._make_scalar(record_name, field_info, values, builder.aIn)

    def _make_scalar_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return self._make_scalar(record_name, field_info, values, builder.aOut)

    def _make_bit(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, RecordWrapper]:

        return {
            record_name: self._create_record(
                record_name,
                field_info.description,
                record_creation_func,
                ZNAM=self.ZNAM_STR,
                ONAM=self.ONAM_STR,
                initial_value=int(values[record_name]),
            )
        }

    def _make_bit_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return self._make_bit(record_name, field_info, values, builder.boolIn)

    def _make_bit_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return self._make_bit(record_name, field_info, values, builder.boolOut)

    def _make_action_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        raise Exception(
            "Documentation says this field isn't useful for non-write types"
        )  # TODO: What am I supposed to do here? Could delete this and let
        # create_record throw an exception when the mapping isn't in the dict

    def _make_action_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return {
            record_name: self._create_record(
                record_name,
                field_info.description,
                builder.boolOut,
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
    ) -> Dict[str, RecordWrapper]:
        # RAW attribute ignored - EPICS should never care about it
        return {
            record_name: self._create_record(
                record_name,
                field_info.description,
                record_creation_func,
                initial_value=values[record_name],
            ),
        }

    def _make_lut_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_lut(record_name, field_info, values, builder.stringIn)

    def _make_lut_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_lut(record_name, field_info, values, builder.stringOut)

    def _make_enum(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
        record_creation_func: Callable,
    ) -> Dict[str, RecordWrapper]:
        assert isinstance(field_info, EnumFieldInfo)
        assert field_info.labels

        index_value = self._get_enum_index_value(field_info.labels, values[record_name])

        return {
            record_name: self._create_record(
                record_name,
                field_info.description,
                record_creation_func,
                *field_info.labels,
                initial_value=index_value,
            )
        }

    def _make_enum_read(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return self._make_enum(record_name, field_info, values, builder.mbbIn)

    def _make_enum_write(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return self._make_enum(record_name, field_info, values, builder.mbbOut)

    def create_record(
        self, record_name: str, field_info: FieldInfo, field_values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        """Create the record (and any child records) for the PandA field specified in
        the parameters.
        TODO: Argument documentation?"""
        key = (field_info.type, field_info.subtype)
        if key not in self._field_record_mapping:
            return {}  # TODO: Eventually make this into an exception
        return self._field_record_mapping[key](
            self, record_name, field_info, field_values
        )

    # Map a field's (type, subtype) to a function that creates and returns record(s)
    _field_record_mapping: Dict[
        Tuple[str, Optional[str]],
        Callable[
            ["IocRecordFactory", str, FieldInfo, Dict[str, str]],
            Dict[str, RecordWrapper],
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


async def create_records(
    client: AsyncioClient,
    dispatcher: asyncio_dispatcher.AsyncioDispatcher,
) -> Dict[
    str, RecordWrapper
]:  # TODO: RecordWrapper doesn't exist for GraphQL, will need to change
    """Create the relevant records from the given block_dict"""

    panda_dict = await introspect_panda(client)

    # Set the record prefix
    builder.SetDeviceName("ABC")

    # Dictionary containing every record of every type
    all_records: Dict[str, RecordWrapper] = {}

    record_factory = IocRecordFactory(client)

    # For each field in each block, create block_num records of each field
    for block, panda_info in panda_dict.items():
        block_info = panda_info.block_info
        values = panda_info.values

        for field, field_info in panda_info.fields.items():

            for block_num in range(block_info.number):
                # ":" separator for EPICS Record names, unlike PandA's "."
                record_name = block + ":" + field
                if block_info.number > 1:
                    # TODO: If there is only 1 block, <block> and <block>1 are
                    # synonomous. Perhaps just always use number?
                    # If more than 1 block, the block number becomes part of the PandA
                    # field and hence should become part of the record.
                    # Note PandA block counter is 1-indexed, hence +1
                    # Only replace first instance to avoid problems with awkward field
                    # names like "DIV1:DIVISOR"
                    record_name = record_name.replace(
                        block, block + str(block_num + 1), 1
                    )

                # Get the value of the field and all its sub-fields
                field_values = {
                    field: value
                    for field, value in values.items()
                    if field.startswith(record_name)
                }

                records = record_factory.create_record(
                    record_name, field_info, field_values
                )

                for new_record in records:
                    if new_record in all_records:
                        raise Exception(
                            f"Duplicate record name {new_record} detected!"
                            # TODO: More explanation here?
                        )

                all_records.update(records)

    # Boilerplate get the IOC started
    # TODO: Move these lines somewhere else - having them here won't work for GraphQL
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    return all_records


async def update(client: AsyncioClient, all_records: Dict[str, RecordWrapper]):
    """Query the PandA at regular intervals for any changed fields, and update
    the records accordingly"""
    while True:

        changes = await client.send(GetChanges(ChangeGroup.ALL))
        if changes.in_error:
            raise Exception("Problem here!")
            # TODO: Combine with getChanges error handling in introspect_panda?
            # TODO: Use changes.no_value?

        for field, value in changes.values.items():
            field = _create_values_key(field)
            if field not in all_records:
                # TODO: uncomment when we have all fields
                # raise Exception("Unknown record returned from GetChanges")
                pass
            record = all_records[field]
            record.set(value)
        await asyncio.sleep(1)


if __name__ == "__main__":
    create_softioc()
