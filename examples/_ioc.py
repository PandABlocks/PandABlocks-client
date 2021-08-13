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
    BlockInfo,
    FieldInfo,
    SubtypeTimeFieldInfo,
    TimeFieldInfo,
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
    _client: AsyncioClient
    _dispatcher: asyncio_dispatcher.AsyncioDispatcher = (
        asyncio_dispatcher.AsyncioDispatcher()
    )
    """Class to handle creating PythonSoftIOC records for a given field defined in
    a PandA"""

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

        rec1 = record_creation_func(
            record_name, initial_value=float(values[record_name])
        )
        record_dict[record_name] = rec1

        units_record = record_name + ":UNITS"
        initial_unit = field_info.units_labels.index(values[units_record])
        rec2 = builder.mbbIn(
            units_record, *field_info.units_labels, initial_value=initial_unit
        )
        record_dict[units_record] = rec2

        if field_info.type == "time":
            assert isinstance(field_info, TimeFieldInfo)
            min_record = record_name + ":MIN"
            rec3 = builder.aIn(min_record, initial_value=field_info.min)
            record_dict[min_record] = rec3

        return record_dict

    def _make_time_write(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_time(record_name, field_info, values, builder.aOut)

    def _make_time_read(
        self,
        record_name: str,
        field_info: FieldInfo,
        values: Dict[str, str],
    ) -> Dict[str, RecordWrapper]:
        return self._make_time(record_name, field_info, values, builder.aIn)

    def _make_boolin(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return {
            record_name: builder.boolIn(
                record_name,
                ZNAM=self.ZNAM_STR,
                ONAM=self.ONAM_STR,
                initial_value=int(values[record_name]),
            )
        }

    def _make_action(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return {
            record_name: builder.boolOut(
                record_name, ZNAM=self.ZNAM_STR, ONAM=self.ONAM_STR
            )
        }

    def _make_param_uint(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        rec1 = builder.aOut(record_name, initial_value=int(values[record_name]))
        return {record_name: rec1}
        # TODO: Make MAX field record

    def _make_param_int(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return {
            record_name: builder.aOut(
                record_name, initial_value=int(values[record_name])
            )
        }

    def _make_param_scalar(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        # TODO: Create any/all of the child attributes?
        # UNITS, RAW, OFFSET, SCALE, INFO.
        # Would need a lot of GETs for all the initial values.
        return {
            record_name: builder.aOut(
                record_name, initial_value=float(values[record_name])
            )
        }

    def _make_param_bit(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        return {
            record_name: builder.boolOut(
                record_name,
                ZNAM=self.ZNAM_STR,
                ONAM=self.ONAM_STR,
                initial_value=int(values[record_name]),
            )
        }

    def _make_param_action(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        raise Exception(
            "Documentation says this field isn't useful for non-write types"
        )  # TODO: What am I supposed to do here? Could delete this and let
        # create_record throw an exception when the mapping isn't in the dict

    def _make_param_lut(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:

        rec1 = builder.stringOut(record_name, initial_value=values[record_name])
        # TODO: Create the RAW attribute record?
        # raw_name = record_name + ":RAW"
        # raw_rec = builder.aOut(raw_name, initial_value=int(value, 16))
        return {
            record_name: rec1,
        }

    def _make_param_enum(
        self, record_name: str, field_info: FieldInfo, values: Dict[str, str]
    ) -> Dict[str, RecordWrapper]:
        index_value = self._get_enum_index_value(field_info.labels, values[record_name])
        return {
            record_name: builder.mbbIn(
                record_name,
                *field_info.labels,
                initial_value=index_value,
            )
        }

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
        ("time", None): _make_time_write,
        ("param", "time"): _make_time_write,
        ("read", "time"): _make_time_read,
        ("bit_out", None): _make_boolin,
        ("write", "action"): _make_action,
        ("param", "uint"): _make_param_uint,
        ("param", "int"): _make_param_int,
        ("param", "scalar"): _make_param_scalar,
        # ("read", "scalar"): _make_param_scalar,
        # ("write", "scalar"): _make_param_scalar,
        ("param", "bit"): _make_param_bit,
        ("param", "action"): _make_param_action,
        ("param", "lut"): _make_param_lut,
        ("param", "enum"): _make_param_enum,
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

                # TODO: It seems that some record creation requires additional network
                # calls. So we should probably make this into a set of simultaneous
                # async tasks, rather than sequential for loops.
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
