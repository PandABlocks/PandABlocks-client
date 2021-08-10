# Creating PythonSoftIOCs directly from PandA Blocks and Fields

import asyncio
import sys
from dataclasses import dataclass
from string import digits
from typing import Callable, Dict, Optional, Tuple

from softioc import asyncio_dispatcher, builder, softioc
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import ChangeGroup, GetBlockInfo, GetChanges, GetFieldInfo
from pandablocks.responses import BlockInfo, FieldInfo

# Define the public API of this module
# TODO!
# __all__ = [""]

TIMEOUT = 2


@dataclass
class PandaInfo:
    """Contains all available information for a Block, including Fields and all the
    Values for `block_info.number` instances of the Fields."""

    block_info: BlockInfo
    fields: Dict[str, FieldInfo]
    values: Dict[str, str]
    # See `create_values_key` to create key for Dict from keys returned from GetChanges


# TODO: Be really fancy and create a custom type for the key
def create_values_key(field_name: str) -> str:
    """Convert the dictionary key style used in `GetChanges.values` to that used
    throughout this application"""
    # EPICS record names use ":" as separators rather than PandA's ".".
    # GraphQL doesn't care, so we'll standardise on EPICS version.
    return field_name.replace(".", ":")


def create_softioc():
    """Main function of this file. Queries the PandA and creates records from it"""
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()

    client = AsyncioClient(sys.argv[1])
    # TODO: Either re-insert the `with` block, or clean the client up manually
    asyncio.run_coroutine_threadsafe(client.connect(), dispatcher.loop).result(TIMEOUT)

    panda_dict = asyncio.run_coroutine_threadsafe(
        introspect_panda(client), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    all_records = asyncio.run_coroutine_threadsafe(
        create_records(client, dispatcher, panda_dict), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    asyncio.run_coroutine_threadsafe(update(client, all_records), dispatcher.loop)
    # TODO: Check return from line above periodically to see if there was an exception

    # Temporarily leave this running forever to aid debugging
    softioc.interactive_ioc(globals())


async def introspect_panda(client: AsyncioClient) -> Dict[str, PandaInfo]:
    """Query the PandA for all its Blocks, Fields, and Values of fields"""

    # Get the list of all blocks in the PandA
    # TODO: Do we care about the decription of the Block itself?
    block_dict = await client.send(GetBlockInfo())

    # Concurrently request info for all fields of all blocks
    field_infos = await asyncio.gather(
        *[client.send(GetFieldInfo(block)) for block in block_dict]
    )

    # Request initial value of all fields
    changes = await client.send(GetChanges(ChangeGroup.ALL))

    if changes.in_error:
        raise Exception("TODO: Some better error handling")
    # TODO: Do something with changes.no_value ?

    values: Dict[str, Dict[str, str]] = {}
    for field_name, value in changes.values.items():
        block_name_number, subfield_name = field_name.split(".", maxsplit=1)
        block_name = block_name_number.rstrip(digits)

        field_name = create_values_key(field_name)

        if block_name not in values:
            values[block_name] = {}
        values[block_name][field_name] = value

    panda_dict = {}
    for (block_name, block_info), field_info in zip(block_dict.items(), field_infos):
        panda_dict[block_name] = PandaInfo(
            block_info=block_info, fields=field_info, values=values[block_name]
        )

    return panda_dict


def make_time(
    record_name: str, values: Dict[str, str], record_creation_func: Callable
) -> Dict[str, RecordWrapper]:
    assert len(values) == 2, "Incorrect number of values passed, expected 2"
    # TODO: add more info?

    rec1 = record_creation_func(record_name, initial_value=float(values[record_name]))
    units_record = record_name + ":UNITS"
    # TODO: Does this need tobe stringIn for read fields?
    # rec2 = builder.stringOut(units_record, initial_value=values[units_record])

    valid_units = ["s", "ms", "us"]
    initial_unit = valid_units.index(values[units_record])

    rec2 = builder.mbbIn(units_record, *valid_units, initial_value=initial_unit)
    return {record_name: rec1, units_record: rec2}
    # TODO: pandablockscontroller L272 gives this array of valid values:
    # ["s", "ms", "us"]. How do I do this with EPICS records?


def make_time_write(
    record_name: str, values: Dict[str, str]
) -> Dict[str, RecordWrapper]:
    """Make one writeable record for the timer itself, and a sub-record for its units"""
    return make_time(record_name, values, builder.aOut)


def make_time_read(
    record_name: str, values: Dict[str, str]
) -> Dict[str, RecordWrapper]:
    """Make one readable record for the timer itself, and a sub-record for its units"""
    return make_time(record_name, values, builder.aIn)


def make_boolin(record_name: str, values: Dict[str, str]) -> Dict[str, RecordWrapper]:

    return {
        record_name: builder.boolIn(
            record_name, ZNAM="0", ONAM="1", initial_value=int(values[record_name])
        )
    }


def make_action(record_name: str, values: Dict[str, str]) -> Dict[str, RecordWrapper]:
    return {record_name: builder.boolOut(record_name, ZNAM="0", ONAM="1")}


# Map a field's (type, subtype) to a function that creates and returns record(s)
_field_record_mapping: Dict[
    Tuple[str, Optional[str]],
    Callable[[str, Dict[str, str]], Dict[str, RecordWrapper]],
] = {
    ("time", None): make_time_write,
    ("param", "time"): make_time_write,
    ("read", "time"): make_time_read,
    ("bit_out", None): make_boolin,
    ("write", "action"): make_action,
}


async def create_records(
    client: AsyncioClient,
    dispatcher: asyncio_dispatcher.AsyncioDispatcher,
    panda_dict: Dict[str, PandaInfo],
) -> Dict[
    str, RecordWrapper
]:  # TODO: RecordWrapper doesn't exist for GraphQL, will need to change
    """Create the relevant records from the given block_dict"""

    # Set the record prefix
    builder.SetDeviceName("ABC")

    # Dictionary containing every record of every type
    all_records: Dict[str, RecordWrapper] = {}

    # For each field in each block, create block_num records of each field
    for block, panda_info in panda_dict.items():
        block_info = panda_info.block_info
        values = panda_info.values

        for field, field_info in panda_info.fields.items():

            key = (field_info.type, field_info.subtype)
            if key not in _field_record_mapping:
                continue  # TODO: Eventually make this into an exception
            create_record_func = _field_record_mapping[key]

            for block_num in range(block_info.number):
                # ":" separator for EPICS Record names, unlike PandA's "."
                record_name = block + ":" + field
                if block_info.number > 1:
                    # If more than 1 block, the block number becomes part of the PandA
                    # field and hence should become part of the record.
                    # Note PandA block counter is 1-indexed, hence +1
                    record_name = record_name.replace(block, block + str(block_num + 1))

                # Get the value of the field and all its sub-fields
                field_values = {
                    field: value
                    for field, value in values.items()
                    if field.startswith(record_name)
                }
                records = create_record_func(record_name, field_values)

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
            field = create_values_key(field)
            if field not in all_records:
                # TODO: uncomment when we have all fields
                # raise Exception("Unknown record returned from GetChanges")
                pass
            record = all_records[field]
            record.set(value)
        await asyncio.sleep(1)


if __name__ == "__main__":
    create_softioc()
