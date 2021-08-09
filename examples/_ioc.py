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
    # TODO: Either re-insert the `with` block, or clean this up manually
    asyncio.run_coroutine_threadsafe(client.connect(), dispatcher.loop).result(TIMEOUT)

    panda_dict = asyncio.run_coroutine_threadsafe(
        introspect_panda(client), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    all_records = asyncio.run_coroutine_threadsafe(
        create_records(client, dispatcher, panda_dict), dispatcher.loop
    ).result()  # TODO add TIMEOUT and exception handling

    async def update(client: AsyncioClient, all_records: Dict[str, RecordWrapper]):
        """Query the PandA at regular intervals for any changes fields, and update
        the records accordingly"""
        while True:

            changes = await client.send(GetChanges(ChangeGroup.ALL))
            if changes.in_error:
                raise Exception("Problem here!")
                # TODO: Combine with getChanges error handling in introspect_panda?

            for field, value in changes.values.items():
                field = create_values_key(field)
                if field not in all_records:
                    raise Exception("Unknown record returned from GetChanges")

                record = all_records[field]
                record.set(value)
            await asyncio.sleep(1)

    asyncio.run_coroutine_threadsafe(update(client, all_records), dispatcher.loop)

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


# def make_time(block: str, block_info: BlockInfo, field: str, values: Dict[str, str]):
#     pass


def make_boolin(record_name: str, value: str) -> RecordWrapper:

    return builder.boolIn(record_name, ZNAM="0", ONAM="1", initial_value=int(value))
    # TODO: Keep a link to the record, so that we can update it using
    # GetChanges somehow


# Map a field's (type, subtype) to a function used to create a record for it.
# That function will return the PandA field name and the record itself.
_field_record_mapping: Dict[
    Tuple[str, Optional[str]],
    Callable[[str, str], RecordWrapper],
] = {
    # ("time", None): make_time,
    # ("param", "time"): make_time,
    # ("read", "time"): make_time,
    ("bit_out", None): make_boolin,
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

                record = create_record_func(record_name, values[record_name])

                if record_name in all_records:
                    raise Exception(
                        "Duplicate record names detected! TODO: more explanation here"
                    )
                all_records[record_name] = record

    # Boilerplate get the IOC started
    # TODO: Move these lines somewhere else - having them here won't work for GraphQL
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    return all_records


if __name__ == "__main__":
    create_softioc()
