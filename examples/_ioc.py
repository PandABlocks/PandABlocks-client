# Creating PythonSoftIOCs directly from PandA Blocks and Fields

import asyncio
import sys
from dataclasses import dataclass
from string import digits
from typing import Callable, Dict, Optional, Tuple

from softioc import asyncio_dispatcher, builder, softioc

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import ChangeGroup, GetBlockInfo, GetChanges, GetFieldInfo
from pandablocks.responses import BlockInfo, FieldInfo

# Define the public API of this module
# __all__ = [""]


@dataclass
class PandaInfo:
    """Contains all available information for a Block, including Fields and all the
    Values"""

    block_info: BlockInfo
    fields: Dict[str, FieldInfo]
    values: Dict[str, str]


async def create_softioc():
    """Main function of this file"""
    panda_dict = await introspect_panda()

    create_records(panda_dict)


async def introspect_panda() -> Dict[str, PandaInfo]:
    """Query the PandA for all of its blocks and fields"""
    async with AsyncioClient(sys.argv[1]) as client:

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

        values: Dict[str, Dict[str, str]] = {}
        for name, value in changes.values.items():
            block_name_number, subfield_name = name.split(".", maxsplit=1)
            block_name = block_name_number.rstrip(digits)

            if block_name not in values:
                values[block_name] = {}
            values[block_name][name] = value
            # TODO: This gives us a structure like {"TTLIN" : {"TTLIN1.VAL" : "1"}}.
            # i.e. the block number is embedded alongside the field name.
            # Come back to this when I have a better idea what structure I'll need

        panda_dict = {}
        for (block_name, block_info), field_info in zip(
            block_dict.items(), field_infos
        ):
            panda_dict[block_name] = PandaInfo(
                block_info=block_info, fields=field_info, values=values[block_name]
            )

        return panda_dict


def make_time(block: str, block_info: BlockInfo, field: str, values: Dict[str, str]):
    pass


def make_boolin(block: str, block_info: BlockInfo, field: str, values: Dict[str, str]):
    for block_num in range(block_info.number):
        block_num += 1  # PandA block counter is 1-indexed

        # ":" separator for EPICS Record names, unlike PandA's "."
        record_name = block + ":" + field
        if block_info.number > 1:
            # If more than 1 block, the block number becomes part of the PandA field
            # and hence should become part of the record
            record_name = record_name.replace(block, block + str(block_num))

        # values dict uses "." as its separator, as that's what PandA uses.
        values_key = record_name.replace(":", ".")
        builder.boolIn(
            record_name, ZNAM="0", ONAM="1", initial_value=int(values[values_key])
        )

        # TODO: Keep a link to the record, so that we can update it using
        # GetChanges somehow


# Map a field's (type, subtype) to a function used to create a record for it
_field_record_mapping: Dict[
    Tuple[str, Optional[str]], Callable[[str, BlockInfo, str, Dict[str, str]], None]
] = {
    ("time", None): make_time,
    ("param", "time"): make_time,
    ("read", "time"): make_time,
    ("bit_out", None): make_boolin,
}


def create_records(panda_dict: Dict[str, PandaInfo]):
    """Create the relevant records from the given block_dict"""

    # Set the record prefix
    builder.SetDeviceName("ABC")

    for block, panda_info in panda_dict.items():
        for field, field_info in panda_info.fields.items():

            key = (field_info.type, field_info.subtype)
            if key not in _field_record_mapping:
                continue
            f = _field_record_mapping[key]
            f(block, panda_info.block_info, field, panda_info.values)

    # Boilerplate get the IOC started
    # Create an asyncio dispatcher, the event loop is now running
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    # Temorary leave running forever
    softioc.interactive_ioc(globals())


if __name__ == "__main__":
    # One-shot run of a co-routine
    asyncio.run(create_softioc())
