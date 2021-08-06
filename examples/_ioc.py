# Creating PythonSoftIOCs directly from PandA responses

import asyncio
import sys
from dataclasses import dataclass
from string import digits
from typing import Callable, Dict, List, Optional, Tuple

from softioc import builder, softioc

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
            block_name_number, _ = name.split(".", maxsplit=1)
            # block_name = "".join(i for i in block_name_number if not i.isdigit())
            block_name = block_name_number.rstrip(digits)

            if block_name not in values:
                values[block_name] = {}
            values[block_name][name] = value

        panda_dict = {}
        for (block_name, block_info), field_info in zip(
            block_dict.items(), field_infos
        ):
            panda_dict[block_name] = PandaInfo(
                block_info=block_info, fields=field_info, values=values[block_name]
            )

        return panda_dict


def make_time():
    pass


# Map a field's (type, subtype) to a function used to create a record for it
_field_record_mapping: Dict[Tuple[str, Optional[str]], Callable[[], None]] = {
    ("time", None): make_time,
    ("param", "time"): make_time,
    ("read", "time"): make_time,
}


def create_records(panda_dict: Dict[str, PandaInfo]):
    """Create the relevant records from the given block_dict"""

    # Set the record prefix
    builder.SetDeviceName("ABC")

    for block, panda_info in panda_dict.items():
        for field, field_info in panda_info.fields.items():
            pass

    # Boilerplate get the IOC started
    builder.LoadDatabase()
    softioc.iocInit()


if __name__ == "__main__":
    # One-shot run of a co-routine
    asyncio.run(create_softioc())
