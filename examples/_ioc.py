# Creating PythonSoftIOCs directly from PandA responses

import asyncio
import concurrent.futures
import sys
from typing import Dict

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetBlockInfo, GetFieldInfo
from pandablocks.responses import FieldInfo

# Define the public API of this module
# __all__ = [""]


async def introspect_panda() -> Dict[str, Dict[str, FieldInfo]]:
    """Query the PandA for all of its blocks and fields"""
    async with AsyncioClient(sys.argv[1]) as client:

        # Get the list of all blocks in the PandA
        # TODO: Do we care about the decription of the Block itself?
        block_info = await client.send(GetBlockInfo())

        # Concurrently request info for all fields of all blocks
        field_infos = await asyncio.gather(
            *[client.send(GetFieldInfo(block)) for block in block_info]
        )

        block_dict = {}
        for block, field_info in zip(block_info, field_infos):
            block_dict[block] = field_info

        return block_dict


if __name__ == "__main__":
    # One-shot run of a co-routine
    block_dict = asyncio.run(introspect_panda())
