import asyncio
import pprint
import sys

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetBlockInfo, GetFieldInfo, GetPcapBitsLabels


async def introspect():
    # Create a client and connect the control and data ports
    async with AsyncioClient(sys.argv[1]) as client:
        # Get the list of all blocks in the PandA
        block_info = await client.send(GetBlockInfo())
        # Find and print all fields for each block
        for block in block_info:
            field_info = await client.send(GetFieldInfo(block))
            pprint.pprint({block: field_info})

        # Get the labels for every PCAP.BITS[n] fields
        labels = await client.send(GetPcapBitsLabels())
        pprint.pprint(labels)


if __name__ == "__main__":
    # One-shot run of a co-routine
    asyncio.run(introspect())
