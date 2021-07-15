import asyncio
import pprint
import sys

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetPcapBitsLabels


# TODO: Add this to documentation somewhere!
async def introspect():
    # Create a client and connect the control and data ports
    async with AsyncioClient(sys.argv[1]) as client:

        labels = await client.send(GetPcapBitsLabels())
        pprint.pprint(labels)


if __name__ == "__main__":
    # One-shot run of a co-routine
    asyncio.run(introspect())
