import asyncio
import sys

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Put
from pandablocks.hdf import write_hdf_files


async def arm_and_hdf():
    # Create a client and connect the control and data ports
    async with AsyncioClient(sys.argv[1]) as client:
        # Put to 2 fields simultaneously
        await asyncio.gather(
            client.send(Put("SEQ1.REPEATS", 1000)),
            client.send(Put("SEQ1.PRESCALE", 1000)),
        )
        # Listen for data, arming the PandA at the beginning
        await write_hdf_files(client, scheme="/tmp/panda-capture-%d.h5", arm=True)


if __name__ == "__main__":
    # One-shot run of a co-routine
    asyncio.run(arm_and_hdf())
