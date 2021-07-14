import asyncio
import sys
import time

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Arm, Put
from pandablocks.hdf import FrameProcessor, HDFWriter, create_pipeline, stop_pipeline
from pandablocks.responses import EndData, EndReason, FrameData, ReadyData


def print_progress_bar(fraction: float):
    # Print a simple progress bar, with a carriage return rather than newline
    # so that subsequent progress bars can be drawn on top
    print(f"{fraction * 100:5.1f}% [{'=' * int(fraction * 40):40s}]", end="\r")


async def hdf_queue_reporting():
    # Create the pipeline to scale and write HDF files
    pipeline = create_pipeline(
        FrameProcessor(), HDFWriter(scheme="/tmp/panda-capture-%d.h5")
    )
    try:
        async with AsyncioClient(sys.argv[1]) as client:
            # Gather data at 45MByte/s, should take about 60s
            repeats = 40000000
            await asyncio.gather(
                client.send(Put("SEQ1.REPEATS", repeats)),
                client.send(Put("SEQ1.PRESCALE", 0.5)),
            )
            progress = 0
            async for data in client.data(scaled=False, flush_period=1):
                # Always pass the data down the pipeline
                pipeline[0].queue.put_nowait(data)
                if isinstance(data, ReadyData):
                    # Data connection is ready, arm PandA
                    await client.send(Arm())
                elif isinstance(data, FrameData):
                    # Got some frame data, print a progress bar
                    progress += len(data.data)
                    print_progress_bar(progress / repeats)
                elif isinstance(data, EndData):
                    # We've done a single acquisition, check ok and return
                    assert data.reason == EndReason.OK, data.reason
                    break

    finally:
        start = time.time()
        print("\nClosing file...", end=" ")
        # Stop and wait for the pipeline to complete
        stop_pipeline(pipeline)
        print(f"took {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    # One-shot run of a co-routine
    asyncio.run(hdf_queue_reporting())
