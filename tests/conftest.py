import asyncio
import threading
from collections import deque
from io import BufferedReader
from pathlib import Path
from typing import Deque, Iterator, List

import numpy as np
import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.blocking import BlockingClient
from pandablocks.connection import Buffer
from pandablocks.responses import DataField, EndData, EndReason, FrameData, StartData


def chunked_read(f: BufferedReader, size: int) -> Iterator[bytes]:
    data = f.read(size)
    while data:
        yield data
        data = f.read(size)


@pytest.fixture
def slow_dump():
    with open(Path(__file__).parent / "slow_dump.txt", "rb") as f:
        # Simulate small chunked read
        yield chunked_read(f, 50)


@pytest.fixture
def fast_dump():
    with open(Path(__file__).parent / "fast_dump.txt", "rb") as f:
        # Simulate larger chunked read
        yield chunked_read(f, 500)


@pytest.fixture
def raw_dump():
    with open(Path(__file__).parent / "raw_dump.txt", "rb") as f:
        # Simulate largest chunked read
        yield chunked_read(f, 200000)


DUMP_FIELDS = [
    DataField(
        name="PCAP.BITS2", type=np.uint32, capture="Value", scale=1, offset=0, units="",
    ),
    DataField(
        name="COUNTER1.OUT", type=np.double, capture="Min", scale=1, offset=0, units="",
    ),
    DataField(
        name="COUNTER1.OUT", type=np.double, capture="Max", scale=1, offset=0, units="",
    ),
    DataField(
        name="COUNTER3.OUT",
        type=np.double,
        capture="Value",
        scale=1,
        offset=0,
        units="",
    ),
    DataField(
        name="PCAP.TS_START",
        type=np.double,
        capture="Value",
        scale=8e-09,
        offset=0,
        units="s",
    ),
    DataField(
        name="COUNTER1.OUT",
        type=np.double,
        capture="Mean",
        scale=1,
        offset=0,
        units="",
    ),
    DataField(
        name="COUNTER2.OUT",
        type=np.double,
        capture="Mean",
        scale=1,
        offset=0,
        units="",
    ),
]


class Rows:
    def __init__(self, *rows):
        self.rows = rows

    def __eq__(self, o):
        same = o.tolist() == [pytest.approx(row) for row in self.rows]
        return same


@pytest.fixture
def slow_dump_expected():
    yield [
        StartData(DUMP_FIELDS, 0, "Scaled", "Framed", 52),
        FrameData(Rows([0, 1, 1, 3, 5.6e-08, 1, 2])),
        FrameData(Rows([8, 2, 2, 6, 1.000000056, 2, 4])),
        FrameData(Rows([0, 3, 3, 9, 2.000000056, 3, 6])),
        FrameData(Rows([8, 4, 4, 12, 3.000000056, 4, 8])),
        FrameData(Rows([0, 5, 5, 15, 4.000000056, 5, 10])),
        EndData(5, EndReason.DISARMED),
    ]


@pytest.fixture
def fast_dump_expected():
    yield [
        StartData(DUMP_FIELDS, 0, "Scaled", "Framed", 52),
        FrameData(
            Rows(
                [0, 1, 1, 3, 5.6e-08, 1, 2],
                [0, 2, 2, 6, 0.010000056, 2, 4],
                [8, 3, 3, 9, 0.020000056, 3, 6],
                [8, 4, 4, 12, 0.030000056, 4, 8],
                [8, 5, 5, 15, 0.040000056, 5, 10],
                [8, 6, 6, 18, 0.050000056, 6, 12],
                [8, 7, 7, 21, 0.060000056, 7, 14],
                [8, 8, 8, 24, 0.070000056, 8, 16],
                [8, 9, 9, 27, 0.080000056, 9, 18],
                [8, 10, 10, 30, 0.090000056, 10, 20],
            )
        ),
        FrameData(
            Rows(
                [0, 11, 11, 33, 0.100000056, 11, 22],
                [8, 12, 12, 36, 0.110000056, 12, 24],
                [8, 13, 13, 39, 0.120000056, 13, 26],
                [8, 14, 14, 42, 0.130000056, 14, 28],
                [8, 15, 15, 45, 0.140000056, 15, 30],
                [8, 16, 16, 48, 0.150000056, 16, 32],
                [8, 17, 17, 51, 0.160000056, 17, 34],
                [8, 18, 18, 54, 0.170000056, 18, 36],
                [8, 19, 19, 57, 0.180000056, 19, 38],
                [0, 20, 20, 60, 0.190000056, 20, 40],
                [8, 21, 21, 63, 0.200000056, 21, 42],
            )
        ),
        FrameData(
            Rows(
                [8, 22, 22, 66, 0.210000056, 22, 44],
                [8, 23, 23, 69, 0.220000056, 23, 46],
                [8, 24, 24, 72, 0.230000056, 24, 48],
                [8, 25, 25, 75, 0.240000056, 25, 50],
                [8, 26, 26, 78, 0.250000056, 26, 52],
                [8, 27, 27, 81, 0.260000056, 27, 54],
                [8, 28, 28, 84, 0.270000056, 28, 56],
                [0, 29, 29, 87, 0.280000056, 29, 58],
                [8, 30, 30, 90, 0.290000056, 30, 60],
                [8, 31, 31, 93, 0.300000056, 31, 62],
            )
        ),
        FrameData(
            Rows(
                [8, 32, 32, 96, 0.310000056, 32, 64],
                [8, 33, 33, 99, 0.320000056, 33, 66],
                [8, 34, 34, 102, 0.330000056, 34, 68],
                [8, 35, 35, 105, 0.340000056, 35, 70],
                [8, 36, 36, 108, 0.350000056, 36, 72],
                [8, 37, 37, 111, 0.360000056, 37, 74],
                [0, 38, 38, 114, 0.370000056, 38, 76],
                [8, 39, 39, 117, 0.380000056, 39, 78],
                [8, 40, 40, 120, 0.390000056, 40, 80],
                [8, 41, 41, 123, 0.400000056, 41, 82],
            )
        ),
        FrameData(
            Rows(
                [8, 42, 42, 126, 0.410000056, 42, 84],
                [8, 43, 43, 129, 0.420000056, 43, 86],
                [8, 44, 44, 132, 0.430000056, 44, 88],
                [8, 45, 45, 135, 0.440000056, 45, 90],
                [8, 46, 46, 138, 0.450000056, 46, 92],
                [0, 47, 47, 141, 0.460000056, 47, 94],
                [8, 48, 48, 144, 0.470000056, 48, 96],
                [8, 49, 49, 147, 0.480000056, 49, 98],
                [8, 50, 50, 150, 0.490000056, 50, 100],
                [8, 51, 51, 153, 0.500000056, 51, 102],
            )
        ),
        FrameData(
            Rows(
                [8, 52, 52, 156, 0.510000056, 52, 104],
                [8, 53, 53, 159, 0.520000056, 53, 106],
                [8, 54, 54, 162, 0.530000056, 54, 108],
                [8, 55, 55, 165, 0.540000056, 55, 110],
                [0, 56, 56, 168, 0.550000056, 56, 112],
                [8, 57, 57, 171, 0.560000056, 57, 114],
                [8, 58, 58, 174, 0.570000056, 58, 116],
            )
        ),
        EndData(58, EndReason.DISARMED),
    ]


class DummyServer:
    def __init__(self):
        self.received: List[str] = []
        self.send: Deque[str] = deque()
        self.data = b""

    async def handle_ctrl(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        buf = Buffer()
        is_multiline = False
        while True:
            received = await reader.read(4096)
            if not received:
                break
            buf += received
            for line in buf:
                self.received.append(line.decode())
                if line.endswith(b"<"):
                    is_multiline = True
                if not is_multiline or not line:
                    is_multiline = False
                    to_send = self.send.popleft() + "\n"
                    writer.write(to_send.encode())
                    await writer.drain()

    async def handle_data(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        # oneshot data writer
        await reader.read(4096)
        for data in self.data:
            await asyncio.sleep(0.1)
            writer.write(data)
            await writer.drain()

    async def open(self):
        self._ctrl_server = await asyncio.start_server(
            self.handle_ctrl, "127.0.0.1", 8888
        )
        self._data_server = await asyncio.start_server(
            self.handle_data, "127.0.0.1", 8889
        )

    async def close(self):
        self._ctrl_server.close()
        self._data_server.close()
        await self._ctrl_server.wait_closed()
        await self._data_server.wait_closed()


@pytest.fixture
async def dummy_server_async():
    server = DummyServer()
    await server.open()
    yield server
    await server.close()


@pytest.fixture
def dummy_server_in_thread():
    loop = asyncio.new_event_loop()
    server = DummyServer()
    t = threading.Thread(target=loop.run_forever)
    t.start()
    f = asyncio.run_coroutine_threadsafe(server.open(), loop)
    f.result()
    yield server
    asyncio.run_coroutine_threadsafe(server.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop())
    t.join()


@pytest.fixture
def blocking_client():
    client = BlockingClient("localhost")
    yield client
    client.close()


@pytest.fixture
async def asyncio_client():
    client = AsyncioClient("localhost")
    await client.connect()
    yield client
    await client.close()
