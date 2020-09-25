import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List

import numpy as np
import pytest

from pandablocks.core import DataField


@pytest.fixture
def fast_dump():
    with open(Path(__file__).parent / "fast_dump.txt", "rb") as f:
        yield f.read()


@pytest.fixture
def slow_dump():
    with open(Path(__file__).parent / "slow_dump.txt", "rb") as f:
        yield f.read()


@pytest.fixture
def dump_fields():
    yield [
        DataField(
            name="PCAP.BITS2",
            type=np.uint32,
            capture="Value",
            scale=1,
            offset=0,
            units="",
        ),
        DataField(
            name="COUNTER1.OUT",
            type=np.double,
            capture="Min",
            scale=1,
            offset=0,
            units="",
        ),
        DataField(
            name="COUNTER1.OUT",
            type=np.double,
            capture="Max",
            scale=1,
            offset=0,
            units="",
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


@dataclass
class ServerIO:
    received: List[str] = field(default_factory=list)
    send: Deque[str] = field(default_factory=deque)
    data: bytes = b""


@pytest.fixture
async def dummy_server():
    io = ServerIO()

    async def handle_ctrl(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        response = await reader.read(4096)
        if response:
            io.received += response.decode().splitlines()
            writer.write((io.send.popleft() + "\n").encode())
            await writer.drain()

    async def handle_data(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # oneshot data writer
        await reader.read(4096)
        writer.write(io.data)
        await writer.drain()

    ctrl_server = await asyncio.start_server(handle_ctrl, "127.0.0.1", 8888)
    data_server = await asyncio.start_server(handle_data, "127.0.0.1", 8889)
    yield io
    ctrl_server.close()
    data_server.close()
    await ctrl_server.wait_closed()
    await data_server.wait_closed()
