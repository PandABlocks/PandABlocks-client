import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Get


@dataclass
class ServerIO:
    received: List[str] = field(default_factory=list)
    send: Deque[str] = field(default_factory=deque)


@pytest.fixture
async def dummy_server():
    io = ServerIO()

    async def handle_echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        response = await reader.read(4096)
        io.received += response.decode().splitlines()
        writer.write((io.send.popleft() + "\n").encode())
        await writer.drain()

    server = await asyncio.start_server(handle_echo, "127.0.0.1", 8888)
    yield io
    server.close()
    await server.wait_closed()


@pytest.fixture
async def asyncio_client():
    client = AsyncioClient("localhost")
    await client.connected()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_asyncio_get(dummy_server: ServerIO, asyncio_client: AsyncioClient):
    dummy_server.send.append("OK =something")
    response = await asyncio_client.send(Get("PCAP.ACTIVE"))
    assert response == b"something"
    assert dummy_server.received == ["PCAP.ACTIVE?"]
