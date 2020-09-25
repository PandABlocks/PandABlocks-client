import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Get
from pandablocks.core import EndData, EndReason, FrameData, StartData

from .conftest import Rows


@pytest.fixture
async def asyncio_client():
    client = AsyncioClient("localhost")
    await client.connected()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_asyncio_get(dummy_server, asyncio_client: AsyncioClient):
    dummy_server.send.append("OK =something")
    response = await asyncio_client.send(Get("PCAP.ACTIVE"))
    assert response == b"something"
    assert dummy_server.received == ["PCAP.ACTIVE?"]


@pytest.mark.asyncio
async def test_asyncio_data(
    dummy_server, asyncio_client: AsyncioClient, slow_dump, dump_fields
):
    dummy_server.data = slow_dump
    events = []
    async for data in asyncio_client.data(frame_timeout=1):
        events.append(data)
        if len(events) == 7:
            break
    assert [
        StartData(dump_fields, 0, "Scaled", "Framed", 52),
        FrameData(Rows([0, 1, 1, 3, 5.6e-08, 1, 2])),
        FrameData(Rows([8, 2, 2, 6, 1.000000056, 2, 4])),
        FrameData(Rows([0, 3, 3, 9, 2.000000056, 3, 6])),
        FrameData(Rows([8, 4, 4, 12, 3.000000056, 4, 8])),
        FrameData(Rows([0, 5, 5, 15, 4.000000056, 5, 10])),
        EndData(5, EndReason.DISARMED),
    ] == events
