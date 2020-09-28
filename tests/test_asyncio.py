import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Get


@pytest.fixture
async def asyncio_client():
    client = AsyncioClient("localhost")
    await client.connect()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_asyncio_get(dummy_server_async, asyncio_client: AsyncioClient):
    dummy_server_async.send.append("OK =something")
    response = await asyncio_client.send(Get("PCAP.ACTIVE"))
    assert response == b"something"
    assert dummy_server_async.received == ["PCAP.ACTIVE?"]


@pytest.mark.asyncio
async def test_asyncio_data(dummy_server_async, fast_dump, fast_dump_expected):
    dummy_server_async.data = fast_dump
    events = []
    async for data in AsyncioClient("localhost").data(frame_timeout=1):
        events.append(data)
        if len(events) == 8:
            break
    assert fast_dump_expected == events
