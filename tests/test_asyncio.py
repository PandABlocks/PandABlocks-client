import asyncio

import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import CommandException, Get, Put

from .conftest import DummyServer

# Timeout in seconds
TIMEOUT = 3


@pytest.mark.asyncio
async def test_asyncio_get(dummy_server_async):
    dummy_server_async.send.append("OK =something")
    async with AsyncioClient("localhost") as client:
        response = await asyncio.wait_for(client.send(Get("PCAP.ACTIVE")), timeout=1)
    assert response == "something"
    assert dummy_server_async.received == ["PCAP.ACTIVE?"]


@pytest.mark.asyncio
async def test_asyncio_bad_put_raises(dummy_server_async):
    dummy_server_async.send.append("ERR no such field")
    async with AsyncioClient("localhost") as client:
        with pytest.raises(CommandException) as cm:
            await asyncio.wait_for(client.send(Put("PCAP.thing", 1)), timeout=1)
        assert str(cm.value) == "Put(field='PCAP.thing', value=1) -> ERR no such field"
    assert dummy_server_async.received == ["PCAP.thing=1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("disarmed", [True, False])
@pytest.mark.parametrize("flush_period", [0.1, None])
async def test_asyncio_data(
    dummy_server_async, fast_dump, fast_dump_expected, disarmed, flush_period
):
    if not disarmed:
        # simulate getting the data without the END marker as if arm was not pressed
        fast_dump = (x.split(b"END")[0] for x in fast_dump)
        fast_dump_expected = list(fast_dump_expected)[:-1]
    dummy_server_async.data = fast_dump
    events = []
    async with AsyncioClient("localhost") as client:
        async for data in client.data(frame_timeout=1, flush_period=flush_period):
            events.append(data)
            if len(events) == len(fast_dump_expected):
                break
    assert fast_dump_expected == events


async def test_asyncio_data_timeout(dummy_server_async, fast_dump):
    dummy_server_async.data = fast_dump
    async with AsyncioClient("localhost") as client:
        with pytest.raises(asyncio.TimeoutError, match="No data received for 0.1s"):
            async for data in client.data(frame_timeout=0.1):
                "This goes forever, when it runs out of data we will get our timeout"


@pytest.mark.asyncio
async def test_asyncio_connects(dummy_server_async: DummyServer):
    async with AsyncioClient("localhost") as client:
        assert client.is_connected()

    assert not client.is_connected()


@pytest.mark.asyncio
async def test_asyncio_client_uncontactable():
    """Test that a client raises an exception when the remote end is not
    contactable"""
    client = AsyncioClient("localhost")
    with pytest.raises(OSError):
        await client.connect()


@pytest.mark.asyncio
async def test_asyncio_client_fails_when_cannot_drain(dummy_server_async: DummyServer):
    """Test that we don't hang indefinitely when failing to drain data from the OS
    send buffer"""

    # Note this value is probably OS-dependant. I found it experimentally.
    large_data = b"ABC" * 100000000

    client = AsyncioClient("localhost")
    await client.connect()
    await dummy_server_async.close()
    with pytest.raises(asyncio.TimeoutError):
        await client._ctrl_stream.write_and_drain(large_data, timeout=TIMEOUT)

    # Can't use client.close() as it gets endlessly stuck. Do the important part.
    assert client._ctrl_task
    client._ctrl_task.cancel()
