import asyncio

import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import CommandException, Get, Put


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
async def test_asyncio_data(dummy_server_async, fast_dump, fast_dump_expected):
    dummy_server_async.data = fast_dump
    events = []
    async with AsyncioClient("localhost") as client:
        async for data in client.data(frame_timeout=1):
            events.append(data)
            if len(events) == 9:
                break
    assert fast_dump_expected == events
