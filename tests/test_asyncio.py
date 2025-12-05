import asyncio
import copy

import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import CommandError, Get, Put

# Timeout in seconds
TIMEOUT = 3


@pytest.mark.asyncio
async def test_asyncio_get(dummy_server_async):
    server = await dummy_server_async()
    server.send.append("OK =something")
    async with AsyncioClient("localhost") as client:
        response = await asyncio.wait_for(client.send(Get("PCAP.ACTIVE")), timeout=1)
    assert response == "something"
    assert server.received[1:] == ["PCAP.ACTIVE?"]


@pytest.mark.asyncio
async def test_asyncio_bad_put_raises(dummy_server_async):
    server = await dummy_server_async()
    server.send.append("ERR no such field")
    async with AsyncioClient("localhost") as client:
        with pytest.raises(CommandError) as cm:
            await asyncio.wait_for(client.send(Put("PCAP.thing", 1)), timeout=1)
        assert (
            str(cm.value) == "Put(field='PCAP.thing', value=1) raised error:\n"
            "AssertionError: 'PCAP.thing=1' -> 'ERR no such field'"
        )
    assert server.received[1:] == ["PCAP.thing=1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("disarmed", [True, False])
@pytest.mark.parametrize("flush_period", [0.1, None])
async def test_asyncio_data(
    dummy_server_async, fast_dump, fast_dump_expected, disarmed, flush_period
):
    server = await dummy_server_async()
    if not disarmed:
        # simulate getting the data without the END marker as if arm was not pressed
        fast_dump = (x.split(b"END")[0] for x in fast_dump)
        fast_dump_expected = list(fast_dump_expected)[:-1]
    server.data = fast_dump
    events = []
    async with AsyncioClient("localhost") as client:
        async for data in client.data(frame_timeout=1, flush_period=flush_period):
            events.append(data)
            if len(events) == len(fast_dump_expected):
                break
    assert fast_dump_expected == events


@pytest.mark.parametrize(
    "timing_params",
    [
        {},
        {
            "arm_time": "2024-03-05T20:27:12.607841574Z",
            "start_time": "2024-03-05T20:27:12.608875498Z",
        },
        {
            "arm_time": "2024-03-05T20:27:12.607841574Z",
            "start_time": "2024-03-05T20:27:12.608875498Z",
            "hw_time_offset_ns": 100555,
        },
    ],
)
async def test_asyncio_data_with_abs_timing(
    dummy_server_async,
    fast_dump_with_extra_header_params,
    fast_dump_expected,
    timing_params,
):
    """
    The test for handling of `arm_time`, `start_time` and `hw_time_offset_ns`
    parameters passed in the header. The test is reusing the existing `fast_dump`
    and `fast_dump_expected` by adding timing parameters to the header in
    the binary stream and replacing the expected `StartData` attributes with
    the expected values.
    """
    server = await dummy_server_async()
    server.data = fast_dump_with_extra_header_params(timing_params)
    events = []
    async with AsyncioClient("localhost") as client:
        async for data in client.data(frame_timeout=1):
            events.append(data)
            if len(events) == len(fast_dump_expected):
                break
    fast_dump_expected = list(fast_dump_expected)

    # Replace attributes in `StartData` with the expected values
    fast_dump_expected[1] = copy.deepcopy(fast_dump_expected[1])
    for attr_name in timing_params:
        setattr(fast_dump_expected[1], attr_name, timing_params[attr_name])

    assert fast_dump_expected == events


async def test_asyncio_data_timeout(dummy_server_async, fast_dump):
    server = await dummy_server_async()
    server.data = fast_dump
    async with AsyncioClient("localhost") as client:
        with pytest.raises(asyncio.TimeoutError, match="No data received for 0.1s"):
            async for _ in client.data(frame_timeout=0.1):
                "This goes forever, when it runs out of data we will get our timeout"


async def test_asyncio_empty_frame_error():
    dummy_data = [b"ABC"] * 10 + [b""]
    dummy_data_iter = iter(dummy_data)

    async def dummy_read(n):
        return dummy_data_iter.__next__()

    reader = asyncio.StreamReader()
    reader.read = dummy_read

    written = []

    class DummyControlStream:
        async def write_and_drain(self, data):
            written.append(data)

    class DummyControlConnection:
        def receive_bytes(self, data):
            return data

    client = AsyncioClient("localhost")
    client._ctrl_stream = DummyControlStream()
    client._ctrl_connection = DummyControlConnection()
    with pytest.raises(
        ConnectionError, match="Received an empty packet. Closing connection."
    ):
        await client._ctrl_read_forever(reader)
    assert written == dummy_data[:-1]


@pytest.mark.asyncio
async def test_asyncio_connects(dummy_server_async):
    await dummy_server_async()
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
async def test_asyncio_client_fails_when_cannot_drain(dummy_server_async):
    """Test that we don't hang indefinitely when failing to drain data from the OS
    send buffer"""
    await dummy_server_async()

    # Note this value is probably OS-dependant. I found it experimentally.
    large_data = b"ABC" * 100000000

    client = AsyncioClient("localhost")
    await client.connect()
    with pytest.raises(asyncio.TimeoutError):
        await client._ctrl_stream.write_and_drain(large_data, timeout=TIMEOUT)

    # Can't use client.close() as it gets endlessly stuck. Do the important part.
    assert client._ctrl_task
    client._ctrl_task.cancel()
