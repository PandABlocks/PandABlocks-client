import pytest

from pandablocks.blocking import BlockingClient
from pandablocks.commands import CommandException, Get, Put


def test_blocking_get(dummy_server_in_thread):
    dummy_server_in_thread.send.append("OK =something")
    with BlockingClient("localhost") as client:
        response = client.send(Get("PCAP.ACTIVE"), timeout=1)
    assert response == "something"
    assert dummy_server_in_thread.received == ["PCAP.ACTIVE?"]


def test_blocking_bad_put_raises(dummy_server_in_thread):
    dummy_server_in_thread.send.append("ERR no such field")
    with BlockingClient("localhost") as client:
        with pytest.raises(CommandException) as cm:
            client.send(Put("PCAP.thing", 1), timeout=1)
        assert str(cm.value) == "Put(field='PCAP.thing', value=1) -> ERR no such field"
    assert dummy_server_in_thread.received == ["PCAP.thing=1"]


def test_blocking_data(
    dummy_server_in_thread,
    slow_dump,
    slow_dump_expected,
):
    dummy_server_in_thread.data = slow_dump
    events = []
    with BlockingClient("localhost") as client:
        for data in client.data(frame_timeout=1):
            events.append(data)
            if len(events) == 8:
                break
    assert slow_dump_expected == events
