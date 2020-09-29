from pandablocks.blocking import BlockingClient
from pandablocks.commands import Get


def test_blocking_get(dummy_server_in_thread, blocking_client: BlockingClient):
    dummy_server_in_thread.send.append("OK =something")
    response = blocking_client.send(Get("PCAP.ACTIVE"))
    assert response == b"something"
    assert dummy_server_in_thread.received == ["PCAP.ACTIVE?"]


def test_blocking_data(
    dummy_server_in_thread,
    blocking_client: BlockingClient,
    slow_dump,
    slow_dump_expected,
):
    dummy_server_in_thread.data = slow_dump
    events = []
    for data in blocking_client.data(frame_timeout=1):
        events.append(data)
        if len(events) == 7:
            break
    assert slow_dump_expected == events
