from typing import Iterator

import pytest

from pandablocks.commands import FieldType, Get, GetBlockNumbers, GetFields, Put
from pandablocks.core import ControlConnection, Data, DataConnection


def test_connection_gets_split_value():
    conn = ControlConnection()
    cmd = Get("PCAP.ACTIVE")
    assert conn.send(cmd) == b"PCAP.ACTIVE?\n"
    assert list(conn.receive_bytes(b"OK =1")) == []
    assert list(conn.receive_bytes(b"\nAnySpamWeLike")) == [(cmd, b"1")]


def test_connection_gets_muliline():
    conn = ControlConnection()
    cmd = Get("SEQ1.TABLE")
    assert conn.send(cmd) == b"SEQ1.TABLE?\n"
    assert list(conn.receive_bytes(b"!1048576\n!0\n!10")) == []
    assert list(conn.receive_bytes(b"00\n!1000\n.\n")) == [
        (cmd, [b"1048576", b"0", b"1000", b"1000"])
    ]


def test_connect_put_single_line():
    conn = ControlConnection()
    cmd = Put("PCAP.TRIG", "PULSE1.OUT")
    assert conn.send(cmd) == b"PCAP.TRIG=PULSE1.OUT\n"
    assert list(conn.receive_bytes(b"OK\n")) == [(cmd, None)]


def test_connect_put_multi_line():
    conn = ControlConnection()
    cmd = Put("SEQ1.TABLE", ["1048576", "0", "1000", "1000"])
    assert conn.send(cmd) == b"SEQ1.TABLE<\n1048576\n0\n1000\n1000\n\n"
    assert list(conn.receive_bytes(b"OK\n")) == [(cmd, None)]


def test_get_block_numbers():
    conn = ControlConnection()
    cmd = GetBlockNumbers()
    assert conn.send(cmd) == b"*BLOCKS?\n"
    events = list(conn.receive_bytes(b"!PCAP 1\n!LUT 8\n.\n"))
    assert events == [(cmd, {"PCAP": 1, "LUT": 8})]
    assert list(events[0][1]) == ["LUT", "PCAP"]


def test_get_fields():
    conn = ControlConnection()
    cmd = GetFields("LUT")
    assert conn.send(cmd) == b"LUT.*?\n"
    events = list(conn.receive_bytes(b"!TYPEA 5 param enum\n!INPA 1 bit_mux\n.\n"))
    assert events == [
        (
            cmd,
            dict(
                INPA=FieldType(type="bit_mux"),
                TYPEA=FieldType(type="param", subtype="enum"),
            ),
        )
    ]
    assert list(events[0][1]) == ["INPA", "TYPEA"]


def get_data(it: Iterator[bytes]) -> Iterator[Data]:
    conn = DataConnection()
    assert conn.connect(scaled=True) == b"XML FRAMED SCALED\n"
    for received in it:
        yield from conn.receive_bytes(received)
        fd = conn.flush()
        if fd:
            yield fd


def test_slow_data_collection(slow_dump, slow_dump_expected):
    events = list(get_data(slow_dump))
    assert slow_dump_expected == events


def test_fast_data_collection(fast_dump, fast_dump_expected):
    events = list(get_data(fast_dump))
    assert fast_dump_expected == events
    assert events[1].data["COUNTER1.OUT.Mean"] == pytest.approx(range(1, 11))
