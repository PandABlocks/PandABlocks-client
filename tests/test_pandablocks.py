from typing import Iterator

import pytest

from pandablocks.commands import (
    CommandException,
    FieldType,
    Get,
    GetBlockNumbers,
    GetFields,
    GetState,
    Put,
    SetState,
    is_multiline_command,
)
from pandablocks.connections import ControlConnection, DataConnection
from pandablocks.responses import Data
from tests.conftest import STATE_RESPONSES, STATE_SAVEFILE


def get_responses(conn: ControlConnection, received=b""):
    assert not conn.receive_bytes(received)
    return list(conn.responses())


def test_connection_gets_split_value():
    conn = ControlConnection()
    cmd = Get("PCAP.ACTIVE")
    assert conn.send(cmd) == b"PCAP.ACTIVE?\n"
    assert not get_responses(conn, b"OK =1")
    assert get_responses(conn, b"\nAnySpamWeLike") == [(cmd, "1")]


def test_connection_gets_muliline():
    conn = ControlConnection()
    cmd = Get("SEQ1.TABLE")
    assert conn.send(cmd) == b"SEQ1.TABLE?\n"
    assert not get_responses(conn, b"!1048576\n!0\n!10")
    assert get_responses(conn, b"00\n!1000\n.\n") == [
        (cmd, ["1048576", "0", "1000", "1000"])
    ]


def test_connect_put_single_line():
    conn = ControlConnection()
    cmd = Put("PCAP.TRIG", "PULSE1.OUT")
    assert conn.send(cmd) == b"PCAP.TRIG=PULSE1.OUT\n"
    assert get_responses(conn, b"OK\n") == [(cmd, None)]


class ACommandException(Exception):
    # Compare equal to a CommandException with the same message
    def __eq__(self, other):
        return isinstance(other, CommandException) and other.args == self.args


def test_put_fails_with_single_line_exception():
    conn = ControlConnection()
    cmd = Put("PCAP.blah", "something")
    assert conn.send(cmd) == b"PCAP.blah=something\n"
    assert get_responses(conn, b"ERR No such field\n") == [
        (
            cmd,
            ACommandException(
                "Put(field='PCAP.blah', value='something') -> ERR No such field"
            ),
        )
    ]


def test_put_fails_with_multiline_exception():
    conn = ControlConnection()
    cmd = Put("PCAP.blah", "something")
    assert conn.send(cmd) == b"PCAP.blah=something\n"
    assert get_responses(conn, b"!This is bad\n!Very bad\n!Don't do this\n.\n") == [
        (
            cmd,
            ACommandException(
                """\
Put(field='PCAP.blah', value='something') ->
    This is bad
    Very bad
    Don't do this"""
            ),
        )
    ]


def test_connect_put_multi_line():
    conn = ControlConnection()
    cmd = Put("SEQ1.TABLE", ["1048576", "0", "1000", "1000"])
    assert conn.send(cmd) == b"SEQ1.TABLE<\n1048576\n0\n1000\n1000\n\n"
    assert get_responses(conn, b"OK\n") == [(cmd, None)]


def test_get_block_numbers():
    conn = ControlConnection()
    cmd = GetBlockNumbers()
    assert conn.send(cmd) == b"*BLOCKS?\n"
    responses = get_responses(conn, b"!PCAP 1\n!LUT 8\n.\n")
    assert responses == [(cmd, {"PCAP": 1, "LUT": 8})]
    assert list(responses[0][1]) == ["LUT", "PCAP"]


def test_get_fields():
    conn = ControlConnection()
    cmd = GetFields("LUT")
    assert conn.send(cmd) == b"LUT.*?\n"
    responses = get_responses(conn, b"!TYPEA 5 param enum\n!INPA 1 bit_mux\n.\n")
    assert responses == [
        (
            cmd,
            dict(
                INPA=FieldType(type="bit_mux"),
                TYPEA=FieldType(type="param", subtype="enum"),
            ),
        )
    ]
    assert list(responses[0][1]) == ["INPA", "TYPEA"]


def test_save():
    conn = ControlConnection()
    cmd = GetState()
    assert (
        conn.send(cmd)
        == b"*CHANGES.ATTR?\n*CHANGES.CONFIG?\n*CHANGES.TABLE?\n*CHANGES.METADATA?\n"
    )
    response_bytes = "\n".join(STATE_RESPONSES).encode() + b"\n"
    assert (
        conn.receive_bytes(response_bytes[:109])
        == b"Table.B?\nMultiLineMeta1?\nMultiLineMeta2?\n"
    )
    assert not get_responses(conn)
    assert get_responses(conn, response_bytes[109:]) == [(cmd, STATE_SAVEFILE)]


def test_load():
    conn = ControlConnection()
    cmd = SetState(STATE_SAVEFILE)
    assert conn.send(cmd) == ("\n".join(STATE_SAVEFILE)).encode() + b"\n"
    response_bytes = "\n".join(["OK"] * 7).encode() + b"\n"
    assert get_responses(conn, response_bytes) == [(cmd, None)]


def get_data(it: Iterator[bytes]) -> Iterator[Data]:
    conn = DataConnection()
    assert conn.connect(scaled=True) == b"XML FRAMED SCALED\n"
    for received in it:
        yield from conn.receive_bytes(received)


def test_slow_data_collection(slow_dump, slow_dump_expected):
    responses = list(get_data(slow_dump))
    assert slow_dump_expected == responses


def test_fast_data_collection(fast_dump, fast_dump_expected):
    responses = list(get_data(fast_dump))
    assert fast_dump_expected == responses
    assert responses[2].column_names == (
        "PCAP.BITS2.Value",
        "COUNTER1.OUT.Min",
        "COUNTER1.OUT.Max",
        "COUNTER3.OUT.Value",
        "PCAP.TS_START.Value",
        "COUNTER1.OUT.Mean",
        "COUNTER2.OUT.Mean",
    )
    assert responses[2].data["COUNTER1.OUT.Mean"] == pytest.approx(range(1, 11))


def test_is_multiline_command():
    assert is_multiline_command("SEQ.TABLE<")
    assert is_multiline_command("SEQ.TABLE<B")
    assert is_multiline_command("*METADATA.DESIGN<")
    assert not is_multiline_command("SEQ.TABLE?")
    assert not is_multiline_command("*METADATA.DESIGN?")
    assert not is_multiline_command("*METADATA.DESIGN=B<B<B<B?")
