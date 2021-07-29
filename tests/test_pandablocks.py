from typing import Iterator, OrderedDict

import pytest

from pandablocks.commands import (
    BlockInfo,
    CommandException,
    FieldType,
    Get,
    GetBlockInfo,
    GetFields,
    GetPcapBitsLabels,
    GetState,
    Put,
    SetState,
    is_multiline_command,
)
from pandablocks.connections import (
    ControlConnection,
    DataConnection,
    NoContextAvailable,
)
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


def test_get_block_info():
    conn = ControlConnection()
    cmd = GetBlockInfo()
    assert conn.send(cmd) == b"*BLOCKS?\n"

    # Respond to first yield, the return from the BLOCKS? command
    assert conn.receive_bytes(b"!PCAP 1\n!LUT 8\n.\n")

    # First of the *DESC.{block}? yields
    assert (
        conn.receive_bytes(b"Description for PCAP field\n") == b""
    )  # No data returned as there's still one outstanding request

    # Create an OrderedDict of the output to test key order - that won't happen
    # with a regular dict
    ordered_dict = OrderedDict(
        [
            ("LUT", BlockInfo(number=8, description="Description for LUT field")),
            ("PCAP", BlockInfo(number=1, description="Description for PCAP field")),
        ]
    )

    # Second and last of the *DESC.{block}? yields - as this is the last response we
    # can call get_responses to also get the overall result
    assert not get_responses(conn)
    assert get_responses(conn, b"Description for LUT field\n") == [
        (cmd, ordered_dict,),
    ]


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


def test_get_pcap_bits_labels():
    """Simple working testcase for GetPcapBitsLabels"""

    # PandA's return data when it receives "PCAP.*?"
    PCAP_RETURN = [
        "!BITS2 12 ext_out bits",
        "!SHIFT_SUM 4 param uint",
        "!BITS0 10 ext_out bits",
        ".",
    ]

    # PandA's return data when it receives "PCAP.BITS2.BITS?"
    BITS2_RETURN = ["!PCOMP2.OUT", "!PGEN1.ACTIVE", "!PGEN2.ACTIVE", "!PULSE1.OUT", "."]

    # PandA's return data when it receives "PCAP.BITS0.BITS?"
    BITS0_RETURN = [
        "!SFP3_SYNC_IN.BIT8",
        "!SFP3_SYNC_IN.BIT9",
        "!SFP3_SYNC_IN.BIT10",
        ".",
    ]

    conn = ControlConnection()
    cmd = GetPcapBitsLabels()
    assert conn.send(cmd) == b"PCAP.*?\n"

    # First yield, requesting response for PCAP.*?
    response_bytes = "\n".join(PCAP_RETURN).encode() + b"\n"
    assert conn.receive_bytes(response_bytes) == b"PCAP.BITS2.BITS?\nPCAP.BITS0.BITS?\n"

    # First of the .BITS? yields
    response_bytes = "\n".join(BITS2_RETURN).encode() + b"\n"
    assert (
        conn.receive_bytes(response_bytes) == b""
    )  # No data returned as there's still one outstanding request

    # Second of the .BITS? yields - as this is the last response we can call
    # get_responses to also get the overall result
    response_bytes = "\n".join(BITS0_RETURN).encode() + b"\n"

    assert not get_responses(conn)
    assert get_responses(conn, response_bytes) == [
        (
            cmd,
            {
                "PCAP.BITS0": [
                    "SFP3_SYNC_IN.BIT8",
                    "SFP3_SYNC_IN.BIT9",
                    "SFP3_SYNC_IN.BIT10",
                ],
                "PCAP.BITS2": [
                    "PCOMP2.OUT",
                    "PGEN1.ACTIVE",
                    "PGEN2.ACTIVE",
                    "PULSE1.OUT",
                ],
            },
        )
    ]


def test_get_pcap_bits_labels_no_bits_fields():
    """Test we get no response when no BITS fields are returned by the PandA"""

    # PandA's return data when it receives "PCAP.*?"
    PCAP_RETURN = [
        "!SHIFT_SUM 4 param uint",
        "!ACTIVE 5 bit_out",
        "!ENABLE 0 bit_mux",
        ".",
    ]
    conn = ControlConnection()
    cmd = GetPcapBitsLabels()
    assert conn.send(cmd) == b"PCAP.*?\n"

    # As there are no BITS fields in the PCAP return, expect no response
    response_bytes = "\n".join(PCAP_RETURN).encode() + b"\n"
    assert conn.receive_bytes(response_bytes) == b""


def test_expected_exception_when_receive_without_send():
    """Test that calling receive_bytes() without first calling send() raises the
    expected exception"""

    conn = ControlConnection()
    with pytest.raises(NoContextAvailable):
        conn.receive_bytes(b"abc\n")


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
