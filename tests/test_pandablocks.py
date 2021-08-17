from typing import Iterator, OrderedDict

import pytest

from pandablocks.commands import (
    CommandException,
    Get,
    GetBlockInfo,
    GetFieldInfo,
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
from pandablocks.responses import (
    BitMuxFieldInfo,
    BlockInfo,
    Data,
    EnumFieldInfo,
    ExtOutFieldInfo,
    UintFieldInfo,
)
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
    assert conn.receive_bytes(b"!PCAP 1\n!LUT 8\n.\n") == b"*DESC.PCAP?\n*DESC.LUT?\n"

    # First of the *DESC.{block}? yields
    assert (
        conn.receive_bytes(b"OK =Description for PCAP field\n") == b""
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
    assert get_responses(conn, b"OK =Description for LUT field\n") == [
        (
            cmd,
            ordered_dict,
        ),
    ]


def test_get_block_info_skip_description():
    """Test that the skip_description flag causes GetBlockInfo to not retrieve
    descriptions"""
    conn = ControlConnection()
    cmd = GetBlockInfo(skip_description=True)
    assert conn.send(cmd) == b"*BLOCKS?\n"

    ordered_dict = OrderedDict(
        [
            ("PCAP", BlockInfo(number=1, description=None)),
        ]
    )
    # Only a yield for the BLOCKS.* request, not for description as well
    assert get_responses(conn, b"!PCAP 1\n.\n") == [(cmd, ordered_dict)]


def test_get_block_info_error():
    """Test that any errors from *BLOCKS command are correctly reported"""
    conn = ControlConnection()
    cmd = GetBlockInfo()
    assert conn.send(cmd) == b"*BLOCKS?\n"

    # Provide error from PandA server
    assert conn.receive_bytes(b"ERR Cannot read blocks\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            ACommandException(
                "GetBlockInfo(skip_description=False) -> ERR Cannot read blocks"
            ),
        )
    ]


def test_get_block_info_desc_err():
    """Test when the DESC command returns an error"""
    conn = ControlConnection()
    cmd = GetBlockInfo()
    assert conn.send(cmd) == b"*BLOCKS?\n"

    # Respond to first yield, the return from the BLOCKS? command
    assert conn.receive_bytes(b"!PCAP 1\n.\n") == b"*DESC.PCAP?\n"

    # First of the *DESC.{block}? yields
    assert (
        conn.receive_bytes(b"ERR could not get description\n") == b""
    )  # No data returned as there's still one outstanding request

    assert get_responses(conn) == [
        (
            cmd,
            ACommandException(
                "GetBlockInfo(skip_description=False) -> ERR could not get description"
            ),
        )
    ]


def test_get_fields():
    """Simple test case for GetFieldInfo"""
    conn = ControlConnection()
    cmd = GetFieldInfo("LUT")
    assert conn.send(cmd) == b"LUT.*?\n"

    # First yield, the response to "LUT.*?"
    assert (
        conn.receive_bytes(b"!TYPEA 5 param enum\n!INPA 1 bit_mux\n.\n")
        == b"*ENUMS.LUT.TYPEA?\n*DESC.LUT.TYPEA?\nLUT1.INPA.MAX_DELAY?\n"
        + b"*ENUMS.LUT.INPA?\n*DESC.LUT.INPA?\n"
    )

    # Responses to the 2 *DESC, 2 *ENUM, and MAX_DELAY commands
    responses = [
        b"!Input-Level\n!Pulse-On-Rising-Edge\n.\n",
        b"OK =Source of the value of A for calculation\n",
        b"OK =10\n",
        b"!TTLIN1.VAL\n!LVDSIN1.VAL\n.\n",
        b"OK =Input A\n",
    ]
    for response in responses:
        assert (
            conn.receive_bytes(response) == b""
        )  # Expect no bytes back as none of these trigger further commands

    assert get_responses(conn) == [
        (
            cmd,
            {
                "INPA": BitMuxFieldInfo(
                    type="bit_mux",
                    subtype=None,
                    description="Input A",
                    labels=["TTLIN1.VAL", "LVDSIN1.VAL"],
                    max_delay=10,
                ),
                "TYPEA": EnumFieldInfo(
                    type="param",
                    subtype="enum",
                    description="Source of the value of A for calculation",
                    labels=["Input-Level", "Pulse-On-Rising-Edge"],
                ),
            },
        )
    ]


def test_get_fields_type_ext_out():
    """Test for field type == ext_out, ensuring we add .CAPTURE to the end of the
    *ENUMS command"""
    conn = ControlConnection()
    cmd = GetFieldInfo("PCAP")
    assert conn.send(cmd) == b"PCAP.*?\n"

    # First yield, the response to "PCAP.*?"
    assert (
        conn.receive_bytes(b"!SAMPLES 9 ext_out samples\n.\n")
        == b"*ENUMS.PCAP.SAMPLES.CAPTURE?\n*DESC.PCAP.SAMPLES?\n"
    )

    # Responses to the *DESC and *ENUM commands
    responses = [
        b"!No\n!Value\n.\n",
        b"OK =Number of gated samples in the current capture\n",
    ]
    for response in responses:
        assert (
            conn.receive_bytes(response) == b""
        )  # Expect no bytes back as none of these trigger further commands

    assert get_responses(conn) == [
        (
            cmd,
            {
                "SAMPLES": ExtOutFieldInfo(
                    type="ext_out",
                    subtype="samples",
                    description="Number of gated samples in the current capture",
                    labels=["No", "Value"],
                )
            },
        )
    ]


def test_get_fields_skip_description():
    """Test that the skip_description flag causes no description to be retrieved
    for the field"""
    conn = ControlConnection()
    cmd = GetFieldInfo("PCAP", True)
    assert conn.send(cmd) == b"PCAP.*?\n"

    assert (
        conn.receive_bytes(b"!SAMPLES 9 ext_out samples\n.\n")
        == b"*ENUMS.PCAP.SAMPLES.CAPTURE?\n"
    )

    assert conn.receive_bytes(b"!No\n!Value\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            {
                "SAMPLES": ExtOutFieldInfo(
                    type="ext_out",
                    subtype="samples",
                    description=None,
                    labels=["No", "Value"],
                )
            },
        )
    ]


def test_get_fields_non_existant_block():
    """Test that querying for an unknown block returns a sensible error"""
    conn = ControlConnection()
    cmd = GetFieldInfo("FOO")
    assert conn.send(cmd) == b"FOO.*?\n"

    # Provide the error string the PandA would provide
    assert conn.receive_bytes(b"ERR No such block\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            ACommandException(
                "GetFieldInfo(block='FOO', skip_description=False) -> ERR No such block"
            ),
        )
    ]


@pytest.mark.parametrize(
    "field_type, field_subtype, expected_get_string, responses, expected_field_info",
    [
        (
            "param",
            "uint",
            "TEST1.TEST_FIELD.MAX?\n",
            ["OK =10\n"],
            UintFieldInfo("param", "uint", max=10),
        )
    ],
)
def test_get_fields_parameterized_type(
    field_type, field_subtype, expected_get_string, responses, expected_field_info
):
    """Test every field type-subtype pair that has a defined function
    and confirm it sends the expected Get commands to the server"""
    conn = ControlConnection()
    cmd = GetFieldInfo("TEST", skip_description=True)
    assert conn.send(cmd) == b"TEST.*?\n"

    field_definition_str = f"!TEST_FIELD 1 {field_type} {field_subtype}\n.\n"
    assert (
        conn.receive_bytes(field_definition_str.encode())
        == expected_get_string.encode()
    )

    for response in responses:
        assert conn.receive_bytes(response.encode()) == b""

    assert get_responses(conn) == [
        (
            cmd,
            {"TEST_FIELD": expected_field_info},
        )
    ]


# TODO: What happens when returned field type and/or subype is garbage?
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
        conn.receive_bytes(response_bytes[:107])
        == b"Table.B?\nMultiLineMeta1?\nMultiLineMeta2?\n"
    )
    assert not get_responses(conn)
    assert get_responses(conn, response_bytes[107:]) == [(cmd, STATE_SAVEFILE)]


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
