from typing import Iterator, OrderedDict

import pytest

from pandablocks.commands import (
    Append,
    ChangeGroup,
    CommandException,
    Get,
    GetBlockInfo,
    GetChanges,
    GetFieldInfo,
    GetLine,
    GetMultiline,
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
    BitOutFieldInfo,
    BlockInfo,
    Changes,
    Data,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TableFieldDetails,
    TableFieldInfo,
    TimeFieldInfo,
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


def test_get_line():
    conn = ControlConnection()
    cmd = GetLine("PCAP.ACTIVE")
    assert conn.send(cmd) == b"PCAP.ACTIVE?\n"
    assert get_responses(conn, b"OK =1\n") == [(cmd, "1")]


def test_get_line_error_when_multiline():
    conn = ControlConnection()
    cmd = GetLine("PCAP.ACTIVE")
    assert conn.send(cmd) == b"PCAP.ACTIVE?\n"
    assert get_responses(conn, b"!ACTIVE 5 bit_out\n.\n") == [
        (
            cmd,
            ACommandException("GetLine(field='PCAP.ACTIVE') ->\n    ACTIVE 5 bit_out"),
        )
    ]


def test_get_line_error_no_ok():
    conn = ControlConnection()
    cmd = GetLine("PCAP.ACTIVE")
    assert conn.send(cmd) == b"PCAP.ACTIVE?\n"
    assert get_responses(conn, b"NOT OK\n") == [
        (
            cmd,
            ACommandException("GetLine(field='PCAP.ACTIVE') -> NOT OK"),
        )
    ]


def test_get_multiline():
    conn = ControlConnection()
    cmd = GetMultiline("PCAP.*")
    assert conn.send(cmd) == b"PCAP.*?\n"
    assert get_responses(conn, b"!ACTIVE 5 bit_out\n.\n") == [
        (cmd, ["ACTIVE 5 bit_out"])
    ]


def test_get_multiline_error_when_single_line():
    conn = ControlConnection()
    cmd = GetMultiline("PCAP.*")
    assert conn.send(cmd) == b"PCAP.*?\n"

    assert get_responses(conn, b"1\n") == [
        (
            cmd,
            ACommandException("GetMultiline(field='PCAP.*') -> 1"),
        )
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


def test_connect_put_multi_line_bad_list_format():
    """Confirm that an invalid data format raises the expected exception"""
    conn = ControlConnection()
    cmd = Put("SEQ1.TABLE", [1, 2, 3])
    with pytest.raises(TypeError):
        assert conn.send(cmd) == b""


def test_connect_put_no_value():
    """Confirm Put works with no value"""
    conn = ControlConnection()
    cmd = Put("SFP3_SYNC_IN1.SYNC_RESET")
    assert conn.send(cmd) == b"SFP3_SYNC_IN1.SYNC_RESET=\n"
    assert get_responses(conn, b"OK\n") == [(cmd, None)]


def test_connect_append():
    conn = ControlConnection()
    cmd = Append("SEQ1.TABLE", ["1048576", "0", "1000", "1000"])
    assert conn.send(cmd) == b"SEQ1.TABLE<<\n1048576\n0\n1000\n1000\n\n"
    assert get_responses(conn, b"OK\n") == [(cmd, None)]


def test_connect_append_multi_bad_list_format():
    """Confirm that an invalid data format raises the expected exception"""
    conn = ControlConnection()
    cmd = Append("SEQ1.TABLE", [1, 2, 3])
    with pytest.raises(TypeError):
        assert conn.send(cmd) == b""


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
        == b"*DESC.LUT.TYPEA?\n*ENUMS.LUT.TYPEA?\n*DESC.LUT.INPA?\n"
        + b"LUT1.INPA.MAX_DELAY?\n*ENUMS.LUT.INPA?\n"
    )

    # Responses to the 2 *DESC, 2 *ENUM, and MAX_DELAY commands
    responses = [
        b"OK =Source of the value of A for calculation\n",
        b"!Input-Level\n!Pulse-On-Rising-Edge\n.\n",
        b"OK =Input A\n",
        b"OK =10\n",
        b"!TTLIN1.VAL\n!LVDSIN1.VAL\n.\n",
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
        == b"*DESC.PCAP.SAMPLES?\n*ENUMS.PCAP.SAMPLES.CAPTURE?\n"
    )

    # Responses to the *DESC and *ENUM commands
    responses = [
        b"OK =Number of gated samples in the current capture\n",
        b"!No\n!Value\n.\n",
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
                    capture_labels=["No", "Value"],
                )
            },
        )
    ]


def test_get_fields_skip_metadata():
    """Test that the skip_metadata flag causes no description to be retrieved
    for the field"""
    conn = ControlConnection()
    cmd = GetFieldInfo("PCAP", False)
    assert conn.send(cmd) == b"PCAP.*?\n"

    assert conn.receive_bytes(b"!SAMPLES 9 ext_out samples\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            {"SAMPLES": FieldInfo(type="ext_out", subtype="samples", description=None)},
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
                "GetFieldInfo(block='FOO', extended_metadata=True) -> ERR No such block"
            ),
        )
    ]


def test_get_fields_unknown_fields():
    """Test that querying for an unknown field and/or subtype still tries to request a
    description"""
    conn = ControlConnection()
    cmd = GetFieldInfo("PCAP")
    assert conn.send(cmd) == b"PCAP.*?\n"

    # TEST1 has unknown field name, TEST2 has both unknown field and subtype name,
    # TEST3 has known field name but unknown subtype, TEST4 has unknown type but known
    # subtype
    assert (
        conn.receive_bytes(
            b"!TEST1 1 foo\n!TEST2 2 flibble bibble\n!TEST3 3 param unknown\n"
            + b"!TEST4 4 fish uint\n.\n"
        )
        == b"*DESC.PCAP.TEST1?\n*DESC.PCAP.TEST2?\n*DESC.PCAP.TEST3?\n"
        + b"*DESC.PCAP.TEST4?\n"
    )

    responses = [
        b"OK =TEST1 Desc\n",
        b"OK =TEST2 Desc\n",
        b"OK =TEST3 Desc\n",
        b"OK =TEST4 Desc\n",
    ]

    for response in responses:
        assert conn.receive_bytes(response) == b""

    assert get_responses(conn) == [
        (
            cmd,
            {
                "TEST1": FieldInfo(type="foo", subtype=None, description="TEST1 Desc"),
                "TEST2": FieldInfo(
                    type="flibble", subtype="bibble", description="TEST2 Desc"
                ),
                "TEST3": FieldInfo(
                    type="param", subtype="unknown", description="TEST3 Desc"
                ),
                "TEST4": FieldInfo(
                    type="fish", subtype="uint", description="TEST4 Desc"
                ),
            },
        ),
    ]


def idfn(val):
    """helper function to nicely name parameterized test IDs"""
    if isinstance(val, str):
        if val.count("?\n") > 0:
            return ""
        return val
    else:
        return ""


# Table field handled in separate test due to extra round of network calls required
@pytest.mark.parametrize(
    "field_type, field_subtype, requests_responses, expected_field_info",
    [
        (
            "param",
            "uint",
            [
                ("TEST1.TEST_FIELD.MAX?", "OK =10"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            UintFieldInfo("param", "uint", max_val=10, description="Test Description"),
        ),
        (
            "read",
            "uint",
            [
                ("TEST1.TEST_FIELD.MAX?", "OK =67"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            UintFieldInfo("read", "uint", max_val=67, description="Test Description"),
        ),
        (
            "write",
            "uint",
            [
                ("TEST1.TEST_FIELD.MAX?", "OK =58"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            UintFieldInfo("write", "uint", max_val=58, description="Test Description"),
        ),
        (
            "param",
            "int",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("param", "int", description="Test Description"),
        ),
        (
            "read",
            "int",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("read", "int", description="Test Description"),
        ),
        (
            "write",
            "int",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("write", "int", description="Test Description"),
        ),
        (
            "param",
            "scalar",
            [
                ("TEST.TEST_FIELD.UNITS?", "OK =some_units"),
                ("TEST.TEST_FIELD.SCALE?", "OK =0.5"),
                ("TEST.TEST_FIELD.OFFSET?", "OK =8"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            ScalarFieldInfo(
                "param",
                "scalar",
                units="some_units",
                scale=0.5,
                offset=8,
                description="Test Description",
            ),
        ),
        (
            "read",
            "scalar",
            [
                ("TEST.TEST_FIELD.UNITS?", "OK =some_units"),
                ("TEST.TEST_FIELD.SCALE?", "OK =0.5"),
                ("TEST.TEST_FIELD.OFFSET?", "OK =8"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            ScalarFieldInfo(
                "read",
                "scalar",
                units="some_units",
                scale=0.5,
                offset=8,
                description="Test Description",
            ),
        ),
        (
            "write",
            "scalar",
            [
                ("TEST.TEST_FIELD.UNITS?", "OK =some_units"),
                ("TEST.TEST_FIELD.SCALE?", "OK =0.5"),
                ("TEST.TEST_FIELD.OFFSET?", "OK =8"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            ScalarFieldInfo(
                "write",
                "scalar",
                units="some_units",
                scale=0.5,
                offset=8,
                description="Test Description",
            ),
        ),
        (
            "param",
            "bit",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("param", "bit", description="Test Description"),
        ),
        (
            "read",
            "bit",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("read", "bit", description="Test Description"),
        ),
        (
            "read",
            "bit",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("read", "bit", description="Test Description"),
        ),
        (
            "param",
            "action",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("param", "action", description="Test Description"),
        ),
        (
            "read",
            "action",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("read", "action", description="Test Description"),
        ),
        (
            "write",
            "action",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("write", "action", description="Test Description"),
        ),
        (
            "param",
            "lut",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("param", "lut", description="Test Description"),
        ),
        (
            "read",
            "lut",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("read", "lut", description="Test Description"),
        ),
        (
            "write",
            "lut",
            [
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            FieldInfo("write", "lut", description="Test Description"),
        ),
        (
            "param",
            "enum",
            [
                ("*ENUMS.TEST.TEST_FIELD?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            EnumFieldInfo(
                "param", "enum", labels=["VAL1", "VAL2"], description="Test Description"
            ),
        ),
        (
            "read",
            "enum",
            [
                ("*ENUMS.TEST.TEST_FIELD?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            EnumFieldInfo(
                "read", "enum", labels=["VAL1", "VAL2"], description="Test Description"
            ),
        ),
        (
            "write",
            "enum",
            [
                ("*ENUMS.TEST.TEST_FIELD?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            EnumFieldInfo(
                "write", "enum", labels=["VAL1", "VAL2"], description="Test Description"
            ),
        ),
        (
            "param",
            "time",
            [
                ("*ENUMS.TEST.TEST_FIELD.UNITS?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            SubtypeTimeFieldInfo(
                "param",
                "time",
                units_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "read",
            "time",
            [
                ("*ENUMS.TEST.TEST_FIELD.UNITS?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            SubtypeTimeFieldInfo(
                "read",
                "time",
                units_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "write",
            "time",
            [
                ("*ENUMS.TEST.TEST_FIELD.UNITS?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            SubtypeTimeFieldInfo(
                "write",
                "time",
                units_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "time",
            None,
            [
                ("*ENUMS.TEST.TEST_FIELD.UNITS?", "!VAL1\n!VAL2\n."),
                ("TEST1.TEST_FIELD.MIN?", "OK =5e-8"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            TimeFieldInfo(
                "time",
                None,
                units_labels=["VAL1", "VAL2"],
                min_val=5e-8,
                description="Test Description",
            ),
        ),
        (
            "bit_out",
            None,
            [
                ("TEST1.TEST_FIELD.CAPTURE_WORD?", "OK =PCAP.BITS1"),
                ("TEST1.TEST_FIELD.OFFSET?", "OK =12"),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            BitOutFieldInfo(
                "bit_out",
                None,
                capture_word="PCAP.BITS1",
                offset=12,
                description="Test Description",
            ),
        ),
        (
            "pos_out",
            None,
            [
                ("*ENUMS.TEST.TEST_FIELD.CAPTURE?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            PosOutFieldInfo(
                "pos_out",
                None,
                capture_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "ext_out",
            "timestamp",
            [
                ("*ENUMS.TEST.TEST_FIELD.CAPTURE?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            ExtOutFieldInfo(
                "ext_out",
                "timestamp",
                capture_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "ext_out",
            "samples",
            [
                ("*ENUMS.TEST.TEST_FIELD.CAPTURE?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            ExtOutFieldInfo(
                "ext_out",
                "samples",
                capture_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "ext_out",
            "bits",
            [
                ("TEST.TEST_FIELD.BITS?", "!BITS1\n!BITS2\n."),
                ("*ENUMS.TEST.TEST_FIELD.CAPTURE?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            ExtOutBitsFieldInfo(
                "ext_out",
                "bits",
                bits=["BITS1", "BITS2"],
                capture_labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "bit_mux",
            None,
            [
                ("TEST1.TEST_FIELD.MAX_DELAY?", "OK =25"),
                ("*ENUMS.TEST.TEST_FIELD?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            BitMuxFieldInfo(
                "bit_mux",
                None,
                max_delay=25,
                labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
        (
            "pos_mux",
            None,
            [
                ("*ENUMS.TEST.TEST_FIELD?", "!VAL1\n!VAL2\n."),
                ("*DESC.TEST.TEST_FIELD?", "OK =Test Description"),
            ],
            PosMuxFieldInfo(
                "pos_mux",
                None,
                labels=["VAL1", "VAL2"],
                description="Test Description",
            ),
        ),
    ],
    ids=idfn,
)
def test_get_fields_parameterized_type(
    field_type, field_subtype, requests_responses, expected_field_info
):
    """Test every defined field type-subtype pair that has a defined function
    and confirm it creates the expected FieldInfo (or subclass) with the expected
    data"""
    conn = ControlConnection()
    cmd = GetFieldInfo("TEST", extended_metadata=True)
    assert conn.send(cmd) == b"TEST.*?\n"

    if field_subtype is None:
        field_subtype = ""  # Ensure we don't write the literal string "None"

    field_definition_str = f"!TEST_FIELD 1 {field_type} {field_subtype}\n.\n"

    # Split to get individual requests in a list - we don't care about the order
    received_bytes = conn.receive_bytes(field_definition_str.encode())
    received_requests_list = received_bytes.decode().splitlines()

    expected_requests = [x[0] for x in requests_responses]
    responses = [x[1] for x in requests_responses]

    for request in received_requests_list:
        idx = expected_requests.index(request)
        response = responses[idx] + "\n"
        assert conn.receive_bytes(response.encode()) == b""
        expected_requests.pop(idx)
        responses.pop(idx)

    assert (
        not expected_requests
    ), f"Did not receive all expected requests: {expected_requests}"

    assert get_responses(conn) == [
        (
            cmd,
            {"TEST_FIELD": expected_field_info},
        )
    ]


def test_get_fields_type_table():
    """Test for table field type, including descriptions and retrieving enum labels"""
    conn = ControlConnection()
    cmd = GetFieldInfo("SEQ")
    assert conn.send(cmd) == b"SEQ.*?\n"

    assert (
        conn.receive_bytes(b"!TABLE 7 table\n.\n")
        == b"*DESC.SEQ.TABLE?\nSEQ1.TABLE.MAX_LENGTH?\nSEQ1.TABLE.FIELDS?\n"
    )

    assert conn.receive_bytes(b"OK =Sequencer table of lines\n") == b""

    assert conn.receive_bytes(b"OK =16384\n") == b""

    assert (
        conn.receive_bytes(
            b"!15:0 REPEATS uint\n!19:16 TRIGGER enum\n!63:32 POSITION int\n.\n"
        )
        == b"*ENUMS.SEQ1.TABLE[].TRIGGER?\n*DESC.SEQ1.TABLE[].REPEATS?\n"
        b"*DESC.SEQ1.TABLE[].TRIGGER?\n*DESC.SEQ1.TABLE[].POSITION?\n"
    )

    responses = [
        b"!Immediate\n!BITA=0\n.\n",
        b"OK =Number of times the line will repeat\n",
        b"OK =The trigger condition to start the phases\n",
        b"OK =The position that can be used in trigger condition\n",
    ]
    for response in responses:
        assert (
            conn.receive_bytes(response) == b""
        )  # Expect no bytes back as none of these trigger further commands

    assert get_responses(conn) == [
        (
            cmd,
            {
                "TABLE": TableFieldInfo(
                    type="table",
                    subtype=None,
                    description="Sequencer table of lines",
                    max_length=16384,
                    row_words=2,  # Calculated from POSITION field's highest used bit
                    fields={
                        "REPEATS": TableFieldDetails(
                            subtype="uint",
                            bit_low=0,
                            bit_high=15,
                            description="Number of times the line will repeat",
                            labels=None,
                        ),
                        "TRIGGER": TableFieldDetails(
                            subtype="enum",
                            bit_low=16,
                            bit_high=19,
                            description="The trigger condition to start the phases",
                            labels=["Immediate", "BITA=0"],
                        ),
                        "POSITION": TableFieldDetails(
                            subtype="int",
                            bit_low=32,
                            bit_high=63,
                            description=(
                                "The position that can be used in trigger condition"
                            ),
                            labels=None,
                        ),
                    },
                )
            },
        )
    ]


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


def test_get_changes_values():
    """Test that the `values` field returned from `GetChanges` is correctly populated"""
    conn = ControlConnection()
    cmd = GetChanges()

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field 1=Value1\n!Field2=Other Value\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            Changes(
                values={"Field 1": "Value1", "Field2": "Other Value"},
                no_value=[],
                in_error=[],
                multiline_values={},
            ),
        )
    ]


def test_get_changes_values_empty():
    """Test that the `values` field returned from `GetChanges` is correctly populated
    when the value of a field is empty"""
    conn = ControlConnection()
    cmd = GetChanges()

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field 1=\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            Changes(
                values={"Field 1": ""},
                no_value=[],
                in_error=[],
                multiline_values={},
            ),
        )
    ]


def test_get_changes_no_value():
    """Test that the `no_value` field returned from `GetChanges` is correctly
    populated"""
    conn = ControlConnection()
    cmd = GetChanges()

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field 1<\n!Field2<\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            Changes(
                values={},
                no_value=["Field 1", "Field2"],
                in_error=[],
                multiline_values={},
            ),
        )
    ]


def test_get_changes_error():
    """Test that the `error` field returned from `GetChanges` is correctly populated"""
    conn = ControlConnection()
    cmd = GetChanges()

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field1 (error)\n!Field2 (error)\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            Changes(
                values={},
                no_value=[],
                in_error=["Field1", "Field2"],
                multiline_values={},
            ),
        )
    ]


def test_get_changes_multiline():
    """Test that the `multiline_values` field returned from `GetChanges` is correctly
    populated"""
    conn = ControlConnection()
    cmd = GetChanges(ChangeGroup.ALL, True)

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field1<\n.\n") == b"Field1?\n"

    assert get_responses(conn, b"!Val1\n!Val2\n.\n") == [
        (
            cmd,
            Changes(
                values={},
                no_value=[],
                in_error=[],
                multiline_values={"Field1": ["Val1", "Val2"]},
            ),
        )
    ]


def test_get_changes_multiline_no_values():
    """Test that the `multiline_values` field returned from `GetChanges` is correctly
    populated when the table in question has no values"""
    conn = ControlConnection()
    cmd = GetChanges(ChangeGroup.ALL, True)

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field1<\n.\n") == b"Field1?\n"

    assert get_responses(conn, b".\n") == [
        (
            cmd,
            Changes(
                values={},
                no_value=[],
                in_error=[],
                multiline_values={"Field1": []},
            ),
        )
    ]


def test_get_changes_multiline_no_multiline_fields():
    """Test retrieving multiline fields when none are defined returns expected empty
    multiline_values field."""
    conn = ControlConnection()
    cmd = GetChanges(ChangeGroup.ALL, True)

    assert conn.send(cmd) == b"*CHANGES?\n"

    assert conn.receive_bytes(b"!Field1=Value1\n.\n") == b""

    assert get_responses(conn) == [
        (
            cmd,
            Changes(
                values={"Field1": "Value1"},
                no_value=[],
                in_error=[],
                multiline_values={},
            ),
        )
    ]


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
