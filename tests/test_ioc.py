import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.ioc import (
    IocRecordFactory,
    _BlockAndFieldInfo,
    _ensure_block_number_present,
    _epics_to_panda_name,
    _panda_to_epics_name,
    introspect_panda,
)
from pandablocks.responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    BlockInfo,
    EnumFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)

TEST_PREFIX = "TEST-PREFIX"
counter = 0


@pytest.fixture
def ioc_record_factory():
    """Create a new IocRecordFactory instance with a new, unique, namespace.
    This means each test can run in the same process, as each test will get
    its own namespace.
    """
    global counter
    counter += 1
    return IocRecordFactory(TEST_PREFIX + str(counter), AsyncioClient("123"))


@pytest.fixture
def dummy_server_introspect_panda(dummy_server_in_thread):
    dummy_server_in_thread.send += [
        # "!PCAP 1\n!LUT 8\n!SRGATE 2\n.",
        # "OK =PCAP Desc\nOK =LUT Desc\nOK =SRGATE Desc\n",
        # "!INPB 1 bit_mux\n!TYPEA 5 param enum\n.",  # LUT fields
        # "!TRIG_EDGE 3 param enum\n!GATE 1 bit_mux\n.",  # PCAP fields
        # "!OUT 1 bit_out\n.",  # SRGATE fields
        # "!FOO=1\nBAR=12.34\n.",  # GetChanges
        "!PCAP 1\n.",
        "OK =PCAP Desc",
        # "!TRIG_EDGE 3 param enum\n!GATE 1 bit_mux\n.",  # PCAP fields
        "!TRIG_EDGE 3 param enum\n.",  # PCAP fields temp
        "!PCAP.FOO=1\n!PCAP.BAR=12.34\n.",  # GetChanges
        "!Label1\n!Label2\n.",  # TRIG_EDGE enum labels
        "OK =Trig Edge Desc",
        # "Ok =Gate Desc",
        # "OK =100",  # GATE MAX_DELAY
        # "!LabelA\n!LabelB\n.",  # GATE labels
    ]
    yield dummy_server_in_thread


TEST_RECORD = "TEST:RECORD"


def test_panda_to_epics_name_conversion():
    assert _panda_to_epics_name("ABC.123.456") == "ABC:123:456"


def test_epics_to_panda_name_conversion():
    assert _epics_to_panda_name("ABC:123:456") == "ABC.123.456"


def test_panda_to_epics_and_back_name_conversion():
    """Test panda->EPICS->panda round trip name conversion works"""
    assert _epics_to_panda_name(_panda_to_epics_name("ABC.123.456")) == "ABC.123.456"


def test_ensure_block_number_present():
    assert _ensure_block_number_present("ABC.DEF.GHI") == "ABC1.DEF.GHI"
    assert _ensure_block_number_present("JKL1.MNOP") == "JKL1.MNOP"


@pytest.mark.asyncio
async def test_introspect_panda(dummy_server_introspect_panda):
    """High-level test that introspect_panda returns expected data structures"""
    async with AsyncioClient("localhost") as client:
        foo = await introspect_panda(client)
        assert foo == {
            "PCAP": _BlockAndFieldInfo(
                block_info=BlockInfo(number=1, description="PCAP Desc"),
                fields={
                    "TRIG_EDGE": EnumFieldInfo(
                        type="param",
                        subtype="enum",
                        description="Trig Edge Desc",
                        labels=["Label1", "Label2"],
                    )
                },
                values={"PCAP1:FOO": "1", "PCAP1:BAR": "12.34"},
            )
        }


# TODO: Test the special types
# Tests for every known type-subtype pair except the following, which have their own
# separate tests:
# ext_out - bits
# table
# param - action
# read - action
@pytest.mark.parametrize(
    "field_info, values, expected_records",
    [
        (
            TimeFieldInfo(
                "time",
                units_labels=["s", "ms", "min"],
                min=8e-09,
            ),
            {
                f"{TEST_RECORD}": "0.1",
                f"{TEST_RECORD}:UNITS": "s",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:UNITS", f"{TEST_RECORD}:MIN"],
        ),
        (
            SubtypeTimeFieldInfo(
                "param",
                "time",
                units_labels=["s", "ms", "min"],
            ),
            {
                f"{TEST_RECORD}": "1",
                f"{TEST_RECORD}:UNITS": "s",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:UNITS"],
        ),
        (
            SubtypeTimeFieldInfo(
                "read",
                "time",
                units_labels=["s", "ms", "min"],
            ),
            {
                f"{TEST_RECORD}": "1",
                f"{TEST_RECORD}:UNITS": "s",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:UNITS"],
        ),
        (
            SubtypeTimeFieldInfo(
                "write",
                "time",
                units_labels=["s", "ms", "min"],
            ),
            {
                f"{TEST_RECORD}:UNITS": "s",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:UNITS"],
        ),
        (
            BitOutFieldInfo(
                "bit_out",
                capture_word="ABC.DEF",
                offset=10,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:CAPTURE_WORD", f"{TEST_RECORD}:OFFSET"],
        ),
        (
            PosOutFieldInfo("pos_out", capture_labels=["No", "Diff"]),
            {
                f"{TEST_RECORD}": "0",
                f"{TEST_RECORD}:CAPTURE": "Diff",
                f"{TEST_RECORD}:OFFSET": "5",
                f"{TEST_RECORD}:SCALE": "0.5",
                f"{TEST_RECORD}:UNITS": "MyUnits",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:CAPTURE",
                f"{TEST_RECORD}:OFFSET",
                f"{TEST_RECORD}:SCALE",
                f"{TEST_RECORD}:UNITS",
            ],
        ),
        (
            ExtOutFieldInfo("ext_out", "timestamp", capture_labels=["No", "Diff"]),
            {
                f"{TEST_RECORD}:CAPTURE": "Diff",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:CAPTURE",
            ],
        ),
        (
            ExtOutFieldInfo("ext_out", "samples", capture_labels=["No", "Diff"]),
            {
                f"{TEST_RECORD}:CAPTURE": "Diff",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:CAPTURE",
            ],
        ),
        (
            BitMuxFieldInfo(
                "bit_mux",
                max_delay=5,
                labels=["TTLIN1.VAL", "TTLIN2.VAL", "TTLIN3.VAL"],
            ),
            {
                f"{TEST_RECORD}": "SRGATE4.OUT",
                f"{TEST_RECORD}:DELAY": "0",
                f"{TEST_RECORD}:MAX_DELAY": "31",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:DELAY",
                f"{TEST_RECORD}:MAX_DELAY",
            ],
        ),
        (
            PosMuxFieldInfo(
                "pos_mux",
                labels=["INENC1.VAL", "INENC2.VAL", "INENC3.VAL"],
            ),
            {
                f"{TEST_RECORD}": "ZERO",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            UintFieldInfo(
                "param",
                "uint",
                max=63,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:MAX",
            ],
        ),
        (
            UintFieldInfo(
                "read",
                "uint",
                max=63,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:MAX",
            ],
        ),
        (
            UintFieldInfo(
                "write",
                "uint",
                max=63,
            ),
            {},
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:MAX",
            ],
        ),
        (
            FieldInfo(
                "param",
                "int",
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "read",
                "int",
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "write",
                "int",
            ),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            ScalarFieldInfo("param", "scalar", offset=0, scale=0.001, units="deg"),
            {
                f"{TEST_RECORD}": "48.48",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:OFFSET",
                f"{TEST_RECORD}:SCALE",
                f"{TEST_RECORD}:UNITS",
            ],
        ),
        (
            ScalarFieldInfo("read", "scalar", offset=0, scale=0.001, units="deg"),
            {
                f"{TEST_RECORD}": "48.48",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:OFFSET",
                f"{TEST_RECORD}:SCALE",
                f"{TEST_RECORD}:UNITS",
            ],
        ),
        (
            ScalarFieldInfo("write", "scalar", offset=0, scale=0.001, units="deg"),
            {},
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:OFFSET",
                f"{TEST_RECORD}:SCALE",
                f"{TEST_RECORD}:UNITS",
            ],
        ),
        (
            FieldInfo(
                "param",
                "bit",
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "read",
                "bit",
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "write",
                "bit",
            ),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "write",
                "action",
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "param",
                "lut",
            ),
            {
                f"{TEST_RECORD}": "0x00000000",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "read",
                "lut",
            ),
            {
                f"{TEST_RECORD}": "0x00000000",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "write",
                "lut",
            ),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            EnumFieldInfo("param", "enum", labels=["Value", "-Value"]),
            {
                f"{TEST_RECORD}": "-Value",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            EnumFieldInfo("read", "enum", labels=["Value", "-Value"]),
            {
                f"{TEST_RECORD}": "-Value",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            EnumFieldInfo("write", "enum", labels=["Value", "-Value"]),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
    ],
)
def test_create_record(
    ioc_record_factory: IocRecordFactory, field_info, values, expected_records
):
    """Test that the expected records are returned for each field info and values
    inputs"""
    returned_records = ioc_record_factory.create_record(TEST_RECORD, field_info, values)
    assert len(returned_records) == len(expected_records)
    assert all(key in returned_records for key in expected_records)
