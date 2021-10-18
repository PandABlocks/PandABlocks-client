import pytest
from mock import AsyncMock

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Put
from pandablocks.ioc._types import EpicsName
from pandablocks.ioc.ioc import (
    IocRecordFactory,
    _BlockAndFieldInfo,
    _ensure_block_number_present,
    _RecordUpdater,
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
    TableFieldDetails,
    TableFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)
from tests.conftest import DummyServer

TEST_PREFIX = "TEST-PREFIX"
counter = 0


# TODO: Split tests up - tables and HDF5 are in their own files now
@pytest.fixture
def ioc_record_factory():
    """Create a new IocRecordFactory instance with a new, unique, namespace.
    This means each test can run in the same process, as each test will get
    its own namespace.
    """
    global counter
    counter += 1
    return IocRecordFactory(TEST_PREFIX + str(counter), AsyncioClient("123"), {})


@pytest.fixture
def dummy_server_introspect_panda(dummy_server_in_thread: DummyServer):
    """A dummy server that responds to all the requests introspect_panda makes
    during its operation.
    Note that the order of responses was determined by trial and error."""
    get_changes_scalar_data = (
        # Note the deliberate concatenation across lines - this must be a single
        # entry in the list
        "!PCAP.FOO=1\n!PCAP.BAR=12.34\n!*METADATA.LABEL_PCAP1=PcapMetadataLabel\n"
        "!SEQ1.TABLE<\n"
        "."
    )
    get_changes_multiline_data = "!1\n!2\n!3\n."

    dummy_server_in_thread.send += [
        "!PCAP 1\n!SEQ 3\n.",
        "OK =PCAP Desc",
        "OK =SEQ Desc",
        "!TRIG_EDGE 3 param enum\n!GATE 1 bit_mux\n.",  # PCAP fields
        "!TABLE 7 table\n.",  # SEQ field
        get_changes_scalar_data,
        "!Label1\n!Label2\n.",  # TRIG_EDGE enum labels
        "OK =100",  # GATE MAX_DELAY
        "!LabelA\n!LabelB\n.",  # GATE labels
        "OK =Trig Edge Desc",
        "OK =Gate Desc",
        "OK =16384",  # TABLE MAX_LENGTH
        "!15:0 REPEATS uint\n!19:16 TRIGGER enum\n!63:32 POSITION int\n.",  # TABLE flds
        "OK =Sequencer table of lines",  # TABLE Desc
        get_changes_multiline_data,
        "!Immediate\n!BITA=0\n.",  # TRIGGER field labels
        "OK =Number of times the line will repeat",  # Repeats field desc
        "OK =The trigger condition to start the phases",  # TRIGGER field desc
        "OK =The position that can be used in trigger condition",  # POSITION field desc
    ]
    # If you need to change the above responses,
    # it'll probably help to enable debugging on the server
    # dummy_server_in_thread.debug = True
    yield dummy_server_in_thread


TEST_RECORD = EpicsName("TEST:RECORD")


def test_ensure_block_number_present():
    assert _ensure_block_number_present("ABC.DEF.GHI") == "ABC1.DEF.GHI"
    assert _ensure_block_number_present("JKL1.MNOP") == "JKL1.MNOP"


@pytest.mark.asyncio
async def test_introspect_panda(dummy_server_introspect_panda):
    """High-level test that introspect_panda returns expected data structures"""
    async with AsyncioClient("localhost") as client:
        (data, all_values_dict) = await introspect_panda(client)
        assert data["PCAP"] == _BlockAndFieldInfo(
            block_info=BlockInfo(number=1, description="PCAP Desc"),
            fields={
                "TRIG_EDGE": EnumFieldInfo(
                    type="param",
                    subtype="enum",
                    description="Trig Edge Desc",
                    labels=["Label1", "Label2"],
                ),
                "GATE": BitMuxFieldInfo(
                    type="bit_mux",
                    subtype=None,
                    description="Gate Desc",
                    max_delay=100,
                    labels=["LabelA", "LabelB"],
                ),
            },
            values={
                "PCAP1:FOO": "1",
                "PCAP1:BAR": "12.34",
                "PCAP1:LABEL": "PcapMetadataLabel",
            },
        )

        assert data["SEQ"] == _BlockAndFieldInfo(
            block_info=BlockInfo(number=3, description="SEQ Desc"),
            fields={
                "TABLE": TableFieldInfo(
                    type="table",
                    subtype=None,
                    description="Sequencer table of lines",
                    max_length=16384,
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
                    row_words=2,
                )
            },
            values={"SEQ1:TABLE": ["1", "2", "3"]},
        )

        assert all_values_dict == {
            "PCAP1:BAR": "12.34",
            "PCAP1:FOO": "1",
            "PCAP1:LABEL": "PcapMetadataLabel",
            "SEQ1:TABLE": ["1", "2", "3"],
        }


@pytest.mark.asyncio
async def test_record_updater():
    """Test that the record updater succesfully Put's data to the client"""
    client = AsyncioClient("123")
    client.send = AsyncMock()
    updater = _RecordUpdater("ABC:DEF", client, float, {})

    await updater.update("1.0")

    client.send.assert_called_once_with(Put("ABC.DEF", "1.0"))


@pytest.mark.asyncio
async def test_record_updater_labels():
    """Test that the record updater succesfully Put's data to the client
    when the data is a label index"""
    client = AsyncioClient("123")
    client.send = AsyncMock()
    updater = _RecordUpdater(
        "ABC:DEF", client, float, None, labels=["Label1", "Label2", "Label3"]
    )

    await updater.update("2")

    client.send.assert_called_once_with(Put("ABC.DEF", "Label3"))


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
                f"{TEST_RECORD}:CAPTURE",
            ],
        ),
        (
            ExtOutFieldInfo("ext_out", "samples", capture_labels=["No", "Diff"]),
            {
                f"{TEST_RECORD}:CAPTURE": "Diff",
            },
            [
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
