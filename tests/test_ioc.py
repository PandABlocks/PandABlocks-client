from typing import Dict, List

import numpy
import pytest
from mock import AsyncMock
from mock.mock import MagicMock
from numpy import array, int32, ndarray, uint8, uint16, uint32
from softioc import alarm

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Put
from pandablocks.ioc import (
    IocRecordFactory,
    TableModeEnum,
    TablePacking,
    _BlockAndFieldInfo,
    _ensure_block_number_present,
    _epics_to_panda_name,
    _panda_to_epics_name,
    _RecordInfo,
    _RecordUpdater,
    _TableUpdater,
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


@pytest.fixture
def table_fields() -> Dict[str, TableFieldDetails]:
    """Table field definitions, taken from a SEQ.TABLE instance.
    Associated with table_data and table_field_info fixtures"""
    return {
        "REPEATS": TableFieldDetails(
            subtype="uint",
            bit_low=0,
            bit_high=15,
            description="Number of times the line will repeat ",
            labels=None,
        ),
        "TRIGGER": TableFieldDetails(
            subtype="enum",
            bit_low=16,
            bit_high=19,
            description="The trigger condition to start the phases ",
            labels=[
                "Immediate",
                "BITA=0",
                "BITA=1",
                "BITB=0",
                "BITB=1",
                "BITC=0",
                "BITC=1",
                "POSA>=POSITION",
                "POSA<=POSITION",
                "POSB>=POSITION",
                "POSB<=POSITION",
                "POSC>=POSITION",
                "POSC<=POSITION",
            ],
        ),
        "POSITION": TableFieldDetails(
            subtype="int",
            bit_low=32,
            bit_high=63,
            description="The position that can be used in trigger condition ",
            labels=None,
        ),
        "TIME1": TableFieldDetails(
            subtype="uint",
            bit_low=64,
            bit_high=95,
            description="The time the optional phase 1 should take ",
            labels=None,
        ),
        "OUTA1": TableFieldDetails(
            subtype="uint",
            bit_low=20,
            bit_high=20,
            description="Output A value during phase 1 ",
            labels=None,
        ),
        "OUTB1": TableFieldDetails(
            subtype="uint",
            bit_low=21,
            bit_high=21,
            description="Output B value during phase 1 ",
            labels=None,
        ),
        "OUTC1": TableFieldDetails(
            subtype="uint",
            bit_low=22,
            bit_high=22,
            description="Output C value during phase 1 ",
            labels=None,
        ),
        "OUTD1": TableFieldDetails(
            subtype="uint",
            bit_low=23,
            bit_high=23,
            description="Output D value during phase 1 ",
            labels=None,
        ),
        "OUTE1": TableFieldDetails(
            subtype="uint",
            bit_low=24,
            bit_high=24,
            description="Output E value during phase 1 ",
            labels=None,
        ),
        "OUTF1": TableFieldDetails(
            subtype="uint",
            bit_low=25,
            bit_high=25,
            description="Output F value during phase 1 ",
            labels=None,
        ),
        "TIME2": TableFieldDetails(
            subtype="uint",
            bit_low=96,
            bit_high=127,
            description="The time the mandatory phase 2 should take ",
            labels=None,
        ),
        "OUTA2": TableFieldDetails(
            subtype="uint",
            bit_low=26,
            bit_high=26,
            description="Output A value during phase 2 ",
            labels=None,
        ),
        "OUTB2": TableFieldDetails(
            subtype="uint",
            bit_low=27,
            bit_high=27,
            description="Output B value during phase 2 ",
            labels=None,
        ),
        "OUTC2": TableFieldDetails(
            subtype="uint",
            bit_low=28,
            bit_high=28,
            description="Output C value during phase 2 ",
            labels=None,
        ),
        "OUTD2": TableFieldDetails(
            subtype="uint",
            bit_low=29,
            bit_high=29,
            description="Output D value during phase 2 ",
            labels=None,
        ),
        "OUTE2": TableFieldDetails(
            subtype="uint",
            bit_low=30,
            bit_high=30,
            description="Output E value during phase 2 ",
            labels=None,
        ),
        "OUTF2": TableFieldDetails(
            subtype="uint",
            bit_low=31,
            bit_high=31,
            description="Output F value during phase 2 ",
            labels=None,
        ),
    }


@pytest.fixture
def table_field_info(table_fields) -> TableFieldInfo:
    """Table data associated with table_fields and table_data fixtures"""
    return TableFieldInfo(
        "table", None, "Test table description", 16384, table_fields, 4
    )


@pytest.fixture
def table_data():
    """Table data associated with table_fields and table_field_info fixtures.
    See table_unpacked_data for the unpacked equivalent"""
    return [
        "2457862149",
        "4294967291",
        "100",
        "0",
        "269877248",
        "678",
        "0",
        "55",
        "4293968720",
        "0",
        "9",
        "9999",
    ]


@pytest.fixture
def table_unpacked_data(table_fields) -> Dict[str, ndarray]:
    """The unpacked equivalent of table_data"""
    array_values = [
        array([5, 0, 50000], dtype=uint16),
        array([0, 6, 0], dtype=uint8),
        array([-5, 678, 0], dtype=int32),
        array([100, 0, 9], dtype=uint32),
        array([0, 1, 1], dtype=uint8),
        array([0, 0, 1], dtype=uint8),
        array([0, 0, 1], dtype=uint8),
        array([1, 0, 1], dtype=uint8),
        array([0, 0, 1], dtype=uint8),
        array([1, 0, 1], dtype=uint8),
        array([0, 55, 9999], dtype=uint32),
        array([0, 0, 1], dtype=uint8),
        array([0, 0, 1], dtype=uint8),
        array([1, 1, 1], dtype=uint8),
        array([0, 0, 1], dtype=uint8),
        array([0, 0, 1], dtype=uint8),
        array([1, 0, 1], dtype=uint8),
    ]
    data = {}
    for field_name, data_array in zip(table_fields.keys(), array_values):
        data[field_name] = data_array

    return data


@pytest.fixture
def table_unpacked_data_records(
    table_fields, table_unpacked_data
) -> Dict[str, _RecordInfo]:
    """A faked list of records containing the table_unpacked_data"""

    data = {}
    for field_name, data_array in zip(table_fields, table_unpacked_data.values()):
        mocked_record = MagicMock()
        mocked_record.get = MagicMock(return_value=data_array)
        info = _RecordInfo(mocked_record, lambda x: None)
        data[field_name] = info
    return data


@pytest.fixture
def table_updater(
    table_field_info: TableFieldInfo,
    table_fields: Dict[str, TableFieldDetails],
    table_unpacked_data_records: Dict[str, _RecordInfo],
    table_data: List[str],
):
    """Provides a _TableUpdater with configured records and mocked functionality"""
    client = AsyncioClient("123")
    client.send = AsyncMock()  # type: ignore
    # mypy doesn't play well with mocking so suppress error

    mocked_mode_record = MagicMock()
    # Default mode record to view, as per default construction
    mocked_mode_record.get = MagicMock(return_value=TableModeEnum.VIEW.value)
    record_info = _RecordInfo(mocked_mode_record, lambda x: None)

    updater = _TableUpdater(
        client,
        "SEQ1.TABLE",
        table_field_info,
        table_fields,
        table_unpacked_data_records,
        table_data,
    )

    updater.set_mode_record_info(record_info)

    return updater


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
        data = await introspect_panda(client)
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


@pytest.mark.asyncio
async def test_record_updater():
    """Test that the record updater succesfully Put's data to the client"""
    client = AsyncioClient("123")
    client.send = AsyncMock()
    updater = _RecordUpdater("ABC:DEF", client, float)

    await updater.update("1.0")

    client.send.assert_called_once_with(Put("ABC.DEF", "1.0"))


@pytest.mark.asyncio
async def test_record_updater_labels():
    """Test that the record updater succesfully Put's data to the client
    when the data is a label index"""
    client = AsyncioClient("123")
    client.send = AsyncMock()
    updater = _RecordUpdater("ABC:DEF", client, float, ["Label1", "Label2", "Label3"])

    await updater.update("2")

    client.send.assert_called_once_with(Put("ABC.DEF", "Label3"))


def test_table_packing_unpack(
    table_field_info: TableFieldInfo,
    table_fields: Dict[str, TableFieldDetails],
    table_data: List[str],
    table_unpacked_data,
):
    """Test table unpacking works as expected"""
    assert table_field_info.row_words
    unpacked = TablePacking.unpack(table_field_info.row_words, table_fields, table_data)

    for actual, expected in zip(unpacked, table_unpacked_data.values()):
        assert numpy.array_equal(actual, expected)


def test_table_packing_pack(
    table_field_info: TableFieldInfo,
    table_unpacked_data_records: Dict[str, _RecordInfo],
    table_fields: Dict[str, TableFieldDetails],
    table_data: List[str],
):
    """Test table unpacking works as expected"""
    assert table_field_info.row_words
    unpacked = TablePacking.pack(
        table_field_info.row_words, table_unpacked_data_records, table_fields
    )

    for actual, expected in zip(unpacked, table_data):
        assert actual == expected


def test_table_packing_pack_length_mismatched(
    table_field_info: TableFieldInfo,
    table_unpacked_data_records: Dict[str, _RecordInfo],
    table_fields: Dict[str, TableFieldDetails],
):
    """Test that mismatching lengths on waveform inputs causes an exception"""
    assert table_field_info.row_words

    # Adjust one of the record lengths so it mismatches
    table_unpacked_data_records["OUTC1"].record.get = MagicMock(
        return_value=array([1, 2, 3, 4, 5, 6, 7, 8])
    )

    with pytest.raises(AssertionError):
        TablePacking.pack(
            table_field_info.row_words, table_unpacked_data_records, table_fields
        )


def test_table_packing_roundtrip(
    table_field_info: TableFieldInfo,
    table_fields: Dict[str, TableFieldDetails],
    table_data: List[str],
):
    """Test that calling unpack -> pack yields the same data"""
    assert table_field_info.row_words
    unpacked = TablePacking.unpack(table_field_info.row_words, table_fields, table_data)

    # Put these values into Mocks for the Records
    data: Dict[str, _RecordInfo] = {}
    for field_name, data_array in zip(table_fields, unpacked):
        mocked_record = MagicMock()
        mocked_record.get = MagicMock(return_value=data_array)
        info = _RecordInfo(mocked_record, lambda x: None)
        data[field_name] = info

    packed = TablePacking.pack(table_field_info.row_words, data, table_fields)

    assert packed == table_data


def test_table_updater_fields_sorted(table_updater: _TableUpdater):
    """Test that the field sorting done in post_init has occurred"""

    # Bits start at 0
    curr_bit = -1
    for field in table_updater.table_fields.values():
        assert curr_bit < field.bit_low, "Fields are not in bit order"
        assert (
            field.bit_low <= field.bit_high  # fields may be 1 bit wide
        ), "Field had incorrect bit_low and bit_high order"
        assert curr_bit < field.bit_high, "Field had bit_high lower than bit_low"
        curr_bit = field.bit_high


def test_table_updater_validate_mode_view(table_updater: _TableUpdater):
    """Test the validate method when mode is View"""

    # View is default in table_updater
    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is False


def test_table_updater_validate_mode_edit(table_updater: _TableUpdater):
    """Test the validate method when mode is Edit"""

    table_updater._mode_record_info.record.get = MagicMock(
        return_value=TableModeEnum.EDIT.value
    )

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is True


def test_table_updater_validate_mode_submit(table_updater: _TableUpdater):
    """Test the validate method when mode is Submit"""

    table_updater._mode_record_info.record.get = MagicMock(
        return_value=TableModeEnum.SUBMIT.value
    )

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is False


def test_table_updater_validate_mode_discard(table_updater: _TableUpdater):
    """Test the validate method when mode is Discard"""

    table_updater._mode_record_info.record.get = MagicMock(
        return_value=TableModeEnum.DISCARD.value
    )

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is False


def test_table_updater_validate_mode_unknown(table_updater: _TableUpdater, capsys):
    """Test the validate method when mode is unknown"""

    table_updater._mode_record_info.record.get = MagicMock(return_value="UnknownValue")
    table_updater._mode_record_info.record.set_alarm = MagicMock()

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")

    assert table_updater.validate_waveform(record, "value is irrelevant") is False
    table_updater._mode_record_info.record.set_alarm.assert_called_once_with(
        alarm.INVALID_ALARM, alarm.UDF_ALARM
    )


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
