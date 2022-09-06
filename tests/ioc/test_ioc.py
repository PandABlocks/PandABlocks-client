import asyncio
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from conftest import TEST_PREFIX
from mock import AsyncMock, patch
from mock.mock import MagicMock, call
from softioc import builder, fields

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetLine, Put
from pandablocks.ioc._types import (
    ONAM_STR,
    ZNAM_STR,
    EpicsName,
    InErrorException,
    RecordInfo,
    ScalarRecordValue,
)
from pandablocks.ioc.ioc import (
    IocRecordFactory,
    StringRecordLabelValidator,
    _RecordUpdater,
    _TimeRecordUpdater,
    update,
)
from pandablocks.responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    Changes,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)


@pytest.fixture
def record_updater() -> _RecordUpdater:
    """Create a near-empty _RecordUpdater with a mocked client"""
    client = AsyncioClient("123")
    client.send = AsyncMock()  # type: ignore
    record_info = RecordInfo(float)
    mocked_record = MagicMock()
    mocked_record.name = "PREFIX:ABC:DEF"
    record_info.add_record(mocked_record)

    return _RecordUpdater(record_info, client, {}, None)


@pytest.fixture
def ioc_record_factory(clear_records: None):
    """Create a new IocRecordFactory instance with a new, unique, namespace.
    This means each test can run in the same process, as each test will get
    its own namespace.
    """
    return IocRecordFactory(AsyncioClient("123"), TEST_PREFIX, {})


TEST_RECORD = EpicsName("TEST:RECORD")


@pytest.mark.asyncio
async def test_record_updater(record_updater: _RecordUpdater):
    """Test that the record updater succesfully Put's data to the client"""

    await record_updater.update("1.0")
    mock: AsyncMock = record_updater.client.send  # type: ignore
    mock.assert_called_once_with(Put("ABC.DEF", "1.0"))


@pytest.mark.asyncio
async def test_record_updater_labels(record_updater: _RecordUpdater):
    """Test that the record updater succesfully Put's data to the client
    when the data is a label index"""

    record_updater.labels = ["Label1", "Label2", "Label3"]

    await record_updater.update("2")
    mock: AsyncMock = record_updater.client.send  # type: ignore
    mock.assert_called_once_with(Put("ABC.DEF", "Label3"))


@pytest.mark.asyncio
async def test_record_updater_value_none(record_updater: _RecordUpdater):
    """Test that the record updater succesfully Put's data to the client
    when the data is 'None' e.g. for action-write fields"""

    await record_updater.update(None)
    mock: AsyncMock = record_updater.client.send  # type: ignore
    mock.assert_called_once_with(Put("ABC.DEF", None))


@pytest.mark.asyncio
async def test_record_updater_restore_previous_value(record_updater: _RecordUpdater):
    """Test that the record updater rolls back records to previous value on
    Put failure"""

    record_updater.all_values_dict = {EpicsName("ABC:DEF"): "999"}

    mocked_send: AsyncMock = record_updater.client.send  # type: ignore
    mocked_send.side_effect = Exception("Injected exception")

    await record_updater.update("1.0")

    record_updater.record_info.record.set.assert_called_once_with("999", process=False)


def idfn(val):
    """helper function to nicely name parameterized test IDs"""
    if isinstance(val, FieldInfo):
        return val.type + "-" + str(val.subtype)  # subtype may be None
    elif isinstance(val, (dict, list)):
        return ""


# Tests for every known type-subtype pair except the following, which have their own
# separate tests:
# ext_out - bits
# table (separate file)
# param - action
# read - action
@pytest.mark.parametrize(
    "field_info, values, expected_records",
    [
        (
            TimeFieldInfo(
                "time",
                None,
                None,
                units_labels=["s", "ms", "min"],
                min_val=8e-09,
            ),
            {
                f"{TEST_RECORD}": "0.1",
                f"{TEST_RECORD}:UNITS": "s",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:UNITS"],
        ),
        (
            SubtypeTimeFieldInfo(
                "param",
                "time",
                None,
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
                None,
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
                None,
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
                None,
                None,
                capture_word="ABC.DEF",
                offset=10,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:CAPTURE_WORD", f"{TEST_RECORD}:OFFSET"],
        ),
        (
            PosOutFieldInfo("pos_out", None, None, capture_labels=["No", "Diff"]),
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
            ExtOutFieldInfo(
                "ext_out", "timestamp", None, capture_labels=["No", "Diff"]
            ),
            {
                f"{TEST_RECORD}:CAPTURE": "Diff",
            },
            [
                f"{TEST_RECORD}:CAPTURE",
            ],
        ),
        (
            ExtOutFieldInfo("ext_out", "samples", None, capture_labels=["No", "Diff"]),
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
                None,
                None,
                max_delay=5,
                labels=["TTLIN1.VAL", "TTLIN2.VAL", "TTLIN3.VAL"],
            ),
            {
                f"{TEST_RECORD}": "TTLIN1.VAL",
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
                None,
                None,
                labels=["INENC1.VAL", "INENC2.VAL", "INENC3.VAL"],
            ),
            {
                f"{TEST_RECORD}": "INENC2.VAL",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            UintFieldInfo(
                "param",
                "uint",
                None,
                max_val=63,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            UintFieldInfo(
                "read",
                "uint",
                None,
                max_val=63,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            UintFieldInfo(
                "write",
                "uint",
                None,
                max_val=63,
            ),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            FieldInfo(
                "param",
                "int",
                None,
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
                None,
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
                None,
            ),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            ScalarFieldInfo(
                "param", "scalar", None, offset=0, scale=0.001, units="deg"
            ),
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
            ScalarFieldInfo("read", "scalar", None, offset=0, scale=0.001, units="deg"),
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
            ScalarFieldInfo(
                "write", "scalar", None, offset=0, scale=0.001, units="deg"
            ),
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
            ),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            EnumFieldInfo("param", "enum", None, labels=["Value", "-Value"]),
            {
                f"{TEST_RECORD}": "-Value",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            EnumFieldInfo("read", "enum", None, labels=["Value", "-Value"]),
            {
                f"{TEST_RECORD}": "-Value",
            },
            [
                f"{TEST_RECORD}",
            ],
        ),
        (
            EnumFieldInfo("write", "enum", None, labels=["Value", "-Value"]),
            {},
            [
                f"{TEST_RECORD}",
            ],
        ),
    ],
    ids=idfn,
)
def test_create_record(
    ioc_record_factory: IocRecordFactory, field_info, values, expected_records
):
    """Test that the expected records are returned for each field info and values
    inputs"""
    returned_records = ioc_record_factory.create_record(TEST_RECORD, field_info, values)
    assert len(returned_records) == len(expected_records)
    assert all(key in returned_records for key in expected_records)


@patch("pandablocks.ioc.ioc.IocRecordFactory._make_ext_out")
@patch("pandablocks.ioc.ioc.builder.records")
def test_make_ext_out_bits(
    mocked_builder_records: MagicMock,
    mocked_ext_out: MagicMock,
    ioc_record_factory: IocRecordFactory,
):
    """Test _make_ext_out_bits creates all the records expected"""

    record_name = EpicsName("PCAP:BITS0")
    bits = [
        "TTLIN1.VAL",
        "TTLIN2.VAL",
        "TTLIN3.VAL",
        "TTLIN4.VAL",
        "TTLIN5.VAL",
        "TTLIN6.VAL",
        "LVDSIN1.VAL",
        "LVDSIN2.VAL",
        "INENC1.A",
        "INENC2.A",
        "INENC3.A",
        "INENC4.A",
        "INENC1.B",
        "INENC2.B",
        "INENC3.B",
        "INENC4.B",
        "INENC1.Z",
        "INENC2.Z",
        "INENC3.Z",
        "INENC4.Z",
        "INENC1.DATA",
        "INENC2.DATA",
        "INENC3.DATA",
        "INENC4.DATA",
        "INENC1.CONN",
        "INENC2.CONN",
        "INENC3.CONN",
        "INENC4.CONN",
        "OUTENC1.CLK",
        "OUTENC2.CLK",
        "OUTENC3.CLK",
        "OUTENC4.CLK",
    ]
    field_info = ExtOutBitsFieldInfo(
        "ext_out", "bits", "Test Description", ["No", "Value"], bits
    )
    values: Dict[EpicsName, ScalarRecordValue] = {
        EpicsName(f"{record_name}:CAPTURE"): "No",
    }

    # Mock the return from _make_ext_out so we can examine what happens
    mocked_capture_record_info = MagicMock()
    mocked_ext_out.return_value = {record_name + ":CAPTURE": mocked_capture_record_info}

    ioc_record_factory._make_ext_out_bits(
        record_name,
        field_info,
        values,
    )

    # Confirm correct aliases added to Capture record
    calls = [
        call(ioc_record_factory._record_prefix + ":BITS:" + str(i) + ":CAPTURE")
        for i in range(0, 32)
    ]

    mocked_capture_record: MagicMock = mocked_capture_record_info.record
    mocked_capture_record.add_alias.assert_has_calls(calls)

    # Confirm correct bi and stringin records created
    # This isn't a great test, but it's very complex to set up all the
    # necessary linked records as a system test, so this'll do.
    for i, label in enumerate(bits):
        link = ioc_record_factory._record_prefix + ":" + label.replace(".", ":") + " CP"
        enumerated_bits_prefix = f"BITS:{i}"
        mocked_builder_records.bi.assert_any_call(
            enumerated_bits_prefix + ":VAL",
            INP=link,
            DESC="Value of field connected to this BIT",
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
        )

        mocked_builder_records.stringin.assert_any_call(
            enumerated_bits_prefix + ":NAME",
            VAL=label,
            DESC="Name of field connected to this BIT",
        )


@pytest.mark.parametrize("type", ["param", "read"])
def test_create_record_action(ioc_record_factory: IocRecordFactory, type: str):
    """Test the param-action and read-action types do not create records"""
    assert (
        ioc_record_factory.create_record(TEST_RECORD, FieldInfo(type, "action", ""), {})
        == {}
    )


def test_create_record_info_value_error(
    ioc_record_factory: IocRecordFactory, tmp_path: Path
):
    """Test _create_record_info when value is an _InErrorException.
    This test succeeds if no exceptions are thrown."""

    ioc_record_factory._create_record_info(
        EpicsName("SomeOutRec"),
        None,
        builder.aOut,
        float,
        initial_value=InErrorException("Mocked exception"),
    )

    ioc_record_factory._create_record_info(
        EpicsName("SomeInRec"),
        None,
        builder.aIn,
        float,
        initial_value=InErrorException("Mocked exception"),
    )

    # TODO: Is this a stupid way to check the SEVR and STAT attributes?
    record_file = tmp_path / "records.db"
    builder.WriteRecords(record_file)

    file_contents = record_file.read_text()

    num_sevr = file_contents.count("SEVR")
    num_stat = file_contents.count("STAT")

    assert (
        num_sevr == 2
    ), f"SEVR not found twice in record file contents: {file_contents}"
    assert (
        num_stat == 2
    ), f"STAT not found twice in record file contents: {file_contents}"


@pytest.mark.asyncio
@patch("pandablocks.ioc.ioc.db_put_field")
@pytest.mark.parametrize("new_val", ["TEST2", 2])
async def test_time_record_updater_update_egu(
    db_put_field: MagicMock, mocked_time_record_updater: _TimeRecordUpdater, new_val
):
    """Test that _TimeRecordUpdater.update_egu works correctly with any valid input"""

    mocked_time_record_updater.update_egu(new_val)

    db_put_field.assert_called_once()

    # Check the expected arguments are passed to db_put_field.
    # Note we don't check the value of `array.ctypes.data` parameter as it's a pointer
    # to a memory address so will always vary
    put_field_args = db_put_field.call_args.args
    expected_args = [TEST_PREFIX + ":BASE:RECORD.EGU", fields.DBF_STRING, 1]
    for arg in expected_args:
        assert arg in put_field_args
    assert type(put_field_args[2]) == int


@pytest.mark.asyncio
@patch("pandablocks.ioc.ioc.db_put_field")
async def test_time_record_updater_update_drvl(
    db_put_field: MagicMock, mocked_time_record_updater: _TimeRecordUpdater
):
    """Test that _TimeRecordUpdater.update_drvl works correctly"""

    await mocked_time_record_updater.update_drvl()

    # ...Just to make mypy happy...
    assert isinstance(mocked_time_record_updater.client, MagicMock)
    mocked_time_record_updater.client.send.assert_called_once_with(GetLine("TEST.MIN"))

    db_put_field.assert_called_once()

    # Check the expected arguments are passed to db_put_field.
    # Note we don't check the value of `array.ctypes.data` parameter as it's a pointer
    # to a memory address so will always vary
    put_field_args = db_put_field.call_args.args
    expected_args = [TEST_PREFIX + ":BASE:RECORD.DRVL", fields.DBF_DOUBLE, 1]
    for arg in expected_args:
        assert arg in put_field_args
    assert type(put_field_args[2]) == int


def test_uint_sets_record_attributes(ioc_record_factory: IocRecordFactory):
    """Test that creating a uint record correctly sets all the attributes"""

    name = EpicsName("TEST1")
    max_val = 500
    uint_field_info = UintFieldInfo("param", "uint", None, max_val)
    record_dict = ioc_record_factory._make_uint(name, uint_field_info, builder.longOut)
    longout_rec = record_dict[name].record
    assert longout_rec.DRVL.Value() == 0
    assert longout_rec.DRVH.Value() == max_val
    assert longout_rec.HOPR.Value() == max_val

    name = EpicsName("TEST2")
    record_dict = ioc_record_factory._make_uint(name, uint_field_info, builder.longIn)
    longin_rec = record_dict[name].record
    assert longin_rec.HOPR.Value() == max_val


def test_uint_truncates_max_value(ioc_record_factory: IocRecordFactory, caplog):
    """Test that we correctly truncate a too large maximum value and emit a warning"""
    name = EpicsName("TEST1")
    max_val = 99999999999999999999
    uint_field_info = UintFieldInfo("param", "uint", None, max_val)

    with caplog.at_level(logging.WARNING):
        record_dict = ioc_record_factory._make_uint(
            name, uint_field_info, builder.longOut
        )

    longout_rec = record_dict[name].record
    assert longout_rec.DRVH.Value() == np.iinfo(np.int32).max
    assert longout_rec.HOPR.Value() == np.iinfo(np.int32).max

    assert len(caplog.messages) == 1
    assert "Restricting to int32 maximum value." in caplog.text


def test_string_record_label_validator_valid_label():
    """Test that StringRecordLabelValidator works with a valid label"""
    labels = ["ABC", "DEF", "GHI"]
    validator = StringRecordLabelValidator(labels)
    assert validator.validate(MagicMock(), "DEF")


def test_string_record_label_validator_invalid_label(caplog):
    """Test that StringRecordLabelValidator fails with an invalid label
    and emits a warning"""
    labels = ["ABC", "DEF", "GHI"]
    record = MagicMock()
    record.name = "TEST:NAME"
    validator = StringRecordLabelValidator(labels)
    assert validator.validate(record, "JKL") is False

    assert "Value JKL not valid for record TEST:NAME" in caplog.text


def test_process_labels_warns_long_label(ioc_record_factory: IocRecordFactory, caplog):
    """Test that _process_labels will automatically truncate long labels and
    emit a warning"""
    labels, index = ioc_record_factory._process_labels(
        ["ABC", "DEF", "AVeryLongLabelThatDoesNotFit"], "AVeryLongLabelThatDoesNotFit"
    )

    assert labels[index] == "AVeryLongLabelThatDoesNot"

    assert "One or more labels do not fit EPICS maximum length" in caplog.text


@pytest.mark.parametrize(
    "type, subtype",
    [
        ("UnknownType", "UnknownSubtype"),
        ("time", "UnknownSubtype"),
        ("UnknownType", "bits"),
    ],
)
def test_unknown_type_subtype(
    ioc_record_factory: IocRecordFactory, caplog, type: str, subtype: str
):
    """Test that an unknown field type logs the expected errors"""

    field_info = FieldInfo(type, subtype, None)
    ioc_record_factory.create_record(EpicsName("TEST:NAME"), field_info, {})

    assert f"Unrecognised type {(type, subtype)} while processing record" in caplog.text


@pytest.mark.asyncio
async def test_changes_update_on_error(caplog):
    """Test that errors reported from *CHANGES? are correctly marked in EPICS records"""
    caplog.set_level(logging.INFO)

    client = AsyncioClient("123")
    client.send = AsyncMock()  # type: ignore

    # Faked response that marks the record as in error
    returned_changes = Changes({}, [], ["ABC.DEF"], {})

    client.send.return_value = returned_changes

    record_info = RecordInfo(None, is_in_record=True)
    record_info.record = MagicMock()

    all_records = {EpicsName("ABC1:DEF"): record_info}
    poll_period = 0.1
    all_values_dict = {}

    try:
        await asyncio.wait_for(
            update(client, all_records, poll_period, all_values_dict), timeout=0.3
        )
    except asyncio.TimeoutError:
        pass

    record_info.record.set_alarm.assert_called_with(3, 17)
    assert "PandA reports field in error" in caplog.text
    assert "Setting record ABC1:DEF to invalid value error state." in caplog.text
