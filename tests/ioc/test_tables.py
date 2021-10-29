from typing import Dict, List

import numpy
import numpy.testing
import pytest
from mock import AsyncMock
from mock.mock import MagicMock, PropertyMock, call
from numpy import array, ndarray
from softioc import alarm

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetMultiline, Put
from pandablocks.ioc._tables import (
    TableFieldRecordContainer,
    TableModeEnum,
    TablePacking,
    _TableUpdater,
)
from pandablocks.ioc._types import (
    EpicsName,
    RecordValue,
    _InErrorException,
    _RecordInfo,
)
from pandablocks.responses import TableFieldDetails, TableFieldInfo

PANDA_FORMAT_TABLE_NAME = "SEQ1.TABLE"
EPICS_FORMAT_TABLE_NAME = "SEQ1:TABLE"


@pytest.fixture
def table_data_dict(table_data: List[str]) -> Dict[EpicsName, RecordValue]:
    return {EpicsName(EPICS_FORMAT_TABLE_NAME): table_data}


@pytest.fixture
def table_fields_records(
    table_fields: Dict[str, TableFieldDetails],
    table_unpacked_data: Dict[EpicsName, ndarray],
) -> Dict[str, TableFieldRecordContainer]:
    """A faked list of records containing the table_unpacked_data"""

    data = {}
    for (field_name, field_info), data_array in zip(
        table_fields.items(), table_unpacked_data.values()
    ):
        mocked_record = MagicMock()
        type(mocked_record).name = PropertyMock(
            return_value=EPICS_FORMAT_TABLE_NAME + ":" + field_name
        )
        mocked_record.get = MagicMock(return_value=data_array)
        record_info = _RecordInfo(mocked_record, lambda x: None)
        data[field_name] = TableFieldRecordContainer(field_info, record_info)
    return data


@pytest.fixture
def table_updater(
    table_field_info: TableFieldInfo,
    table_fields_records: Dict[str, TableFieldRecordContainer],
    table_data_dict: Dict[EpicsName, RecordValue],
    clear_records: None,
    table_unpacked_data: Dict[EpicsName, ndarray],
) -> _TableUpdater:
    """Provides a _TableUpdater with configured records and mocked functionality"""
    client = AsyncioClient("123")
    client.send = AsyncMock()  # type: ignore
    # mypy doesn't play well with mocking so suppress error

    mocked_mode_record = MagicMock()
    # Default mode record to VIEW, as per default construction
    mocked_mode_record.get = MagicMock(return_value=TableModeEnum.VIEW.value)
    mocked_mode_record.set = MagicMock()
    mode_record_info = _RecordInfo(
        mocked_mode_record,
        lambda x: None,
        labels=[
            TableModeEnum.VIEW.name,
            TableModeEnum.EDIT.name,
            TableModeEnum.SUBMIT.name,
            TableModeEnum.DISCARD.name,
        ],
    )

    updater = _TableUpdater(
        client,
        EpicsName(EPICS_FORMAT_TABLE_NAME),
        table_field_info,
        table_data_dict,
        EpicsName(EPICS_FORMAT_TABLE_NAME),
    )

    # Put mocks into TableUpdater
    updater.mode_record_info = mode_record_info
    updater.index_record = MagicMock()
    updater.table_scalar_records[EpicsName("SEQ1:TABLE:POSITION:SCALAR")] = MagicMock()
    for field_name, table_record_container in updater.table_fields_records.items():
        assert table_record_container.record_info
        table_record_container.record_info.record = MagicMock()
        type(table_record_container.record_info.record).name = PropertyMock(
            return_value=EPICS_FORMAT_TABLE_NAME + ":" + field_name
        )
        table_record_container.record_info.record.get = MagicMock(
            return_value=table_unpacked_data[EpicsName(field_name)]
        )

    return updater


def test_table_packing_unpack(
    table_field_info: TableFieldInfo,
    table_fields_records: Dict[str, TableFieldRecordContainer],
    table_data: List[str],
    table_unpacked_data,
):
    """Test table unpacking works as expected"""
    assert table_field_info.row_words
    unpacked = TablePacking.unpack(
        table_field_info.row_words, table_fields_records, table_data
    )

    for actual, expected in zip(unpacked, table_unpacked_data.values()):
        assert numpy.array_equal(actual, expected)


def test_table_packing_pack(
    table_field_info: TableFieldInfo,
    table_fields_records: Dict[str, TableFieldRecordContainer],
    table_data: List[str],
):
    """Test table unpacking works as expected"""
    assert table_field_info.row_words
    unpacked = TablePacking.pack(table_field_info.row_words, table_fields_records)

    for actual, expected in zip(unpacked, table_data):
        assert actual == expected


def test_table_packing_pack_length_mismatched(
    table_field_info: TableFieldInfo,
    table_fields_records: Dict[str, TableFieldRecordContainer],
):
    """Test that mismatching lengths on waveform inputs causes an exception"""
    assert table_field_info.row_words

    # Adjust one of the record lengths so it mismatches
    record_info = table_fields_records[EpicsName("OUTC1")].record_info
    assert record_info
    record_info.record.get = MagicMock(return_value=array([1, 2, 3, 4, 5, 6, 7, 8]))

    with pytest.raises(AssertionError):
        TablePacking.pack(table_field_info.row_words, table_fields_records)


def test_table_packing_roundtrip(
    table_field_info: TableFieldInfo,
    table_fields: Dict[str, TableFieldDetails],
    table_fields_records: Dict[str, TableFieldRecordContainer],
    table_data: List[str],
):
    """Test that calling unpack -> pack yields the same data"""
    assert table_field_info.row_words
    unpacked = TablePacking.unpack(
        table_field_info.row_words, table_fields_records, table_data
    )

    # Put these values into Mocks for the Records
    data: Dict[str, TableFieldRecordContainer] = {}
    for (field_name, field_info), data_array in zip(table_fields.items(), unpacked):
        mocked_record = MagicMock()
        mocked_record.get = MagicMock(return_value=data_array)
        record_info = _RecordInfo(mocked_record, lambda x: None)
        data[field_name] = TableFieldRecordContainer(field_info, record_info)

    packed = TablePacking.pack(table_field_info.row_words, data)

    assert packed == table_data


def test_table_updater_fields_sorted(table_updater: _TableUpdater):
    """Test that the field sorting done in post_init has occurred"""

    # Bits start at 0
    curr_bit = -1
    for field in table_updater.table_fields_records.values():
        field_details = field.field
        assert curr_bit < field_details.bit_low, "Fields are not in bit order"
        assert (
            field_details.bit_low <= field_details.bit_high  # fields may be 1 bit wide
        ), "Field had incorrect bit_low and bit_high order"
        assert (
            curr_bit < field_details.bit_high
        ), "Field had bit_high lower than bit_low"
        curr_bit = field_details.bit_high


def test_table_updater_validate_mode_view(table_updater: _TableUpdater):
    """Test the validate method when mode is View"""

    # View is default in table_updater
    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is False


def test_table_updater_validate_mode_edit(table_updater: _TableUpdater):
    """Test the validate method when mode is Edit"""

    table_updater.mode_record_info.record.get = MagicMock(
        return_value=TableModeEnum.EDIT.value
    )

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is True


def test_table_updater_validate_mode_submit(table_updater: _TableUpdater):
    """Test the validate method when mode is Submit"""

    table_updater.mode_record_info.record.get = MagicMock(
        return_value=TableModeEnum.SUBMIT.value
    )

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is False


def test_table_updater_validate_mode_discard(table_updater: _TableUpdater):
    """Test the validate method when mode is Discard"""

    table_updater.mode_record_info.record.get = MagicMock(
        return_value=TableModeEnum.DISCARD.value
    )

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")
    assert table_updater.validate_waveform(record, "value is irrelevant") is False


def test_table_updater_validate_mode_unknown(table_updater: _TableUpdater):
    """Test the validate method when mode is unknown"""

    table_updater.mode_record_info.record.get = MagicMock(return_value="UnknownValue")
    table_updater.mode_record_info.record.set_alarm = MagicMock()

    record = MagicMock()
    record.name = MagicMock(return_value="NewRecord")

    assert table_updater.validate_waveform(record, "value is irrelevant") is False
    table_updater.mode_record_info.record.set_alarm.assert_called_once_with(
        alarm.INVALID_ALARM, alarm.UDF_ALARM
    )


@pytest.mark.asyncio
async def test_table_updater_update_mode_view(table_updater: _TableUpdater):
    """Test that update_mode with new value of VIEW takes no action"""
    await table_updater.update_mode(TableModeEnum.VIEW.value)

    assert (
        not table_updater.client.send.called  # type: ignore
    ), "client send method was unexpectedly called"
    assert (
        not table_updater.mode_record_info.record.set.called
    ), "record set method was unexpectedly called"


@pytest.mark.asyncio
async def test_table_updater_update_mode_submit(
    table_updater: _TableUpdater, table_data: List[str]
):
    """Test that update_mode with new value of SUBMIT sends data to PandA"""
    await table_updater.update_mode(TableModeEnum.SUBMIT.value)

    assert isinstance(table_updater.client.send, AsyncMock)
    table_updater.client.send.assert_called_once_with(
        Put(PANDA_FORMAT_TABLE_NAME, table_data)
    )

    table_updater.mode_record_info.record.set.assert_called_once_with(
        TableModeEnum.VIEW.value, process=False
    )


@pytest.mark.asyncio
async def test_table_updater_update_mode_submit_exception(
    table_updater: _TableUpdater,
    table_data: List[str],
    table_unpacked_data: Dict[EpicsName, ndarray],
):
    """Test that update_mode with new value of SUBMIT handles an exception from Put
    correctly"""

    assert isinstance(table_updater.client.send, AsyncMock)
    table_updater.client.send.side_effect = Exception("Mocked exception")

    await table_updater.update_mode(TableModeEnum.SUBMIT.value)

    table_updater.client.send.assert_called_once_with(
        Put(PANDA_FORMAT_TABLE_NAME, table_data)
    )

    # Confirm each record received the expected data
    for field_name, data in table_unpacked_data.items():
        # Note table_unpacked_data is deliberately in a different order to the sorted
        # data, hence use this lookup mechanism instead
        record_info = table_updater.table_fields_records[field_name].record_info
        assert record_info
        # numpy arrays don't play nice with mock's equality comparisons, do it ourself
        called_args = record_info.record.set.call_args

        numpy.testing.assert_array_equal(data, called_args[0][0])

    table_updater.mode_record_info.record.set.assert_called_once_with(
        TableModeEnum.VIEW.value, process=False
    )


@pytest.mark.asyncio
async def test_table_updater_update_mode_submit_exception_data_error(
    table_updater: _TableUpdater, table_data: List[str]
):
    """Test that update_mode with an exception from Put and an InErrorException behaves
    as expected"""
    assert isinstance(table_updater.client.send, AsyncMock)
    table_updater.client.send.side_effect = Exception("Mocked exception")

    table_updater.all_values_dict[
        EpicsName(EPICS_FORMAT_TABLE_NAME)
    ] = _InErrorException("Mocked in error exception")

    await table_updater.update_mode(TableModeEnum.SUBMIT.value)

    for field_record in table_updater.table_fields_records.values():
        assert field_record.record_info
        record = field_record.record_info.record
        record.set.assert_not_called()

    table_updater.client.send.assert_called_once_with(
        Put(PANDA_FORMAT_TABLE_NAME, table_data)
    )


@pytest.mark.asyncio
async def test_table_updater_update_mode_discard(
    table_updater: _TableUpdater,
    table_data: List[str],
    table_unpacked_data: Dict[EpicsName, ndarray],
):
    """Test that update_mode with new value of DISCARD resets record data"""
    assert isinstance(table_updater.client.send, AsyncMock)
    table_updater.client.send.return_value = table_data

    await table_updater.update_mode(TableModeEnum.DISCARD.value)

    table_updater.client.send.assert_called_once_with(
        GetMultiline(PANDA_FORMAT_TABLE_NAME)
    )

    # Confirm each record received the expected data
    for field_name, data in table_unpacked_data.items():
        # Note table_unpacked_data is deliberately in a different order to the sorted
        # data, hence use this lookup mechanism instead
        record_info = table_updater.table_fields_records[field_name].record_info
        assert record_info
        # numpy arrays don't play nice with mock's equality comparisons, do it ourself
        called_args = record_info.record.set.call_args

        numpy.testing.assert_array_equal(data, called_args[0][0])

    table_updater.mode_record_info.record.set.assert_called_once_with(
        TableModeEnum.VIEW.value, process=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enum_val", [TableModeEnum.EDIT.value, TableModeEnum.VIEW.value]
)
async def test_table_updater_update_mode_other(
    table_updater: _TableUpdater,
    table_unpacked_data: Dict[EpicsName, ndarray],
    enum_val: int,
):
    """Test that update_mode with non-SUBMIT or DISCARD values takes no action"""

    await table_updater.update_mode(enum_val)

    assert isinstance(table_updater.client.send, AsyncMock)
    table_updater.client.send.assert_not_called()

    # Confirm each record was not called
    for field_name, data in table_unpacked_data.items():
        record_info = table_updater.table_fields_records[field_name].record_info
        assert record_info

        record_info.record.assert_not_called()

    table_updater.mode_record_info.record.set.assert_not_called()


def test_table_updater_update_table(
    table_updater: _TableUpdater,
    table_data: List[str],
    table_unpacked_data: Dict[EpicsName, ndarray],
):
    """Test that update_table updates records with the new values"""

    # update_scalar is too complex to test as well, so mock it out
    table_updater._update_scalar = MagicMock()  # type: ignore

    table_updater.update_table(table_data)

    table_updater.mode_record_info.record.get.assert_called_once()

    # Confirm each record received the expected data
    for field_name, data in table_unpacked_data.items():
        # Note table_unpacked_data is deliberately in a different order to the sorted
        # data, hence use this lookup mechanism instead
        record_info = table_updater.table_fields_records[field_name].record_info
        assert record_info
        # numpy arrays don't play nice with mock's equality comparisons, do it ourself
        called_args = record_info.record.set.call_args

        numpy.testing.assert_array_equal(data, called_args[0][0])

    table_updater._update_scalar.assert_called()


def test_table_updater_update_table_not_view(
    table_updater: _TableUpdater,
    table_data: List[str],
    table_unpacked_data: Dict[EpicsName, ndarray],
):
    """Test that update_table does nothing when mode is not VIEW"""

    # update_scalar is too complex to test as well, so mock it out
    table_updater._update_scalar = MagicMock()  # type: ignore

    table_updater.mode_record_info.record.get.return_value = TableModeEnum.EDIT

    table_updater.update_table(table_data)

    table_updater.mode_record_info.record.get.assert_called_once()

    # Confirm the records were not called
    for field_name, data in table_unpacked_data.items():
        # Note table_unpacked_data is deliberately in a different order to the sorted
        # data, hence use this lookup mechanism instead
        record_info = table_updater.table_fields_records[field_name].record_info
        assert record_info
        record_info.record.set.assert_not_called()


@pytest.mark.asyncio
async def test_table_updater_update_index(
    table_updater: _TableUpdater,
    table_fields: Dict[str, TableFieldDetails],
):
    """Test that update_index passes the full list of records to _update_scalar"""

    # Just need to prove it was called, not that it ran
    table_updater._update_scalar = MagicMock()  # type: ignore

    await table_updater.update_index(None)

    calls = []
    for field in table_fields.keys():
        calls.append(call(EPICS_FORMAT_TABLE_NAME + ":" + field))

    table_updater._update_scalar.assert_has_calls(calls, any_order=True)


def test_table_updater_update_scalar(
    table_updater: _TableUpdater,
):
    """Test that update_scalar correctly updates the scalar record for a waveform"""
    scalar_record_name = EpicsName("SEQ1:TABLE:POSITION:SCALAR")
    scalar_record = table_updater.table_scalar_records[scalar_record_name].record

    table_updater.index_record.get.return_value = 1

    table_updater._update_scalar("ABC:SEQ1:TABLE:POSITION")

    scalar_record.set.assert_called_once_with(
        678, severity=alarm.NO_ALARM, alarm=alarm.UDF_ALARM
    )


def test_table_updater_update_scalar_index_out_of_bounds(
    table_updater: _TableUpdater,
):
    """Test that update_scalar handles an invalid index"""
    scalar_record_name = EpicsName("SEQ1:TABLE:POSITION:SCALAR")
    scalar_record = table_updater.table_scalar_records[scalar_record_name].record

    table_updater.index_record.get.return_value = 99

    table_updater._update_scalar("ABC:SEQ1:TABLE:POSITION")

    scalar_record.set.assert_called_once_with(
        0, severity=alarm.INVALID_ALARM, alarm=alarm.UDF_ALARM
    )
