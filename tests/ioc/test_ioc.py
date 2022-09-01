import asyncio
from pathlib import Path
from typing import Dict, List

import numpy
import pytest
from aioca import caget, camonitor, caput, purge_channel_caches
from conftest import TEST_PREFIX, TIMEOUT
from mock import AsyncMock, patch
from mock.mock import MagicMock, call
from numpy import ndarray
from softioc import builder, fields

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetLine, Put
from pandablocks.ioc._types import (
    ONAM_STR,
    ZNAM_STR,
    EpicsName,
    InErrorException,
    ScalarRecordValue,
)
from pandablocks.ioc.ioc import (
    IocRecordFactory,
    _BlockAndFieldInfo,
    _ensure_block_number_present,
    _RecordUpdater,
    _TimeRecordUpdater,
    introspect_panda,
)
from pandablocks.responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    BlockInfo,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TableFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)
from tests.conftest import DummyServer


@pytest.fixture
def record_updater() -> _RecordUpdater:
    """Create a near-empty _RecordUpdater with a mocked client"""
    client = AsyncioClient("123")
    client.send = AsyncMock()  # type: ignore
    return _RecordUpdater(EpicsName("ABC:DEF"), client, float, {}, None)


@pytest.fixture
def ioc_record_factory(clear_records: None):
    """Create a new IocRecordFactory instance with a new, unique, namespace.
    This means each test can run in the same process, as each test will get
    its own namespace.
    """
    return IocRecordFactory(AsyncioClient("123"), TEST_PREFIX, {})


TEST_RECORD = EpicsName("TEST:RECORD")


@pytest.mark.asyncio
async def test_create_softioc_system(
    dummy_server_system,
    subprocess_ioc,
    table_unpacked_data: Dict[EpicsName, ndarray],
):
    """Top-level system test of the entire program, using some pre-canned data. Tests
    that the input data is turned into a collection of records with the appropriate
    values."""

    assert await caget(TEST_PREFIX + ":PCAP1:TRIG_EDGE") == 1  # == Falling
    assert await caget(TEST_PREFIX + ":PCAP1:GATE") == "CLOCK1.OUT"
    assert await caget(TEST_PREFIX + ":PCAP1:GATE:DELAY") == 1
    assert await caget(TEST_PREFIX + ":PCAP1:GATE:MAX_DELAY") == 100

    pcap1_label = await caget(TEST_PREFIX + ":PCAP1:LABEL")
    assert numpy.array_equal(
        pcap1_label,
        numpy.array(list("PcapMetadataLabel".encode() + b"\0"), dtype=numpy.uint8),
    )

    # Check table fields
    for field_name, expected_array in table_unpacked_data.items():
        actual_array = await caget(TEST_PREFIX + ":SEQ1:TABLE:" + field_name)
        assert numpy.array_equal(actual_array, expected_array)


@pytest.mark.asyncio
async def test_create_softioc_update(
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that the update mechanism correctly changes record values when PandA
    reports values have changed"""

    # Add more GetChanges data. Include some trailing empty changesets to allow test
    # code to run.
    dummy_server_system.send += ["!PCAP1.TRIG_EDGE=Either\n.", ".", "."]

    try:
        # Set up a monitor to wait for the expected change
        capturing_queue: asyncio.Queue = asyncio.Queue()
        monitor = camonitor(TEST_PREFIX + ":PCAP1:TRIG_EDGE", capturing_queue.put)

        curr_val = await asyncio.wait_for(capturing_queue.get(), 2)
        # First response is the current value
        assert curr_val == 1

        # Wait for the new value to appear
        curr_val = await asyncio.wait_for(capturing_queue.get(), 10)
        assert curr_val == 2

    finally:
        monitor.close()
        purge_channel_caches()


# TODO: Enable this test once PythonSoftIOC issue #53 is resolved
# @pytest.mark.asyncio
# async def test_create_softioc_update_in_error(
#     dummy_server_system: DummyServer,
#     subprocess_ioc,
# ):
#     """Test that the update mechanism correctly marks records as in error when PandA
#     reports the associated field is in error"""

#     # Add more GetChanges data. Include some trailing empty changesets to allow test
#     # code to run.
#     dummy_server_system.send += [
#         "!PCAP1.TRIG_EDGE (error)\n.",
#         ".",
#         ".",
#         ".",
#         ".",
#         ".",
#         ".",
#     ]

#     try:
#         # Set up a monitor to wait for the expected change
#         capturing_queue: asyncio.Queue = asyncio.Queue()
#         monitor = camonitor(TEST_PREFIX + ":PCAP1:TRIG_EDGE", capturing_queue.put)

#         curr_val = await asyncio.wait_for(capturing_queue.get(), 2)
#         # First response is the current value
#         assert curr_val == 1

# # Wait for the new value to appear
# Cannot do this due to PythonSoftIOC issue #53.
# err_val: AugmentedValue = await asyncio.wait_for(capturing_queue.get(), 100)
# assert err_val.severity == alarm.INVALID_ALARM
# assert err_val.status == alarm.UDF_ALARM

#     finally:
#         monitor.close()
#         purge_channel_caches()


@pytest.mark.asyncio
async def test_create_softioc_record_update_send_to_panda(
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that updating a record causes the new value to be sent to PandA"""
    # Set the special response for the server
    dummy_server_system.expected_message_responses.update(
        {"PCAP1.TRIG_EDGE=Either": "OK"}
    )

    await caput(TEST_PREFIX + ":PCAP1:TRIG_EDGE", "Either", wait=True, timeout=TIMEOUT)

    # Confirm the server received the expected string
    assert (
        "PCAP1.TRIG_EDGE=Either" not in dummy_server_system.expected_message_responses
    )


@pytest.mark.asyncio
async def test_create_softioc_arm_disarm(
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that the Arm and Disarm commands are correctly sent to PandA"""

    # Set the special response for the server
    dummy_server_system.expected_message_responses.update(
        {"*PCAP.ARM=": "OK", "*PCAP.DISARM=": "OK"}
    )

    await caput(TEST_PREFIX + ":PCAP:ARM", 1, wait=True, timeout=TIMEOUT)

    await caput(TEST_PREFIX + ":PCAP:ARM", 0, wait=True, timeout=TIMEOUT)

    # Confirm the server received the expected strings
    assert "*PCAP.ARM=" not in dummy_server_system.expected_message_responses
    assert "*PCAP.DISARM=" not in dummy_server_system.expected_message_responses


def test_ensure_block_number_present():
    assert _ensure_block_number_present("ABC.DEF.GHI") == "ABC1.DEF.GHI"
    assert _ensure_block_number_present("JKL1.MNOP") == "JKL1.MNOP"


@pytest.mark.asyncio
async def test_create_softioc_time_panda_changes(
    dummy_server_time: DummyServer,
    subprocess_ioc,
):
    """Test that the UNITS and MIN values of a TIME field correctly reflect into EPICS
    records when the value changes on the PandA"""
    # Check that the server has started, and has drained all messages
    assert not dummy_server_time.expected_message_responses

    # Set up monitors for expected changes when the UNITS are changed,
    # and check the initial values are correct
    egu_queue: asyncio.Queue = asyncio.Queue()
    m1 = camonitor(
        TEST_PREFIX + ":PULSE1:DELAY.EGU",
        egu_queue.put,
    )
    assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "ms"

    units_queue: asyncio.Queue = asyncio.Queue()
    m2 = camonitor(TEST_PREFIX + ":PULSE1:DELAY:UNITS", units_queue.put, datatype=str)
    assert await asyncio.wait_for(units_queue.get(), TIMEOUT) == "ms"

    drvl_queue: asyncio.Queue = asyncio.Queue()
    m3 = camonitor(
        TEST_PREFIX + ":PULSE1:DELAY.DRVL",
        drvl_queue.put,
    )
    assert await asyncio.wait_for(drvl_queue.get(), TIMEOUT) == 8e-06

    # These will be responses to repeated *CHANGES? requests made, once per second
    dummy_server_time.send += ["!PULSE.DELAY=0.1\n!PULSE1.DELAY.UNITS=s\n."]
    dummy_server_time.send += ["."] * 5

    # Changing the UNITS should trigger a request for the MIN
    dummy_server_time.expected_message_responses.update(
        {"PULSE1.DELAY.MIN?": "OK =8e-09"}
    )

    assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "s"
    assert await asyncio.wait_for(units_queue.get(), TIMEOUT) == "s"
    assert await asyncio.wait_for(drvl_queue.get(), TIMEOUT) == 8e-09

    m1.close()
    m2.close()
    m3.close()


@pytest.mark.asyncio
async def test_create_softioc_time_epics_changes(
    dummy_server_time: DummyServer,
    subprocess_ioc,
):
    """Test that the UNITS and MIN values of a TIME field correctly sent to the PandA
    when an EPICS record is updated"""
    # Check that the server has started, and has drained all messages
    assert not dummy_server_time.expected_message_responses

    # Set up monitors for expected changes when the UNITS are changed,
    # and check the initial values are correct
    egu_queue: asyncio.Queue = asyncio.Queue()
    m1 = camonitor(
        TEST_PREFIX + ":PULSE1:DELAY.EGU",
        egu_queue.put,
    )
    assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "ms"

    units_queue: asyncio.Queue = asyncio.Queue()
    m2 = camonitor(TEST_PREFIX + ":PULSE1:DELAY:UNITS", units_queue.put, datatype=str)
    assert await asyncio.wait_for(units_queue.get(), TIMEOUT) == "ms"

    drvl_queue: asyncio.Queue = asyncio.Queue()
    m3 = camonitor(
        TEST_PREFIX + ":PULSE1:DELAY.DRVL",
        drvl_queue.put,
    )
    assert await asyncio.wait_for(drvl_queue.get(), TIMEOUT) == 8e-06

    # We should send one message to set the UNITS, and a second to query the new MIN
    dummy_server_time.expected_message_responses.update(
        [
            ("PULSE1.DELAY.UNITS=min", "OK"),
            ("PULSE1.DELAY.MIN?", "OK =1.333333333e-10"),
        ]
    )

    # Change the UNITS
    assert await caput(
        TEST_PREFIX + ":PULSE1:DELAY:UNITS", "min", wait=True, timeout=TIMEOUT
    )

    assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "min"
    assert await asyncio.wait_for(units_queue.get(), TIMEOUT) == "min"
    assert await asyncio.wait_for(drvl_queue.get(), TIMEOUT) == 1.333333333e-10

    # Confirm the second round of expected messages were found
    assert not dummy_server_time.expected_message_responses

    m1.close()
    m2.close()
    m3.close()


@pytest.mark.asyncio
async def test_introspect_panda(
    dummy_server_introspect_panda,
    table_field_info: TableFieldInfo,
    table_data: List[str],
):
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
                    labels=["Rising", "Falling", "Either"],
                ),
                "GATE": BitMuxFieldInfo(
                    type="bit_mux",
                    subtype=None,
                    description="Gate Desc",
                    max_delay=100,
                    labels=["TTLIN1.VAL", "INENC1.A", "CLOCK1.OUT"],
                ),
            },
            values={
                EpicsName("PCAP1:TRIG_EDGE"): "Falling",
                EpicsName("PCAP1:GATE"): "CLOCK1.OUT",
                EpicsName("PCAP1:GATE:DELAY"): "1",
                EpicsName("PCAP1:LABEL"): "PcapMetadataLabel",
            },
        )

        assert data["SEQ"] == _BlockAndFieldInfo(
            block_info=BlockInfo(number=1, description="SEQ Desc"),
            fields={
                "TABLE": table_field_info,
            },
            values={EpicsName("SEQ1:TABLE"): table_data},
        )

        assert all_values_dict == {
            "PCAP1:TRIG_EDGE": "Falling",
            "PCAP1:GATE": "CLOCK1.OUT",
            "PCAP1:GATE:DELAY": "1",
            "PCAP1:LABEL": "PcapMetadataLabel",
            "SEQ1:TABLE": table_data,
        }


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

    # Configure the updater with mocked record and value
    mocked_record = MagicMock()
    record_updater.add_record(mocked_record)

    record_updater.all_values_dict = {EpicsName("ABC:DEF"): "999"}

    mocked_send: AsyncMock = record_updater.client.send  # type: ignore
    mocked_send.side_effect = Exception("Injected exception")

    await record_updater.update("1.0")

    mocked_record.set.assert_called_once_with("999", process=False)


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
                f"{TEST_RECORD}:MAX",
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
                f"{TEST_RECORD}:MAX",
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
                f"{TEST_RECORD}:MAX",
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
async def test_softioc_records_block(
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that the records created are blocking, and wait until they finish their
    on_update processing.

    Note that a lot of other tests implicitly test this feature too - any test that
    uses caput with wait=True is effectively testing this."""
    # Set the special response for the server
    dummy_server_system.expected_message_responses.update({"*PCAP.ARM=": "OK"})

    await caput(TEST_PREFIX + ":PCAP:ARM", 1, wait=True, timeout=TIMEOUT)

    # Confirm the server received the expected string
    assert "*PCAP.ARM=" not in dummy_server_system.expected_message_responses


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
    expected_args = ["BASE:RECORD.EGU", fields.DBF_STRING, 1]
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
    expected_args = ["BASE:RECORD.DRVL", fields.DBF_DOUBLE, 1]
    for arg in expected_args:
        assert arg in put_field_args
    assert type(put_field_args[2]) == int
