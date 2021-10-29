import asyncio
import logging
import time
from multiprocessing import Process
from pathlib import Path
from typing import Dict, Generator, List

import numpy
import pytest
from aioca import caget, camonitor, caput, purge_channel_caches
from conftest import custom_logger
from mock import AsyncMock, patch
from mock.mock import MagicMock, call
from numpy import ndarray
from softioc import asyncio_dispatcher, builder

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Put
from pandablocks.ioc._types import (
    ONAM_STR,
    ZNAM_STR,
    EpicsName,
    ScalarRecordValue,
    _InErrorException,
)
from pandablocks.ioc.ioc import (
    IocRecordFactory,
    _BlockAndFieldInfo,
    _ensure_block_number_present,
    _RecordUpdater,
    create_softioc,
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

TEST_PREFIX = "TEST-PREFIX"


@pytest.fixture
def record_updater() -> _RecordUpdater:
    """Create a near-empty _RecordUpdater with a mocked client"""
    client = AsyncioClient("123")
    client.send = AsyncMock()  # type: ignore
    return _RecordUpdater(EpicsName("ABC:DEF"), client, float, {})


@pytest.fixture
def ioc_record_factory(clear_records: None):
    """Create a new IocRecordFactory instance with a new, unique, namespace.
    This means each test can run in the same process, as each test will get
    its own namespace.
    """
    return IocRecordFactory(AsyncioClient("123"), TEST_PREFIX, {})


@pytest.fixture
def dummy_server_introspect_panda(
    dummy_server_in_thread: DummyServer, table_data: List[str]
):
    """A dummy server that responds to all the requests introspect_panda makes
    during its operation.
    Note that the order of responses was determined by trial and error."""
    get_changes_scalar_data = (
        # Note the deliberate concatenation across lines - this must be a single
        # entry in the list
        "!PCAP.TRIG_EDGE=Falling\n!PCAP.GATE=CLOCK1.OUT\n!PCAP.GATE.DELAY=1\n"
        "!*METADATA.LABEL_PCAP1=PcapMetadataLabel\n"
        "!SEQ1.TABLE<\n"
        "."
    )

    # Transform the plain list of values into one that PandA would send
    tmp = ["!" + s + "\n" for s in table_data]
    tmp.append(".")  # Add the multiline terminator
    get_changes_multiline_data = "".join(tmp)

    table_fields_data = (
        # Note the deliberate concatenation across lines - this must be a single
        # entry in the list
        "!15:0 REPEATS uint\n!19:16 TRIGGER enum\n!63:32 POSITION int\n"
        "!95:64 TIME1 uint\n!20:20 OUTA1 uint\n!21:21 OUTB1 uint\n!22:22 OUTC1 uint\n"
        "!23:23 OUTD1 uint\n!24:24 OUTE1 uint\n!25:25 OUTF1 uint\n!127:96 TIME2 uint\n"
        "!26:26 OUTA2 uint\n!27:27 OUTB2 uint\n!28:28 OUTC2 uint\n!29:29 OUTD2 uint\n"
        "!30:30 OUTE2 uint\n!31:31 OUTF2 uint\n."
    )

    trigger_field_labels = (
        # Note the deliberate concatenation across lines - this must be a single
        # entry in the list
        "!Immediate\n!BITA=0\n!BITA=1\n!BITB=0\n!BITB=1\n!BITC=0\n!BITC=1\n"
        "!POSA>=POSITION\n!POSA<=POSITION\n!POSB>=POSITION\n!POSB<=POSITION\n"
        "!POSC>=POSITION\n!POSC<=POSITION\n."
    )

    dummy_server_in_thread.send += [
        "!PCAP 1\n!SEQ 1\n.",  # BLOCK definitions
        "OK =PCAP Desc",
        "OK =SEQ Desc",
        "!TRIG_EDGE 3 param enum\n!GATE 1 bit_mux\n.",  # PCAP fields
        "!TABLE 7 table\n.",  # SEQ field
        get_changes_scalar_data,
        "!Rising\n!Falling\n!Either\n.",  # TRIG_EDGE enum labels
        "OK =100",  # GATE MAX_DELAY
        "!TTLIN1.VAL\n!INENC1.A\n!CLOCK1.OUT\n.",  # GATE labels
        "OK =Trig Edge Desc",
        "OK =Gate Desc",
        "OK =16384",  # TABLE MAX_LENGTH
        table_fields_data,
        "OK =Sequencer table of lines",  # TABLE Desc
        get_changes_multiline_data,
        trigger_field_labels,
        "OK =Number of times the line will repeat",  # Repeats field desc
        "OK =The trigger condition to start the phases",  # TRIGGER field desc
        "OK =The position that can be used in trigger condition",  # POSITION field desc
        "OK =The time the optional phase 1 should take",  # TIME1 desc
        "OK =Output A value during phase 1",  # OUTA1 desc
        "OK =Output B value during phase 1",  # OUTB1 desc
        "OK =Output C value during phase 1",  # OUTC1 desc
        "OK =Output D value during phase 1",  # OUTD1 desc
        "OK =Output E value during phase 1",  # OUTE1 desc
        "OK =Output F value during phase 1",  # OUTF1 desc
        "OK =The time the mandatory phase 2 should take",  # TIME2 desc
        "OK =Output A value during phase 2",  # OUTA2 desc
        "OK =Output B value during phase 2",  # OUTB2 desc
        "OK =Output C value during phase 2",  # OUTC2 desc
        "OK =Output D value during phase 2",  # OUTD2 desc
        "OK =Output E value during phase 2",  # OUTE2 desc
        "OK =Output F value during phase 2",  # OUTF2 desc
    ]
    # If you need to change the above responses,
    # it'll probably help to enable debugging on the server
    # dummy_server_in_thread.debug = True
    yield dummy_server_in_thread


@pytest.fixture
def dummy_server_system(dummy_server_introspect_panda: DummyServer):
    """A server for a full system test"""

    # Add data for GetChanges to consume. Number of items should be bigger than
    # the sleep time given during IOC startup
    dummy_server_introspect_panda.send += [
        ".",
        ".",
        ".",
        ".",
    ]

    yield dummy_server_introspect_panda


@patch("pandablocks.ioc.ioc.AsyncioClient.close")
@patch("pandablocks.ioc.ioc.softioc.interactive_ioc")
def ioc_wrapper(mocked_interactive_ioc: MagicMock, mocked_client_close: MagicMock):
    """Wrapper function to start the IOC and do some mocking"""

    async def inner_wrapper():
        create_softioc("localhost", TEST_PREFIX)
        # If you see an error on the below line, it probably means an unexpected
        # exception occurred during IOC startup
        mocked_interactive_ioc.assert_called_once()
        mocked_client_close.assert_called_once()
        # Leave this process running until its torn down by pytest
        await asyncio.Event().wait()

    custom_logger()
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    asyncio.run_coroutine_threadsafe(inner_wrapper(), dispatcher.loop).result()


@pytest.fixture
def subprocess_ioc(enable_codecov_multiprocess, caplog, caplog_workaround) -> Generator:
    """Run the IOC in its own subprocess. When finished check logging logged no
    messages of WARNING or higher level."""
    with caplog.at_level(logging.WARNING):
        with caplog_workaround():
            p = Process(target=ioc_wrapper)
            try:
                p.start()
                time.sleep(3)  # Give IOC some time to start up
                yield
            finally:
                p.terminate()
                p.join(10)
                # Should never take anywhere near 10 seconds to terminate, it's just
                # there to ensure the test doesn't hang indefinitely during cleanup

    if len(caplog.messages) > 0:
        # We expect all tests to pass without warnings (or worse) logged.
        assert (
            False
        ), f"At least one warning/error/exception logged during test: {caplog.records}"


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
    assert await caget(TEST_PREFIX + ":PCAP1:LABEL") == "PcapMetadataLabel"

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


@pytest.mark.asyncio
async def test_create_softioc_update_table(
    dummy_server_system: DummyServer,
    subprocess_ioc,
    table_unpacked_data,
):
    """Test that the update mechanism correctly changes table values when PandA
    reports values have changed"""

    # Add more GetChanges data. This adds two new rows and changes row 2 (1-indexed)
    # to all zero values. Include some trailing empty changesets to ensure test code has
    # time to run.
    dummy_server_system.send += [
        "!SEQ1.TABLE<\n.",
        # Deliberate concatenation here
        "!2457862149\n!4294967291\n!100\n!0\n!0\n!0\n!0\n!0\n!4293968720\n!0\n"
        "!9\n!9999\n!2035875928\n!444444\n!5\n!1\n!3464285461\n!4294967197\n!99999\n"
        "!2222\n.",
        ".",
        ".",
    ]

    try:
        # Set up a monitor to wait for the expected change
        capturing_queue: asyncio.Queue = asyncio.Queue()
        monitor = camonitor(TEST_PREFIX + ":SEQ1:TABLE:TIME1", capturing_queue.put)

        curr_val: ndarray = await asyncio.wait_for(capturing_queue.get(), 2)
        # First response is the current value
        assert numpy.array_equal(curr_val, table_unpacked_data["TIME1"])

        # Wait for the new value to appear
        curr_val = await asyncio.wait_for(capturing_queue.get(), 1000)
        assert numpy.array_equal(
            curr_val,
            [100, 0, 9, 5, 99999],
        )

        # And check some other columns too
        curr_val = await caget(TEST_PREFIX + ":SEQ1:TABLE:TRIGGER")
        assert numpy.array_equal(curr_val, [0, 0, 0, 9, 12])

        curr_val = await caget(TEST_PREFIX + ":SEQ1:TABLE:POSITION")
        assert numpy.array_equal(curr_val, [-5, 0, 0, 444444, -99])

        curr_val = await caget(TEST_PREFIX + ":SEQ1:TABLE:OUTD2")
        assert numpy.array_equal(curr_val, [0, 0, 1, 1, 0])

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
    # mocked_put: MagicMock,
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that updating a record causes the new value to be sent to PandA"""
    # Set the special response for the server
    dummy_server_system.expected_message_responses.update(
        {"PCAP1.TRIG_EDGE=Either": "OK"}
    )

    # Few more responses to GetChanges to suppress error messages
    dummy_server_system.send += [
        ".",
        ".",
        ".",
        ".",
        ".",
        ".",
        ".",
    ]
    await caput(TEST_PREFIX + ":PCAP1:TRIG_EDGE", "Either")

    # Give time for the on_update processing to occur
    await asyncio.sleep(5)

    # Confirm the server received the expected string
    assert (
        "PCAP1.TRIG_EDGE=Either" not in dummy_server_system.expected_message_responses
    )


@pytest.mark.asyncio
async def test_create_softioc_table_update_send_to_panda(
    # mocked_put: MagicMock,
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that updating a table causes the new value to be sent to PandA"""

    # Set the special response for the server
    dummy_server_system.expected_message_responses.update({"": "OK"})

    # Few more responses to GetChanges to suppress error messages
    dummy_server_system.send += [".", ".", ".", "."]

    await caput(TEST_PREFIX + ":SEQ1:TABLE:MODE", "EDIT")

    await caput(TEST_PREFIX + ":SEQ1:TABLE:REPEATS", [1, 1, 1])

    await caput(TEST_PREFIX + ":SEQ1:TABLE:MODE", "SUBMIT")

    # Give time for the on_update processing to occur
    await asyncio.sleep(2)

    # Confirm the server received the expected string
    assert "" not in dummy_server_system.expected_message_responses

    # Check the three numbers that should have updated from the REPEATS column change
    assert "2457862145" in dummy_server_system.received
    assert "269877249" in dummy_server_system.received
    assert "4293918721" in dummy_server_system.received


@pytest.mark.asyncio
async def test_create_softioc_arm_disarm(
    # mocked_put: MagicMock,
    dummy_server_system: DummyServer,
    subprocess_ioc,
):
    """Test that the Arm and Disarm commands are correctly sent to PandA"""

    # Set the special response for the server
    dummy_server_system.expected_message_responses.update(
        {"*PCAP.ARM=": "OK", "*PCAP.DISARM=": "OK"}
    )

    # Few more responses to GetChanges to suppress error messages
    dummy_server_system.send += [".", ".", ".", "."]

    await caput(TEST_PREFIX + ":PCAP:ARM", 1)
    # Give time for the on_update processing to occur
    await asyncio.sleep(1)

    await caput(TEST_PREFIX + ":PCAP:ARM", 0)
    # Give time for the on_update processing to occur
    await asyncio.sleep(1)

    # Confirm the server received the expected strings
    assert "*PCAP.ARM=" not in dummy_server_system.expected_message_responses
    assert "*PCAP.DISARM=" not in dummy_server_system.expected_message_responses


def test_ensure_block_number_present():
    assert _ensure_block_number_present("ABC.DEF.GHI") == "ABC1.DEF.GHI"
    assert _ensure_block_number_present("JKL1.MNOP") == "JKL1.MNOP"


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
        ioc_record_factory.create_record(TEST_RECORD, FieldInfo(type, "action"), {})
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
        initial_value=_InErrorException("Mocked exception"),
    )

    ioc_record_factory._create_record_info(
        EpicsName("SomeInRec"),
        None,
        builder.aIn,
        float,
        initial_value=_InErrorException("Mocked exception"),
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
