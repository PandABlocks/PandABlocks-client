import asyncio
import logging
import sys
import time
from multiprocessing import Process
from typing import Dict, Generator, List

import numpy
import pytest
from aioca import caget, camonitor, caput, purge_channel_caches
from epicsdbbuilder import ResetRecords
from mock import AsyncMock, patch
from mock.mock import MagicMock
from numpy import ndarray
from softioc import asyncio_dispatcher
from softioc.device_core import RecordLookup

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Put
from pandablocks.ioc._types import EpicsName
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
    tmp.append(".")  # Add the terminator
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
    dummy_server_in_thread.debug = True
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

    # Have to add our own logger, otherwise pytest doesn't see logging messages from IOC
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    logging.getLogger("").addHandler(sh)

    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    asyncio.run_coroutine_threadsafe(inner_wrapper(), dispatcher.loop).result()


@pytest.fixture
def subprocess_ioc(caplog) -> Generator:
    """Run the IOC in its own subprocess"""

    p = Process(target=ioc_wrapper)
    p.start()
    time.sleep(3)  # Give IOC some time to start up
    yield
    p.terminate()
    p.join(10)
    # Should never take anywhere near 10 seconds to terminate, it's just there
    # to ensure the test doesn't hang indefinitely during cleanup

    # TODO: These shouldn't need to be here - records are on another process.
    # But they will need to go somewhere.
    # Remove any records created at epicsdbbuilder layer
    ResetRecords()
    # And at pythonSoftIoc level
    # TODO: Remove this hack and use use whatever comes out of
    # https://github.com/dls-controls/pythonSoftIOC/issues/56
    RecordLookup._RecordDirectory.clear()


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
    """Test that the update mechanism correctly changes record values when PandA sends
    data"""

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
        curr_val = await asyncio.wait_for(capturing_queue.get(), 30)
        assert curr_val == 2

    finally:
        monitor.close()
        purge_channel_caches()


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


def idfn(val):
    """helper function to nicely name parameterized test IDs"""
    if isinstance(val, FieldInfo):
        return val.type + "-" + str(val.subtype)  # subtype may be None
    elif isinstance(val, (dict, list)):
        return ""


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
