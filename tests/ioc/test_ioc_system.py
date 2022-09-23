import asyncio
from typing import Dict, List

import numpy
import pytest
from aioca import caget, camonitor, caput
from conftest import TEST_PREFIX, TIMEOUT
from numpy import ndarray

from pandablocks.asyncio import AsyncioClient
from pandablocks.ioc._types import EpicsName
from pandablocks.ioc.ioc import (
    _BlockAndFieldInfo,
    _ensure_block_number_present,
    introspect_panda,
)
from pandablocks.responses import (
    BitMuxFieldInfo,
    BlockInfo,
    EnumFieldInfo,
    TableFieldInfo,
)
from tests.conftest import DummyServer

# Test file for all tests that require a full setup system, with an IOC running in one
# process, a DummyServer in another, and the test in the main thread accessing data
# using Channel Access


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
    dummy_server_system.send += ["!PCAP1.TRIG_EDGE=Either\n."]
    dummy_server_system.send += ["."] * 100

    try:
        # Set up a monitor to wait for the expected change
        capturing_queue: asyncio.Queue = asyncio.Queue()
        monitor = camonitor(TEST_PREFIX + ":PCAP1:TRIG_EDGE", capturing_queue.put)

        curr_val = await asyncio.wait_for(capturing_queue.get(), TIMEOUT)
        # First response is the current value
        assert curr_val == 1

        # Wait for the new value to appear
        curr_val = await asyncio.wait_for(capturing_queue.get(), TIMEOUT)
        assert curr_val == 2

    finally:
        monitor.close()


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

    try:
        # Set up monitors for expected changes when the UNITS are changed,
        # and check the initial values are correct
        egu_queue: asyncio.Queue = asyncio.Queue()
        m1 = camonitor(
            TEST_PREFIX + ":PULSE1:DELAY.EGU",
            egu_queue.put,
        )
        assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "ms"

        units_queue: asyncio.Queue = asyncio.Queue()
        m2 = camonitor(
            TEST_PREFIX + ":PULSE1:DELAY:UNITS", units_queue.put, datatype=str
        )
        assert await asyncio.wait_for(units_queue.get(), TIMEOUT) == "ms"

        drvl_queue: asyncio.Queue = asyncio.Queue()
        m3 = camonitor(
            TEST_PREFIX + ":PULSE1:DELAY.DRVL",
            drvl_queue.put,
        )
        assert await asyncio.wait_for(drvl_queue.get(), TIMEOUT) == 8e-06

        # These will be responses to repeated *CHANGES? requests made
        dummy_server_time.send += ["!PULSE.DELAY=0.1\n!PULSE1.DELAY.UNITS=s\n."]
        dummy_server_time.send += ["."] * 100

        # Changing the UNITS should trigger a request for the MIN
        dummy_server_time.expected_message_responses.update(
            {"PULSE1.DELAY.MIN?": "OK =8e-09"}
        )

        assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "s"
        assert await asyncio.wait_for(units_queue.get(), TIMEOUT) == "s"
        assert await asyncio.wait_for(drvl_queue.get(), TIMEOUT) == 8e-09
    finally:
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

    try:
        # Set up monitors for expected changes when the UNITS are changed,
        # and check the initial values are correct
        egu_queue: asyncio.Queue = asyncio.Queue()
        m1 = camonitor(
            TEST_PREFIX + ":PULSE1:DELAY.EGU",
            egu_queue.put,
        )
        assert await asyncio.wait_for(egu_queue.get(), TIMEOUT) == "ms"

        units_queue: asyncio.Queue = asyncio.Queue()
        m2 = camonitor(
            TEST_PREFIX + ":PULSE1:DELAY:UNITS", units_queue.put, datatype=str
        )
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
    finally:
        m1.close()
        m2.close()
        m3.close()


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
async def test_pending_changes_blocks_record_set(
    dummy_server_system: DummyServer, subprocess_ioc
):
    """Test that when a value is Put to PandA and subsequently reported via *CHANGES?
    does not do another .set() on the record"""

    # Trigger a _RecordUpdater.update(), to do a Put command

    dummy_server_system.expected_message_responses.update(
        {"PCAP1.TRIG_EDGE=Either": "OK"}
    )

    await caput(TEST_PREFIX + ":PCAP1:TRIG_EDGE", "Either", wait=True, timeout=TIMEOUT)

    # Confirm the server received the expected string
    assert not dummy_server_system.expected_message_responses

    dummy_server_system.send += ["!PCAP1.TRIG_EDGE=Either\n.", ".", "."]

    async def expected_messages_received():
        """Wait until the expected messages have all been received by the server"""
        while len(dummy_server_system.send) > 2:
            await asyncio.sleep(0.1)

    await asyncio.wait_for(expected_messages_received(), timeout=TIMEOUT)
