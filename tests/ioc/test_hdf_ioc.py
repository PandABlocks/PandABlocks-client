# Tests for the _hdf_ioc.py file

import asyncio
import time
from multiprocessing import Process
from pathlib import Path
from typing import Generator

import h5py
import numpy
import pytest
from aioca import caget, camonitor, caput
from epicsdbbuilder import ResetRecords
from mock.mock import MagicMock
from softioc import asyncio_dispatcher, builder, softioc
from softioc.device_core import RecordLookup

from pandablocks.asyncio import AsyncioClient
from pandablocks.ioc._hdf_ioc import _HDF5RecordController
from tests.conftest import DummyServer

NAMESPACE_PREFIX = "HDF-RECORD-PREFIX"
HDF5_PREFIX = NAMESPACE_PREFIX + ":HDF5"
counter = 0


@pytest.fixture
def hdf5_controller() -> Generator:
    """Construct an HDF5 controller and mock various aspects of it"""
    global counter
    counter += 1

    hdf5_controller = _HDF5RecordController(
        AsyncioClient("localhost"), NAMESPACE_PREFIX + str(counter)
    )
    hdf5_controller._capture_control_record = MagicMock()
    # Default return value for capturing off, allowing validation method to pass
    hdf5_controller._capture_control_record.get = MagicMock(return_value=0)
    # return hdf5_controller
    yield hdf5_controller

    # Remove any records created at epicsdbbuilder layer
    ResetRecords()
    # TODO: Decide if we keep this or not
    # And at pythonSoftIoc level
    RecordLookup._RecordDirectory.clear()


def subprocess_func() -> None:
    """Function to start the HDF5 IOC"""
    # No need for counter as this runs in a separate process

    async def wrapper(dispatcher):
        builder.SetDeviceName(NAMESPACE_PREFIX)
        _HDF5RecordController(AsyncioClient("localhost"), NAMESPACE_PREFIX)
        builder.LoadDatabase()
        softioc.iocInit(dispatcher)

        await asyncio.Event().wait()

    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    asyncio.run_coroutine_threadsafe(wrapper(dispatcher), dispatcher.loop).result()


@pytest.fixture
def hdf5_subprocess_ioc() -> Generator:
    """Create an instance of HDF5 class in its own subprocess, then start the IOC"""

    p = Process(target=subprocess_func)
    p.start()
    time.sleep(2)  # Give IOC some time to start up
    yield
    p.terminate()
    p.join(10)
    # Should never take anywhere near 10 seconds to terminate, it's just there
    # to ensure the test doesn't hang indefinitely during cleanup


@pytest.mark.asyncio
async def test_hdf5_ioc(hdf5_subprocess_ioc):
    """Run the HDF5 module as its own IOC and check the expected records are created,
    with some default values checked"""
    HDF5_PREFIX = NAMESPACE_PREFIX + ":HDF5"
    val = await caget(HDF5_PREFIX + ":FilePath")
    assert val.size == 0

    # Mix and match between CamelCase and UPPERCASE to check aliases work
    val = await caget(HDF5_PREFIX + ":FILENAME")
    assert val.size == 0

    val = await caget(HDF5_PREFIX + ":NumCapture")
    assert val == 0

    val = await caget(HDF5_PREFIX + ":FlushPeriod")
    assert val == 1.0

    val = await caget(HDF5_PREFIX + ":CAPTURE")
    assert val == 0

    val = await caget(HDF5_PREFIX + ":Status")
    assert val == "OK"

    val = await caget(HDF5_PREFIX + ":Capturing")
    assert val == 0


def _string_to_buffer(string: str):
    """Convert a python string into a numpy buffer suitable for caput'ing to a Waveform
    record"""
    return numpy.frombuffer(string.encode(), dtype=numpy.uint8)


@pytest.mark.asyncio
async def test_hdf5_ioc_parameter_validate_works(hdf5_subprocess_ioc):
    """Run the HDF5 module as its own IOC and check the _parameter_validate method
    does not block updates, then blocks when capture record is changed"""

    # EPICS bug means caputs always appear to succeed, so do a caget to prove it worked

    await caput(HDF5_PREFIX + ":FilePath", _string_to_buffer("/new/path"), wait=True)
    val = await caget(HDF5_PREFIX + ":FilePath")
    assert val.tobytes().decode() == "/new/path"

    await caput(HDF5_PREFIX + ":FileName", _string_to_buffer("name.h5"), wait=True)
    val = await caget(HDF5_PREFIX + ":FileName")
    assert val.tobytes().decode() == "name.h5"

    await caput(HDF5_PREFIX + ":Capture", 1, wait=True)
    assert await caget(HDF5_PREFIX + ":Capture") == 1

    await caput(HDF5_PREFIX + ":FilePath", _string_to_buffer("/second/path"), wait=True)
    val = await caget(HDF5_PREFIX + ":FilePath")
    assert val.tobytes().decode() == "/new/path"  # put should have been blocked


@pytest.mark.asyncio
async def test_hdf5_file_writing(
    hdf5_subprocess_ioc, dummy_server_in_thread: DummyServer, raw_dump, tmp_path: Path
):
    """Test that an HDF5 file is written when Capture is enabled"""
    dummy_server_in_thread.data = raw_dump

    test_dir = str(tmp_path) + "\0"
    test_filename = "test.h5\0"

    await caput(HDF5_PREFIX + ":FilePath", _string_to_buffer(str(test_dir)), wait=True)
    val = await caget(HDF5_PREFIX + ":FilePath")
    assert val.tobytes().decode() == test_dir

    await caput(HDF5_PREFIX + ":FileName", _string_to_buffer(test_filename), wait=True)
    val = await caget(HDF5_PREFIX + ":FileName")
    assert val.tobytes().decode() == test_filename

    # Only a single FrameData in the example data
    await caput(HDF5_PREFIX + ":NumCapture", 1, wait=True)
    assert await caget(HDF5_PREFIX + ":NumCapture") == 1

    await caput(HDF5_PREFIX + ":Capture", 1, wait=True)
    assert await caget(HDF5_PREFIX + ":Capture") == 1

    # await asyncio.sleep(5)  # Give the capture some time to process
    # capturing_event = asyncio.Event()

    # async def wait_for_capturing_disabled():
    #     val = await caget(HDF5_PREFIX + ":Capturing")
    #     print(val)
    #     if not val:
    #         capturing_event.set()

    # camonitor(HDF5_PREFIX + ":Capturing", wait_for_capturing_disabled)

    # await asyncio.wait_for(capturing_event.wait(), timeout=1000)

    # TODO: Must be a better way to do this...
    await asyncio.sleep(5)

    # Close capture, thus closing hdf5 file
    await caput(HDF5_PREFIX + ":Capture", 0, wait=True)
    assert await caget(HDF5_PREFIX + ":Capture") == 0

    hdf_file = h5py.File(tmp_path / test_filename[:-1], "r")
    assert list(hdf_file) == [
        "COUNTER1.OUT.Max",
        "COUNTER1.OUT.Mean",
        "COUNTER1.OUT.Min",
        "COUNTER2.OUT.Mean",
        "COUNTER3.OUT.Value",
        "PCAP.BITS2.Value",
        "PCAP.SAMPLES.Value",
        "PCAP.TS_START.Value",
    ]


def test_hdf_parameter_validate_not_capturing(hdf5_controller: _HDF5RecordController):
    """Test that parameter_validate allows record updates when capturing is off"""

    hdf5_controller._capture_control_record.get.return_value = 0
    # Don't care about the record being validated, just mock it
    assert hdf5_controller._parameter_validate(MagicMock(), None) is True


def test_hdf_parameter_validate_capturing(hdf5_controller: _HDF5RecordController):
    """Test that parameter_validate blocks record updates when capturing is on"""

    hdf5_controller._capture_control_record.get.return_value = 1
    # Don't care about the record being validated, just mock it
    assert hdf5_controller._parameter_validate(MagicMock(), None) is False


def test_hdf_get_filename(
    hdf5_controller: _HDF5RecordController,
):
    """Test _get_filename works when all records have valid values"""

    # Mock this method, we test it explicitly later
    hdf5_controller._waveform_record_to_string = MagicMock(  # type: ignore
        side_effect=["/some/path", "some_filename"]
    )

    assert hdf5_controller._get_filename() == "/some/path/some_filename"


def test_hdf_capture_validate_valid_filename(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate passes when a valid filename is given"""
    hdf5_controller._get_filename = MagicMock(  # type: ignore
        return_value="/valid/file.h5"
    )

    assert hdf5_controller._capture_validate(None, 1) is True


def test_hdf_capture_validate_new_value_zero(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate passes when new value is zero"""
    assert hdf5_controller._capture_validate(None, 0) is True


def test_hdf_capture_validate_invalid_filename(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate fails when filename cannot be created"""
    hdf5_controller._get_filename = MagicMock(  # type: ignore
        side_effect=ValueError("Mocked value error")
    )

    assert hdf5_controller._capture_validate(None, 1) is False


def test_hdf_capture_validate_exception(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate fails due to other exceptions"""
    hdf5_controller._get_filename = MagicMock(  # type: ignore
        side_effect=Exception("Mocked error")
    )

    assert hdf5_controller._capture_validate(None, 1) is False


def test_hdf_waveform_record_to_string(
    hdf5_controller: _HDF5RecordController,
):
    """Test _waveform_record_to_string returns string version of array"""
    test_str = "Test String".encode() + b"\0"
    array = numpy.frombuffer(test_str, dtype=numpy.uint8)
    record = MagicMock()
    record.get = MagicMock(return_value=array)
    assert hdf5_controller._waveform_record_to_string(record) == test_str[:-1].decode()


def test_hdf_waveform_record_to_string_no_value(
    hdf5_controller: _HDF5RecordController,
):
    """Test _waveform_record_to_string raises exception when no value"""

    record = MagicMock()
    record.get = MagicMock(return_value=None)
    with pytest.raises(ValueError):
        hdf5_controller._waveform_record_to_string(record)


def test_hdf_numpy_to_string(
    hdf5_controller: _HDF5RecordController,
):
    """Test _numpy_to_string returns expected string"""
    test_str = "Test String".encode() + b"\0"
    array = numpy.frombuffer(test_str, dtype=numpy.uint8)
    assert hdf5_controller._numpy_to_string(array) == test_str[:-1].decode()


def test_hdf_numpy_to_string_bad_dtype(
    hdf5_controller: _HDF5RecordController,
):
    """Test _numpy_to_string raises exception when dtype is wrong"""
    test_str = "Test String".encode() + b"\0"
    array = numpy.frombuffer(test_str, dtype=numpy.uint8)
    array = array.astype(numpy.uint32)
    with pytest.raises(AssertionError):
        hdf5_controller._numpy_to_string(array)
