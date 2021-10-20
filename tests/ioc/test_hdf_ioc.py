# Tests for the _hdf_ioc.py file

from typing import Generator

import numpy
import pytest
from epicsdbbuilder import ResetRecords
from mock.mock import MagicMock
from softioc.device_core import RecordLookup

from pandablocks.asyncio import AsyncioClient
from pandablocks.ioc._hdf_ioc import _HDF5RecordController

TEST_PREFIX = "HDF-RECORD-PREFIX"
counter = 0


@pytest.fixture
def hdf5_controller() -> Generator:
    """Construct an HDF5 controller and mock various aspects of it"""
    global counter
    counter += 1

    hdf5_controller = _HDF5RecordController(
        AsyncioClient("123"), TEST_PREFIX + str(counter)
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


def test_hdf_template_validate(hdf5_controller: _HDF5RecordController):
    """Test _template_validate with some acceptable format strings"""

    array = numpy.frombuffer(
        "Valid Format String %s %s %d".encode() + b"\0", dtype=numpy.uint8
    )
    assert hdf5_controller._template_validate(MagicMock(), array) is True

    array = numpy.frombuffer("%s/%s_%d.h5".encode() + b"\0", dtype=numpy.uint8)
    assert hdf5_controller._template_validate(MagicMock(), array) is True

    array = numpy.frombuffer("%s/%d%s.h5".encode() + b"\0", dtype=numpy.uint8)
    assert hdf5_controller._template_validate(MagicMock(), array) is True


def test_hdf_template_validate_wrong_string_specifiers(
    hdf5_controller: _HDF5RecordController,
):
    """Test _template_validate with an invalid number of string format specifiers"""

    array = numpy.frombuffer("Invalid %s %d".encode() + b"\0", dtype=numpy.uint8)
    assert hdf5_controller._template_validate(MagicMock(), array) is False

    array = numpy.frombuffer("Invalid %s %s %s %d".encode() + b"\0", dtype=numpy.uint8)
    assert hdf5_controller._template_validate(MagicMock(), array) is False


def test_hdf_template_validate_wrong_number_specifiers(
    hdf5_controller: _HDF5RecordController,
):
    """Test _template_validate with an invalid number of number format specifiers"""

    array = numpy.frombuffer("Invalid String %s %s".encode() + b"\0", dtype=numpy.uint8)
    assert hdf5_controller._template_validate(MagicMock(), array) is False

    array = numpy.frombuffer(
        "Invalid String %s %s %d %d".encode() + b"\0", dtype=numpy.uint8
    )
    assert hdf5_controller._template_validate(MagicMock(), array) is False


def test_hdf_get_scheme(
    hdf5_controller: _HDF5RecordController,
):
    """Test _get_scheme works when all records have valid values"""

    # Mock this method, we test it explicitly later
    hdf5_controller._waveform_record_to_string = MagicMock(  # type: ignore
        side_effect=["%s/%s_%d.h5", "/some/path", "some_filename"]
    )

    assert hdf5_controller._get_scheme() == "/some/path/some_filename_%d.h5"


def test_hdf_capture_validate_valid_scheme(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate passes when a valid scheme is given"""
    hdf5_controller._get_scheme = MagicMock(  # type: ignore
        return_value="/valid/file%d.h5"
    )

    assert hdf5_controller._capture_validate(None, 1) is True


def test_hdf_capture_validate_new_value_zero(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate passes when new value is zero"""
    assert hdf5_controller._capture_validate(None, 0) is True


def test_hdf_capture_validate_invalid_scheme(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate fails when scheme cannot be created"""
    hdf5_controller._get_scheme = MagicMock(  # type: ignore
        side_effect=ValueError("Mocked value error")
    )

    assert hdf5_controller._capture_validate(None, 1) is False


def test_hdf_capture_validate_exception(
    hdf5_controller: _HDF5RecordController,
):
    """Test _capture_validate fails due to other exceptions"""
    hdf5_controller._get_scheme = MagicMock(  # type: ignore
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
