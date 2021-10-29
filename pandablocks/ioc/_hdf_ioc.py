import asyncio
import logging
import os
from asyncio import CancelledError
from importlib.util import find_spec
from typing import List, Optional

import numpy as np
from softioc import alarm, builder
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.asyncio import AsyncioClient
from pandablocks.hdf import (
    EndData,
    FrameData,
    Pipeline,
    StartData,
    create_default_pipeline,
    stop_pipeline,
)
from pandablocks.ioc._types import ONAM_STR, ZNAM_STR
from pandablocks.responses import EndReason, ReadyData


class HDF5RecordController:
    """Class to create and control the records that handle HDF5 processing"""

    _HDF5_PREFIX = "HDF5"

    _client: AsyncioClient

    _file_path_record: RecordWrapper
    _file_name_record: RecordWrapper
    _file_number_record: RecordWrapper
    _file_format_record: RecordWrapper
    _num_capture_record: RecordWrapper
    _flush_period_record: RecordWrapper
    _capture_control_record: RecordWrapper  # Turn capture on/off
    _status_message_record: RecordWrapper  # Reports status and error messages
    _currently_capturing_record: RecordWrapper  # If HDF5 file currently being written

    _handle_hdf5_data_task: Optional[asyncio.Task] = None

    def __init__(self, client: AsyncioClient, record_prefix: str):
        if find_spec("h5py") is None:
            logging.warning("No HDF5 support detected - skipping creating HDF5 records")
            return

        self._client = client

        path_length = os.pathconf("/", "PC_PATH_MAX")
        filename_length = os.pathconf("/", "PC_NAME_MAX")

        # Create the records, including an uppercase alias for each
        # Naming convention and settings (mostly) copied from FSCN2 HDF5 records
        file_path_record_name = self._HDF5_PREFIX + ":FilePath"
        self._file_path_record = builder.WaveformOut(
            file_path_record_name,
            length=path_length,
            FTVL="UCHAR",
            DESC="File path for HDF5 files",
            validate=self._parameter_validate,
        )
        self._file_path_record.add_alias(
            record_prefix + ":" + file_path_record_name.upper()
        )

        file_name_record_name = self._HDF5_PREFIX + ":FileName"
        self._file_name_record = builder.WaveformOut(
            file_name_record_name,
            length=filename_length,
            FTVL="UCHAR",
            DESC="File name prefix for HDF5 files",
            validate=self._parameter_validate,
        )
        self._file_name_record.add_alias(
            record_prefix + ":" + file_name_record_name.upper()
        )

        num_capture_record_name = self._HDF5_PREFIX + ":NumCapture"
        self._num_capture_record = builder.longOut(
            num_capture_record_name,
            initial_value=0,  # Infinite capture
            DESC="Number of frames to capture. 0=infinite",
            DRVL=0,
        )
        # No validate - users are allowed to change this at any time
        self._num_capture_record.add_alias(
            record_prefix + ":" + num_capture_record_name.upper()
        )

        flush_period_record_name = self._HDF5_PREFIX + ":FlushPeriod"
        self._flush_period_record = builder.aOut(
            flush_period_record_name,
            initial_value=1.0,
            DESC="Frequency that data is flushed (seconds)",
        )
        self._flush_period_record.add_alias(
            record_prefix + ":" + flush_period_record_name.upper()
        )

        capture_control_record_name = self._HDF5_PREFIX + ":Capture"
        self._capture_control_record = builder.boolOut(
            capture_control_record_name,
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            initial_value=0,  # PythonSoftIOC issue #43
            on_update=self._capture_on_update,
            validate=self._capture_validate,
            DESC="Start/stop HDF5 capture",
        )
        self._capture_control_record.add_alias(
            record_prefix + ":" + capture_control_record_name.upper()
        )

        status_message_record_name = self._HDF5_PREFIX + ":Status"
        self._status_message_record = builder.stringIn(
            status_message_record_name,
            initial_value="OK",
            DESC="Reports current status of HDF5 capture",
        )
        self._status_message_record.add_alias(
            record_prefix + ":" + status_message_record_name.upper()
        )

        currently_capturing_record_name = self._HDF5_PREFIX + ":Capturing"
        self._currently_capturing_record = builder.boolIn(
            currently_capturing_record_name,
            ZNAM=ZNAM_STR,
            ONAM=ONAM_STR,
            initial_value=0,  # PythonSoftIOC issue #43
            DESC="If HDF5 file is currently being written",
        )
        self._currently_capturing_record.add_alias(
            record_prefix + ":" + currently_capturing_record_name.upper()
        )

    def _parameter_validate(self, record: RecordWrapper, new_val) -> bool:
        """Control when values can be written to parameter records
        (file name etc.) based on capturing record's value"""
        logging.debug(f"Validating record {record.name} value {new_val}")
        if self._capture_control_record.get():
            # Currently capturing, discard parameter updates
            logging.warning(
                "Data capture in progress. Update of HDF5 "
                f"record {record.name} with new value {new_val} discarded."
            )
            return False
        return True

    async def _handle_hdf5_data(self):
        """Handles writing HDF5 data from the PandA to file, based on configuration
        in the various HDF5 records.
        This method expects to be run as an asyncio Task."""
        try:
            # Keep the start data around to compare against, for the case where a new
            # capture, and thus new StartData, is sent without Capture ever being
            # disabled
            start_data: Optional[StartData] = None
            captured_frames: int = 0
            # Only one filename - user must stop capture and set new FileName/FilePath
            # for new files
            pipeline: List[Pipeline] = create_default_pipeline(
                iter([self._get_filename()])
            )
            flush_period: float = self._flush_period_record.get()

            async for data in self._client.data(
                scaled=False, flush_period=flush_period
            ):
                logging.debug(f"Received data packet: {data}")
                if isinstance(data, ReadyData):
                    self._currently_capturing_record.set(1)
                    self._status_message_record.set("Starting capture")
                elif isinstance(data, StartData):
                    if start_data and data != start_data:
                        # PandA was disarmed, had config changed, and rearmed.
                        # Cannot process to the same file with different start data.
                        logging.error(
                            "New start data detected, differs from previous start "
                            "data for this file. Aborting HDF5 data capture."
                        )

                        self._status_message_record.set(
                            "Mismatched StartData packet for file",
                            severity=alarm.MAJOR_ALARM,
                            alarm=alarm.STATE_ALARM,
                        )
                        pipeline[0].queue.put_nowait(
                            EndData(captured_frames, EndReason.START_DATA_MISMATCH)
                        )

                        break
                    if start_data is None:
                        # Only pass StartData to pipeline if we haven't previously
                        # - if we have there will already be an in-progress HDF file
                        # that we should just append data to
                        start_data = data
                        pipeline[0].queue.put_nowait(data)

                elif isinstance(data, FrameData):
                    pipeline[0].queue.put_nowait(data)
                    captured_frames += 1
                    num_frames_to_capture: int = self._num_capture_record.get()
                    if (
                        num_frames_to_capture > 0
                        and captured_frames >= num_frames_to_capture
                    ):
                        # Reached configured capture limit, stop the file
                        logging.info(
                            f"Requested number of frames ({num_frames_to_capture}) "
                            "captured, disabling Capture."
                        )
                        self._status_message_record.set(
                            "Requested number of frames captured"
                        )
                        pipeline[0].queue.put_nowait(
                            EndData(captured_frames, EndReason.OK)
                        )

                        break
                # Ignore EndData - handle terminating capture with the Capture
                # record or when we capture the requested number of frames

        except CancelledError:
            logging.info("Capturing task cancelled, closing HDF5 file")
            self._status_message_record.set("Capturing disabled")
            # Only send EndData if we know the file was opened - could be cancelled
            # before PandA has actually send any data
            if start_data:
                pipeline[0].queue.put_nowait(EndData(captured_frames, EndReason.OK))

        except Exception:
            logging.exception("HDF5 data capture terminated due to unexpected error")
            self._status_message_record.set(
                "Capturing disabled, unexpected exception",
                severity=alarm.MAJOR_ALARM,
                alarm=alarm.STATE_ALARM,
            )
            # Only send EndData if we know the file was opened - exception could happen
            # before file was opened
            if start_data:
                pipeline[0].queue.put_nowait(
                    EndData(captured_frames, EndReason.UNKNOWN_EXCEPTION)
                )

        finally:
            logging.debug("Finishing processing HDF5 PandA data")
            stop_pipeline(pipeline)
            self._capture_control_record.set(0)
            self._currently_capturing_record.set(0)

    def _get_filename(self) -> str:
        """Create the file path for the HDF5 file from the relevant records"""

        return "/".join(
            (
                self._waveform_record_to_string(self._file_path_record),
                self._waveform_record_to_string(self._file_name_record),
            )
        )

    async def _capture_on_update(self, new_val: int) -> None:
        """Process an update to the Capture record, to start/stop recording HDF5 data"""
        logging.debug(f"Entering HDF5:Capture record on_update method, value {new_val}")
        if new_val:
            if self._handle_hdf5_data_task:
                logging.warning("Existing HDF5 capture running, cancelling it.")
                self._handle_hdf5_data_task.cancel()

            self._handle_hdf5_data_task = asyncio.create_task(self._handle_hdf5_data())
        else:
            assert self._handle_hdf5_data_task
            self._handle_hdf5_data_task.cancel()  # Abort any HDF5 file writing
            self._handle_hdf5_data_task = None

    def _capture_validate(self, record: RecordWrapper, new_val: int) -> bool:
        """Check the required records have been set before allowing Capture=1"""
        if new_val:
            try:
                self._get_filename()
            except ValueError:
                logging.exception("At least 1 required record had no value")
                return False
            except Exception:
                logging.exception("Unexpected exception creating file name")
                return False

        return True

    def _waveform_record_to_string(self, record: RecordWrapper) -> str:
        """Handle converting WaveformOut record data into python string"""
        val = record.get()
        if val is None:
            raise ValueError(f"Record {record.name} had no value when one is required.")
        return self._numpy_to_string(record.get())

    def _numpy_to_string(self, val: np.ndarray) -> str:
        """Handle converting a numpy array of dtype=uint8 to a python string"""
        assert val.dtype == np.uint8
        # numpy gives byte stream, which we decode, and then remove the trailing \0.
        # Many tools are C-based and so will add (and expect) a null trailing byte
        return val.tobytes().decode()[:-1]
