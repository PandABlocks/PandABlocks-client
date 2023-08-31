import logging
import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Type

import h5py
import numpy as np

from pandablocks.commands import Arm

from .asyncio import AsyncioClient
from .connections import SAMPLES_FIELD
from .responses import EndData, EndReason, FieldCapture, FrameData, ReadyData, StartData

# Define the public API of this module
__all__ = [
    "Pipeline",
    "HDFWriter",
    "FrameProcessor",
    "HDFDataOverrunException",
    "create_pipeline",
    "create_default_pipeline",
    "stop_pipeline",
    "write_hdf_files",
]


class HDFDataOverrunException(Exception):
    """Raised if `DATA_OVERRUN` occurs while receiving data for HDF file"""


class Pipeline(threading.Thread):
    """Helper class that runs a pipeline consumer process in its own thread"""

    #: Subclasses should create this dictionary with handlers for each data
    #: type, returning transformed data that should be passed downstream
    what_to_do: Dict[Type, Callable]
    downstream: Optional["Pipeline"] = None

    def __init__(self):
        super().__init__()
        self.queue: queue.Queue[Any] = queue.Queue()  # type: ignore

    def run(self):
        while True:
            data = self.queue.get()
            if data is None:
                # stop() called below
                break
            else:
                func = self.what_to_do.get(type(data), None)
                if func:
                    # If we have a handler, use it to transform the data
                    data = func(data)
                if self.downstream:
                    # Pass the (possibly transformed) data downstream
                    self.downstream.queue.put_nowait(data)

    def stop(self):
        """Stop the processing after the current queue has been emptied"""
        self.queue.put(None)


class HDFWriter(Pipeline):
    """Write an HDF file per data collection. Each field will be
    written in a 1D dataset ``/<field.name>.<field.capture>``.

    Args:
        file_names: Iterator of file names. Must be full file paths. Will be called once
            per file created.
    """

    def __init__(self, file_names: Iterator[str]):
        super().__init__()
        self.file_names = file_names
        self.hdf_file: Optional[h5py.File] = None
        self.datasets: List[h5py.Dataset] = []
        self.what_to_do = {
            StartData: self.open_file,
            list: self.write_frame,
            EndData: self.close_file,
        }

    def create_dataset(self, field: FieldCapture, raw: bool):
        # Data written in a big stack, growing in that dimension
        assert self.hdf_file, "File not open yet"
        if raw and (field.capture == "Mean" or field.scale != 1 or field.offset != 0):
            # Processor outputs a float
            dtype = np.dtype("float64")
        else:
            # No processor, datatype passed through
            dtype = field.type
        return self.hdf_file.create_dataset(
            f"/{field.name}.{field.capture}",
            dtype=dtype,
            shape=(0,),
            maxshape=(None,),
        )

    def open_file(self, data: StartData):
        try:
            self.file_path = next(self.file_names)
        except IndexError:
            logging.exception(
                "Not enough file names available when opening new HDF5 file"
            )
            raise
        self.hdf_file = h5py.File(self.file_path, "w", libver="latest")
        raw = data.process == "Raw"
        self.datasets = [self.create_dataset(field, raw) for field in data.fields]
        self.hdf_file.swmr_mode = True
        logging.info(
            f"Opened '{self.file_path}' with {data.sample_bytes} byte samples "
            f"stored in {len(self.datasets)} datasets"
        )

    def write_frame(self, data: List[np.ndarray]):
        for dataset, column in zip(self.datasets, data):
            # Append to the end, flush when done
            written = dataset.shape[0]
            dataset.resize((written + column.shape[0],))
            dataset[written:] = column
            dataset.flush()

    def close_file(self, data: EndData):
        assert self.hdf_file, "File not open yet"
        self.hdf_file.close()
        self.hdf_file = None
        logging.info(
            f"Closed '{self.file_path}' after writing {data.samples} "
            f"samples. End reason is '{data.reason.value}'"
        )
        self.file_path = ""


class FrameProcessor(Pipeline):
    """Scale field data according to the information in the StartData"""

    def __init__(self) -> None:
        super().__init__()
        self.processors: List[Callable] = []
        self.what_to_do = {
            StartData: self.create_processors,
            FrameData: self.scale_data,
        }

    def create_processor(self, field: FieldCapture, raw: bool):
        column_name = f"{field.name}.{field.capture}"
        if raw and field.capture == "Mean":
            return (
                lambda data: data[column_name] * field.scale / data[SAMPLES_FIELD]
                + field.offset
            )
        elif raw and (field.scale != 1 or field.offset != 0):
            return lambda data: data[column_name] * field.scale + field.offset
        else:
            return lambda data: data[column_name]

    def create_processors(self, data: StartData) -> StartData:
        raw = data.process == "Raw"
        self.processors = [self.create_processor(field, raw) for field in data.fields]
        return data

    def scale_data(self, data: FrameData) -> List[np.ndarray]:
        return [process(data.data) for process in self.processors]


def create_default_pipeline(
    file_names: Iterator[str],
) -> List[Pipeline]:
    """Create the default processing pipeline consisting of one `FrameProcessor` and
    one `HDFWriter`. See `create_pipeline` for more details.

    Args:
        file_names: Iterator of file names. Must be full file paths. Will be called once
            per file created. As required by `HDFWriter`."""
    return create_pipeline(FrameProcessor(), HDFWriter(file_names))


def create_pipeline(*elements: Pipeline) -> List[Pipeline]:
    """Create a pipeline of elements, wiring them and starting them before
    returning them"""
    pipeline: List[Pipeline] = []
    for element in elements:
        if pipeline:
            pipeline[-1].downstream = element
        pipeline.append(element)
        element.start()
    return pipeline


def stop_pipeline(pipeline: List[Pipeline]):
    """Stop and join each element of the pipeline"""
    for element in pipeline:
        # Note that we stop and join each element in turn.
        # This ensures all data is flushed all the way down
        # even if there is lots left in a queue
        element.stop()
        element.join()


async def write_hdf_files(
    client: AsyncioClient,
    file_names: Iterator[str],
    num: int = 1,
    arm: bool = False,
    flush_period: float = 1,
):
    """Connect to host PandA data port, and write num acquisitions
    to HDF file according to scheme

    Args:
        client: The `AsyncioClient` to use for communications
        file_names: Iterator of file names. Must be full file paths. Will be called once
            per file created.
        num: The number of acquisitions to store in separate files. 0 = Infinite capture
        arm: Whether to arm PCAP at the start, and after each successful acquisition

    Raises:
        HDFDataOverrunException: if there is a data overrun.
    """
    counter = 0
    end_data = None
    pipeline = create_default_pipeline(file_names)
    try:
        async for data in client.data(scaled=False, flush_period=flush_period):
            pipeline[0].queue.put_nowait(data)
            if type(data) == EndData:
                end_data = data
                counter += 1
                if counter == num:
                    # We produced the right number of frames
                    break
            if type(data) in (ReadyData, EndData) and arm:
                # Told to arm at the beginning, and after each acquisition ends
                await client.send(Arm())
        if end_data and end_data.reason == EndReason.DATA_OVERRUN:
            raise HDFDataOverrunException(
                "Data overrun - streaming aborted! Last frame may be corrupt."
            )
    finally:
        stop_pipeline(pipeline)
