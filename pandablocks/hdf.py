import logging
import queue
import threading
from typing import Callable, Dict, List, Optional, Type

import h5py

from .asyncio import AsyncioClient
from .core import Data, DataField, EndData, FrameData, StartData


class HDFWriter(threading.Thread):
    def __init__(self, scheme: str):
        super().__init__()
        self.num = 1
        self.scheme = scheme
        self.hdf_file: Optional[h5py.File] = None
        self.datasets: List[h5py.Dataset] = []
        self.queue: queue.Queue[Data] = queue.Queue()

    def run(self):
        what_to_do: Dict[Type[Data], Callable] = {
            StartData: self.open_file,
            FrameData: self.write_frame,
            EndData: self.close_file,
        }
        while True:
            data = self.queue.get()
            if data:
                what_to_do[type(data)](data)
            else:
                break

    def stop(self):
        self.queue.put(None)

    def create_dataset(self, field: DataField):
        # Data written in a big stack, growing in that dimension
        assert self.hdf_file, "File not open yet"
        return self.hdf_file.create_dataset(
            f"/{field.name}.{field.capture}",
            dtype=field.type,
            shape=(0,),
            maxshape=(None,),
        )

    def open_file(self, data: StartData):
        file_path = self.scheme % self.num
        self.hdf_file = h5py.File(file_path, "w", libver="latest")
        self.datasets = [self.create_dataset(field) for field in data.fields]
        self.hdf_file.swmr_mode = True

    def write_frame(self, data: FrameData):
        for dataset in self.datasets:
            # Append to the end, flush when done
            column = data.data[dataset.name[1:]]
            written = dataset.shape[0]
            dataset.resize((written + column.shape[0],))
            dataset[written:] = column
            dataset.flush()

    def close_file(self, data: EndData):
        logging.info(
            f"Wrote {data.samples} samples into {self.scheme % self.num}, "
            f"end reason '{data.reason.value}'"
        )
        assert self.hdf_file, "File not open yet"
        self.hdf_file.close()
        self.hdf_file = None
        self.num += 1


async def write_hdf_files(host: str, scheme: str, num: int):
    conn = AsyncioClient(host)
    writer = HDFWriter(scheme)
    counter = 0
    writer.start()
    async for data in conn.data():
        writer.queue.put(data, block=False)
        if type(data) == EndData:
            counter += 1
            if counter == num:
                break
    writer.stop()
    writer.join()
