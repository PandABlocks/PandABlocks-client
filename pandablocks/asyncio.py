import asyncio
from collections import deque
from typing import AsyncGenerator, Deque, Dict, Optional

from .core import Command, ControlConnection, Data, DataConnection, T


class AsyncioClient:
    def __init__(self, host: str):
        self._host = host
        self._ctrl_connection = ControlConnection()
        self._ctrl_writer: Optional[asyncio.StreamWriter] = None
        self._ctrl_task: Optional[asyncio.Task] = None
        self._ctrl_queues: Dict[Command, asyncio.Queue] = {}

    async def connect(self):
        reader, self._ctrl_writer = await asyncio.open_connection(self._host, 8888)
        self._ctrl_task = asyncio.create_task(self._ctrl_read_forever(reader))

    async def _ctrl_read_forever(self, reader: asyncio.StreamReader):
        while True:
            bytes = await reader.read(4096)
            for command, response in self._ctrl_connection.receive_bytes(bytes):
                queue = self._ctrl_queues.pop(command)
                queue.put_nowait(response)

    async def close(self):
        assert self._ctrl_writer and self._ctrl_task, "Not connected yet"
        self._ctrl_task.cancel()
        self._ctrl_writer.close()
        await self._ctrl_writer.wait_closed()

    async def send(self, command: Command[T]) -> T:
        assert self._ctrl_writer, "Not connected yet"
        queue: asyncio.Queue[T] = asyncio.Queue()
        self._ctrl_queues[command] = queue
        bytes = self._ctrl_connection.send(command)
        self._ctrl_writer.write(bytes)
        await self._ctrl_writer.drain()
        response = await queue.get()
        return response

    async def data(
        self,
        scaled: bool = True,
        flush_period: float = None,
        frame_timeout: float = None,
    ) -> AsyncGenerator[Data, None]:
        """Connect to data port and yield data frames

        Args:
            scaled: Whether to scale and average data frames, reduces throughput
            flush_period: How often to flush partial data frames, None is on every
                chunk of data from the server
            frame_timeout: If no data is received for this amount of time, raise
                TimeoutError
        """
        connection = DataConnection()
        reader, writer = await asyncio.open_connection(self._host, 8889)

        data: Deque[Optional[Data]] = deque()

        async def queue_flushed_data():
            data.append(connection.flush())

        async def periodic_flush():
            if flush_period:
                while True:
                    # Every flush_period seconds flush and queue data
                    await asyncio.gather(
                        asyncio.sleep(flush_period), queue_flushed_data()
                    )

        flush_task = asyncio.create_task(periodic_flush())
        try:
            writer.write(connection.connect(scaled))
            await writer.drain()
            while True:
                bytes = await asyncio.wait_for(reader.read(4096), frame_timeout)
                for d in connection.receive_bytes(bytes):
                    data.append(d)
                if flush_period is None:
                    # No flush task, do it here
                    data.append(connection.flush())
                while data:
                    yd = data.popleft()
                    if yd:
                        yield yd
        finally:
            flush_task.cancel()
            writer.close()
            await writer.wait_closed()
