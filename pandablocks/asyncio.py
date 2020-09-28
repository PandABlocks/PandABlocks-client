import asyncio
from asyncio.streams import StreamReader
from typing import AsyncGenerator, Dict, Optional

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

    async def _ctrl_read_forever(self, reader: StreamReader):
        while True:
            bytes = await reader.read(4096)
            for command, response in self._ctrl_connection.receive_data(bytes):
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

    async def data(self, frame_timeout: int = None) -> AsyncGenerator[Data, None]:
        reader, writer = await asyncio.open_connection(self._host, 8889)
        try:
            connection = DataConnection()
            writer.write(connection.connect())
            await writer.drain()
            while True:
                bytes = await asyncio.wait_for(reader.read(4096), frame_timeout)
                for data in connection.receive_data(bytes):
                    yield data
        finally:
            writer.close()
            await writer.wait_closed()
