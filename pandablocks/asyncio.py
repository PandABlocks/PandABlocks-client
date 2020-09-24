import asyncio
from asyncio.locks import Event
from typing import Dict, Optional

from .core import Command, ControlConnection, T


class AsyncioClient:
    def __init__(self, host: str):
        self.host = host
        self.connection = ControlConnection()
        self._connected = Event()
        self.ctrl_writer: Optional[asyncio.StreamWriter] = None
        self.queues: Dict[Command, asyncio.Queue] = {}
        self.reader_task = asyncio.create_task(self._ctrl_read_forever())

    async def _ctrl_read_forever(self):
        reader, self.ctrl_writer = await asyncio.open_connection(self.host, 8888)
        self._connected.set()
        while True:
            data = await reader.read(4096)
            for command, response in self.connection.receive_data(data):
                queue = self.queues.pop(command)
                queue.put_nowait(response)

    async def connected(self):
        await self._connected.wait()

    async def close(self):
        self.reader_task.cancel()
        self.ctrl_writer.close()
        await self.ctrl_writer.wait_closed()

    async def send(self, command: Command[T]) -> T:
        assert self.ctrl_writer, "Not connected yet"
        queue: asyncio.Queue[T] = asyncio.Queue()
        self.queues[command] = queue
        bytes = self.connection.send(command)
        self.ctrl_writer.write(bytes)
        await self.ctrl_writer.drain()
        response = await queue.get()
        return response
