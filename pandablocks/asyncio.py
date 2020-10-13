import asyncio
from collections import deque
from typing import AsyncGenerator, Deque, Dict, Optional

from .commands import Command, T
from .connections import ControlConnection, DataConnection
from .responses import Data


class AsyncioClient:
    """Asyncio implementation of a PandABlocks client.
    For example::

        client = AsyncioClient("hostname-or-ip")
        await client.connect_control()
        resp1, resp2 = asyncio.gather(client.send(cmd1), client.send(cmd2))
        for data in client.data():
            handle(data)
        await client.close_control()
    """

    def __init__(self, host: str):
        self._host = host
        self._ctrl_connection = ControlConnection()
        self._ctrl_writer: Optional[asyncio.StreamWriter] = None
        self._ctrl_task: Optional[asyncio.Task] = None
        self._ctrl_queues: Dict[int, asyncio.Queue] = {}

    async def connect_control(self):
        """Connect to the control port, and be ready to handle commands. Not needed
        if only the data connection is needed"""
        reader, self._ctrl_writer = await asyncio.open_connection(self._host, 8888)
        self._ctrl_task = asyncio.create_task(self._ctrl_read_forever(reader))

    async def _ctrl_read_forever(self, reader: asyncio.StreamReader):
        while True:
            bytes = await reader.read(4096)
            for command, response in self._ctrl_connection.receive_bytes(bytes):
                queue = self._ctrl_queues.pop(id(command))
                queue.put_nowait(response)

    async def close_control(self):
        """Close control connection and wait for completion"""
        assert self._ctrl_writer and self._ctrl_task, "Control port not connected yet"
        self._ctrl_task.cancel()
        self._ctrl_writer.close()
        await self._ctrl_writer.wait_closed()

    async def send(self, command: Command[T]) -> T:
        """Send a command to control port of the PandA, returning its response.
        Requires `connect_control` to have been called first.

        Args:
            command: The command to send
        """
        assert self._ctrl_writer, "Control port not connected yet"
        queue: asyncio.Queue[T] = asyncio.Queue()
        # Need to use the id as non-frozen dataclasses don't hash
        self._ctrl_queues[id(command)] = queue
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
                `asyncio.TimeoutError`
        """
        connection = DataConnection()
        reader, writer = await asyncio.open_connection(self._host, 8889)

        data: Deque[Data] = deque()

        async def queue_flushed_data():
            data.extend(connection.flush())

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
            # bool(True) instead of True so IDE sees finally block is reachable
            while bool(True):
                bytes = await asyncio.wait_for(reader.read(4096), frame_timeout)
                for d in connection.receive_bytes(bytes):
                    data.append(d)
                if flush_period is None:
                    # No flush task, do it here
                    data.extend(connection.flush())
                while data:
                    yield data.popleft()
        finally:
            flush_task.cancel()
            writer.close()
            await writer.wait_closed()
