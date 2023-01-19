import asyncio
import json
import os
from typing import Dict

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from pandablocks.ioc.ioc import initialise_server
from tests.conftest import DummyServer

# Tests for the bobfile server

test_bob_file_dict: Dict[str, str] = {"TEST.bob": "<Bobfile>Contents</Bobfile>"}


@pytest.fixture
async def setup_server():
    """Adds the test server to the current event loop and creates a test client."""
    loop = asyncio.get_event_loop()
    app = initialise_server(test_bob_file_dict)
    async with TestClient(TestServer(app), loop=loop) as client:
        yield client


@pytest.mark.asyncio
async def test_get_available_files(setup_server):
    """Tests a request for the available files."""
    client = setup_server
    resp = await client.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert text == '["TEST.bob"]'


@pytest.mark.asyncio
async def test_get_file(setup_server):
    """Tests a request for a single file."""
    client = setup_server
    resp = await client.get("/TEST.bob")
    assert resp.status == 200
    text = await resp.text()
    assert text == test_bob_file_dict["TEST.bob"]


@pytest.mark.asyncio
async def test_system_bobfile_creation(
    dummy_server_system: DummyServer, subprocess_ioc
):
    """A system test for both the bobfile creation and running the server."""
    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession(loop=loop) as session:

        async with session.get("http://0.0.0.0:8080/") as response:
            bob_file_list = json.loads(await response.text())
            assert "PCAP1.bob" in bob_file_list

        async with session.get("http://0.0.0.0:8080/PCAP1.bob") as response:
            result = await response.text()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(dir_path + "/test-bobfiles/PCAP1.bob", "r") as f:
                assert result == f.read()
            with open(dir_path + "/test-bobfiles/PCAP1.bob", "w") as f:
                f.write(result)
