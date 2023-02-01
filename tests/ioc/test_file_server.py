import asyncio
import json
import os
from typing import AsyncGenerator, Dict

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from pandablocks.ioc.ioc import BOB_FILE_HOST, BOB_FILE_PORT, initialise_server
from tests.conftest import DummyServer

# Tests for the bobfile server
TEST_FILE_NAME = "TEST.bob"
TEST_FILE_CONTENTS = "<Bobfile>Contents</Bobfile>"

# These constants correspond to a real file
PCAP1_FILE_NAME = "PCAP1.bob"
PCAP1_FILE_DIRECTORY = "test-bobfiles"


test_bob_file_dict: Dict[str, str] = {TEST_FILE_NAME: TEST_FILE_CONTENTS}


@pytest.fixture
async def setup_server() -> AsyncGenerator:
    """Adds the test server to the current event loop and creates a test client."""
    loop = asyncio.get_event_loop()
    app = initialise_server(test_bob_file_dict)
    async with TestClient(TestServer(app), loop=loop) as client:
        yield client


@pytest.mark.asyncio
async def test_get_available_files(setup_server: TestClient):
    """Tests a request for the available files."""
    client = setup_server
    resp = await client.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert text == f'["{TEST_FILE_NAME}"]'


@pytest.mark.asyncio
async def test_get_file(setup_server: TestClient):
    """Tests a request for a single file."""
    client = setup_server
    resp = await client.get(f"/{TEST_FILE_NAME}")
    assert resp.status == 200
    text = await resp.text()
    assert text == TEST_FILE_CONTENTS


@pytest.mark.asyncio
async def test_system_bobfile_creation(
    dummy_server_system: DummyServer, subprocess_ioc
):
    """A system test for both the bobfile creation and running the server."""
    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession(loop=loop) as session:

        async with session.get(f"http://{BOB_FILE_HOST}:{BOB_FILE_PORT}") as response:
            bob_file_list = json.loads(await response.text())
            assert bob_file_list == ["PCAP1.bob", "SEQ1.bob", "PandA.bob"]

        async with session.get(
            f"http://{BOB_FILE_HOST}:{BOB_FILE_PORT}/{PCAP1_FILE_NAME}"
        ) as response:
            result = await response.text()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(
                dir_path + f"/{PCAP1_FILE_DIRECTORY}/{PCAP1_FILE_NAME}", "r"
            ) as f:
                assert result == f.read()
