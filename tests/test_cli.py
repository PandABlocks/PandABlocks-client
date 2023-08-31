from collections import deque
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from pandablocks import cli
from pandablocks.hdf import HDFDataOverrunException
from tests.conftest import STATE_RESPONSES, STATE_SAVEFILE, DummyServer


def test_writing_fast_hdf(dummy_server_in_thread: DummyServer, raw_dump, tmp_path):
    dummy_server_in_thread.send.append("OK")
    dummy_server_in_thread.data = raw_dump
    runner = CliRunner()
    result = runner.invoke(
        cli.cli, ["hdf", "localhost", str(tmp_path / "%d.h5"), "--arm"]
    )
    assert result.exit_code == 0
    hdf_file = h5py.File(tmp_path / "1.h5", "r")
    assert list(hdf_file) == [
        "COUNTER1.OUT.Max",
        "COUNTER1.OUT.Mean",
        "COUNTER1.OUT.Min",
        "COUNTER2.OUT.Mean",
        "COUNTER3.OUT.Value",
        "PCAP.BITS2.Value",
        "PCAP.SAMPLES.Value",
        "PCAP.TS_START.Value",
    ]
    assert dummy_server_in_thread.received == ["*PCAP.ARM="]
    assert_all_data_in_hdf_file(hdf_file)


def test_writing_overrun_hdf(
    dummy_server_in_thread: DummyServer, overrun_dump, tmp_path
):
    dummy_server_in_thread.send.append("OK")
    dummy_server_in_thread.data = [overrun_dump]
    runner = CliRunner()
    result = runner.invoke(
        cli.cli, ["hdf", "localhost", str(tmp_path / "%d.h5"), "--arm"]
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, HDFDataOverrunException)
    hdf_file = h5py.File(tmp_path / "1.h5", "r")
    assert_all_data_in_hdf_file(hdf_file)


def assert_all_data_in_hdf_file(hdf_file):
    def multiples(num, offset=0):
        return pytest.approx(np.arange(1, 10001) * num + offset)

    assert hdf_file["/COUNTER1.OUT.Max"][:] == multiples(1)
    assert hdf_file["/COUNTER1.OUT.Mean"][:] == multiples(1)
    assert hdf_file["/COUNTER1.OUT.Min"][:] == multiples(1)
    assert hdf_file["/COUNTER2.OUT.Mean"][:] == multiples(2)
    assert hdf_file["/COUNTER3.OUT.Value"][:] == multiples(3)
    assert hdf_file["/PCAP.BITS2.Value"][:] == multiples(0)
    assert hdf_file["/PCAP.SAMPLES.Value"][:] == multiples(0, offset=125)
    assert hdf_file["/PCAP.TS_START.Value"][:] == multiples(2e-6, offset=7.2e-8 - 2e-6)


class MockInput:
    def __init__(self, *commands: str):
        self._commands = deque(commands)

    def __call__(self, prompt):
        assert prompt == cli.PROMPT
        try:
            return self._commands.popleft()
        except IndexError:
            raise EOFError()


def test_interactive_simple(dummy_server_in_thread, capsys):
    mock_input = MockInput("PCAP.ACTIVE?", "SEQ1.TABLE?")
    dummy_server_in_thread.send += ["OK =0", "!1\n!2\n!3\n!4\n."]
    with patch("pandablocks._control.input", side_effect=mock_input):
        runner = CliRunner()
        result = runner.invoke(cli.cli, ["control", "localhost", "--no-readline"])
        assert result.exit_code == 0
        assert result.output == "OK =0\n!1\n!2\n!3\n!4\n.\n\n"


def test_save(dummy_server_in_thread: DummyServer, tmp_path: Path):
    dummy_server_in_thread.send += STATE_RESPONSES
    runner = CliRunner()
    path = tmp_path / "saved_state"
    result = runner.invoke(cli.cli, ["save", "localhost", str(path)])
    assert result.exit_code == 0

    with path.open("r") as stream:
        results = stream.read().splitlines()

    assert results == STATE_SAVEFILE


def test_load(dummy_server_in_thread: DummyServer, tmp_path: Path):
    dummy_server_in_thread.send += ["OK"] * 10
    runner = CliRunner()
    path = tmp_path / "saved_state"

    with path.open("w") as stream:
        for line in STATE_SAVEFILE:
            stream.write(line + "\n")

    result = runner.invoke(cli.cli, ["load", "localhost", str(path)])
    assert result.exit_code == 0, result.exc_info

    assert dummy_server_in_thread.received == STATE_SAVEFILE


def test_load_tutorial(dummy_server_in_thread: DummyServer, tmp_path: Path):
    dummy_server_in_thread.send += ["OK"] * 10000
    runner = CliRunner()

    with cli.TUTORIAL.open("r") as stream:
        commands = stream.read().splitlines()

    result = runner.invoke(cli.cli, ["load", "localhost", "--tutorial"])
    assert result.exit_code == 0, result.exc_info

    assert dummy_server_in_thread.received == commands
