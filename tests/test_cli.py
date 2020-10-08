from collections import deque
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from pandablocks import cli


def test_writing_fast_hdf(dummy_server_in_thread, raw_dump, tmp_path):
    dummy_server_in_thread.data = raw_dump
    cli.main(["hdf", "localhost", str(tmp_path / "%d.h5")])
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
    with patch("pandablocks.control.input", side_effect=mock_input):
        cli.main(["control", "localhost", "--no-readline"])
    captured = capsys.readouterr()
    assert captured.out == "OK =0\n!1\n!2\n!3\n!4\n.\n\n"
    assert captured.err == ""
