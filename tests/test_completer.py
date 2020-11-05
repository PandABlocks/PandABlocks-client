import pytest

from pandablocks._control import BlockCompleter
from pandablocks.blocking import BlockingClient


@pytest.fixture
def dummy_server_with_blocks(dummy_server_in_thread):
    dummy_server_in_thread.send += [
        "!PCAP 1\n!LUT 8\n.",
        "!TYPEA 5 param enum\n!TYPEA_INP 1 bit_mux\n.",
        "!TS_START 6 ext_out timestamp\n!ACTIVE 7 bit_out\n.",
    ]
    yield dummy_server_in_thread


def test_complete_one_block(dummy_server_with_blocks):
    with BlockingClient("localhost") as client:
        completer = BlockCompleter(client)
        assert completer("PCA", 0) == "PCAP"
        assert completer.matches == ["PCAP"]
        assert completer("PCAP.", 0) == "PCAP.TS_START"
        assert completer.matches == ["PCAP.TS_START", "PCAP.ACTIVE"]
        completer("LU", 0)
        assert completer.matches == [f"LUT{i}" for i in range(1, 9)]
        completer("LUT5.", 0)
        assert completer.matches == ["LUT5.TYPEA_INP", "LUT5.TYPEA"]
        completer("LUT5.TYPEA_IN", 0)
        assert completer.matches == ["LUT5.TYPEA_INP"]
        assert completer("LUT5.TYPEA_INP?", 0) is None


def test_complete_stars(dummy_server_with_blocks):
    with BlockingClient("localhost") as client:
        completer = BlockCompleter(client)
        assert completer("*PCA", 0) == "*PCAP.STATUS?"
        assert completer.matches == [
            "*PCAP.STATUS?",
            "*PCAP.CAPTURED?",
            "*PCAP.COMPLETION?",
            "*PCAP.ARM=",
            "*PCAP.DISARM=",
        ]
        assert completer("*DE", 0) == "*DESC.LUT"
        assert completer.matches == ["*DESC.LUT", "*DESC.PCAP"]
        assert completer("*DESC.LUT.", 0) == "*DESC.LUT.TYPEA_INP"
        assert completer.matches == ["*DESC.LUT.TYPEA_INP", "*DESC.LUT.TYPEA"]
        assert completer("*ENUMS.LU", 0) == "*ENUMS.LUT"
        assert completer.matches == ["*ENUMS.LUT"]
