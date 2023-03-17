import pytest_asyncio

from pandablocks._control import BlockCompleter
from pandablocks.blocking import BlockingClient


@pytest_asyncio.fixture
def dummy_server_with_blocks(dummy_server_in_thread):
    dummy_server_in_thread.send += [
        "!PCAP 1\n!LUT 8\n!SRGATE 2\n.",
        "!INPB 1 bit_mux\n!TYPEA 5 param enum\n.",  # LUT fields
        "!TRIG_EDGE 3 param enum\n!GATE 1 bit_mux\n.",  # PCAP fields
        "!OUT 1 bit_out\n.",  # SRGATE fields
    ]
    yield dummy_server_in_thread


def test_complete_one_block(dummy_server_with_blocks):
    with BlockingClient("localhost") as client:
        completer = BlockCompleter(client)
        assert completer("PCA", 0) == "PCAP"
        assert completer.matches == ["PCAP"]
        assert completer("PCAP.", 0) == "PCAP.GATE"
        assert completer.matches == ["PCAP.GATE", "PCAP.TRIG_EDGE"]
        completer("LU", 0)
        assert completer.matches == [f"LUT{i}" for i in range(1, 9)]
        completer("LUT5.", 0)
        assert completer.matches == ["LUT5.INPB", "LUT5.TYPEA"]
        completer("LUT5.TYPEA_IN", 0)
        assert completer.matches == []
        completer("LUT5.TYPE", 0)
        assert completer.matches == ["LUT5.TYPEA"]
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
        assert completer.matches == ["*DESC.LUT", "*DESC.PCAP", "*DESC.SRGATE"]
        assert completer("*DESC.LUT.", 0) == "*DESC.LUT.INPB"
        assert completer.matches == ["*DESC.LUT.INPB", "*DESC.LUT.TYPEA"]
        assert completer("*ENUMS.LU", 0) == "*ENUMS.LUT"
        assert completer.matches == ["*ENUMS.LUT"]
