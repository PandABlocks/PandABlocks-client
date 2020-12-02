from pandablocks.state import State
from tests.conftest import DummyServer

responses = [
    "!Attr=0\n.",
    "!Config=0\n.",
    "!Table<\n.",
    "!tabledata\n.",
    "!SingleLineMeta1=0\n!MultiLineMeta1<\n!SingleLineMeta2=0\n!MultiLineMeta2<\n.",
    "!multimetadata1\n.",
    "!multimetadata2\n.",
]

savefile = [
    "Attr=0",
    "Config=0",
    "Table<B",
    "tabledata",
    "",
    "SingleLineMeta1=0",
    "MultiLineMeta1<",
    "multimetadata1",
    "",
    "SingleLineMeta2=0",
    "MultiLineMeta2<",
    "multimetadata2",
    "",
]


def test_save(dummy_server_in_thread: DummyServer):
    dummy_server_in_thread.send += responses
    state = State("localhost")
    result = state.save()

    assert result == savefile


def test_load(dummy_server_in_thread: DummyServer):
    dummy_server_in_thread.send += ["OK"] * 10
    state = State("localhost")
    state.load(savefile)

    count = len(dummy_server_in_thread.send)
    print(count)
