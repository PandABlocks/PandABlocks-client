from dataclasses import dataclass
from typing import List, Optional

from pandablocks.blocking import BlockingClient
from pandablocks.core import Command, Lines


# Checks whether the server will interpret cmd as a table command: search for
# first of '?', '=', '<', if '<' found first then it's a table command.
def _is_table_command(cmd):
    for ch in cmd:
        if ch in "?=":
            return False
        if ch == "<":
            return True
    return False


def _get_user_input() -> List[str]:
    lines = [input("> ")]
    if _is_table_command(lines[0]):
        while lines[-1]:
            lines.append(input("> "))
    return lines


@dataclass(frozen=True)
class Interact(Command[Lines]):
    """Send an interactive command
    E.g.
        Interact(["PCAP.ACTIVE"]) -> print "< OK =1"
        Interact(["SEQ1.TABLE>", "1", "1", "0", "0"]) -> print "< OK"
    """

    inp: List[str]

    def lines(self) -> Lines:
        return [line.encode() for line in self.inp]

    def response(self, lines: Lines):
        if isinstance(lines, List):
            # Add the multiline markup back in...
            for line in lines:
                print(f"! {line.decode()}")
            print(".")
        else:
            print(lines.decode())


class BlockCompleter:
    def __init__(self, client: BlockingClient):
        self._client = client

    def __call__(self, text: str, state: int) -> Optional[str]:
        """Return the next possible completion for 'text'.
        This is called successively with state == 0, 1, 2, ... until it
        returns None.  The completion should begin with 'text'.
        """


def set_completer(completer):
    try:
        # Give command completion etc.
        import readline  # noqa
    except ImportError:
        # Don't need it on windows
        pass
    else:
        readline.set_completer(completer)
        # ... when tab key pressed
        readline.parse_and_bind("tab: complete")


def control(host: str, readline=True):
    client = BlockingClient(host)
    try:
        if readline:
            # Complete Block names when tab key pressed
            set_completer(BlockCompleter(client))
        while True:
            client.send(Interact(_get_user_input()))
    except (EOFError, KeyboardInterrupt):
        # End with a newline
        print()
    finally:
        client.close()
