from dataclasses import dataclass
from string import digits
from typing import Dict, List, Optional

from pandablocks.blocking import BlockingClient
from pandablocks.commands import FieldType, GetBlockNumbers, GetFields
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


def _get_user_input(prompt) -> List[str]:
    lines = [input(prompt)]
    if _is_table_command(lines[0]):
        while lines[-1]:
            lines.append(input(prompt))
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
                print(f"!{line.decode()}")
            print(".")
        else:
            print(lines.decode())


STATIC_STAR_COMMANDS = [
    "*IDN?",  # Device identification.
    "*WHO?",  # List connected clients.
    "*BLOCKS?",  # List device blocks.
    "*CAPTURE?",  # Report data capture words.
    "*POSITIONS?",  # Enumerate possible capture positions.
    "*VERBOSE=",  # Control command logging.
    "*SAVESTATE=",  # Triggers immediate save to file of the persistence file state.
]

for suff in ("STATUS?", "CAPTURED?", "COMPLETION?", "ARM=", "DISARM="):
    STATIC_STAR_COMMANDS.append(f"*PCAP.{suff}")  # Position capture status/action
for suff in ("", ".CONFIG", ".BITS", ".POSN", ".READ", ".ATTR", ".TABLE"):
    STATIC_STAR_COMMANDS.append(f"*CHANGES{suff}?")  # Report changes to values
    STATIC_STAR_COMMANDS.append(f"*CHANGES{suff}=")  # Reset reported changes


def text_matches(t1, t2):
    return t1.startswith(t2) or t2.startswith(t1)


class BlockCompleter:
    def __init__(self, client: BlockingClient):
        self.matches: List[str] = []
        self._client = client
        self._blocks = self._client.send(GetBlockNumbers(), timeout=2)
        self._fields = self._get_fields(list(self._blocks))

    def _get_fields(self, blocks: List[str]) -> Dict[str, Dict[str, FieldType]]:
        fields = self._client.send([GetFields(block) for block in blocks], timeout=2)
        return dict(zip(blocks, fields))

    def _with_suffixes(self, block: str, numbers: bool) -> List[str]:
        num = self._blocks[block]
        if numbers and num > 1:
            return [f"{block}{i}" for i in range(1, num + 1)]
        else:
            return [block]

    def _block_field_matches(self, text: str, prefix="") -> List[str]:
        matches = []
        text = text[len(prefix) :]
        split = text.split(".", maxsplit=1)
        if len(split) == 1:
            # No dot, text is a partial block name
            for block in self._blocks:
                if text_matches(text, block):
                    for bn in self._with_suffixes(block, numbers=not prefix):
                        if text_matches(text, bn):
                            matches.append(prefix + bn)
        else:
            bn = split[0]
            for field in self._fields[bn.rstrip(digits)]:
                bnf = f"{bn}.{field}"
                if text_matches(text, bnf):
                    matches.append(prefix + bnf)
        return matches

    def __call__(self, text: str, state: int) -> Optional[str]:
        """Return the next possible completion for 'text'.
        This is called successively with state == 0, 1, 2, ... until it
        returns None.  The completion should begin with 'text'.
        """
        if state == 0:
            # Create match cache
            if text.startswith("*"):
                # Star command completions
                self.matches = [
                    x for x in STATIC_STAR_COMMANDS if text_matches(text, x)
                ]
                if not self.matches:
                    # *DESC and *ENUMS have block names in them
                    if text_matches(text, "*DESC"):
                        self.matches = self._block_field_matches(text, "*DESC.")
                    elif text_matches(text, "*ENUMS"):
                        self.matches = self._block_field_matches(text, "*ENUMS.")
            else:
                # Check for block names
                self.matches = self._block_field_matches(text)
            if len(self.matches) == 1 and len(text) > len(self.matches[0]):
                # We only partially match, if the text is longer then don't complete
                self.matches = []
        # Return correct cached match
        try:
            return self.matches[state]
        except IndexError:
            return None


def set_completer(completer):
    try:
        # Give command completion etc.
        import readline
    except ImportError:
        # Don't need it on windows
        pass
    else:
        readline.set_completer(completer)
        # Apple does not ship GNU readline with OS X.
        # It does ship BSD libedit which includes a readline compatibility interface.
        # https://stackoverflow.com/a/7116997
        if "libedit" in readline.__doc__:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        # Only complete at the start of the line, not for values
        readline.set_completer_delims("")


def control(host: str, prompt: str, readline=True):
    client = BlockingClient(host)
    try:
        if readline:
            # Complete Block names when tab key pressed
            set_completer(BlockCompleter(client))
        while True:
            client.send(Interact(_get_user_input(prompt)))
    except (EOFError, KeyboardInterrupt):
        # End with a newline
        print()
    finally:
        client.close()
