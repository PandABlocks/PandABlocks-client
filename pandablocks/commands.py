from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, TypeVar, Union

from .responses import FieldType

T = TypeVar("T")
# One or more lines to send
Lines = Union[bytes, List[bytes]]


class CommandException(Exception):
    """Raised if a `Command` receives a mal-formed response"""


@dataclass
class Command(Generic[T]):
    """Abstract baseclass for all `ControlConnection` commands to be inherited from"""

    def lines(self) -> Lines:
        """Return lines that should be sent to the PandA, with no newlines in them"""
        raise NotImplementedError(self)

    def response(self, lines: Lines) -> T:
        """Create a response from the lines received from the PandA"""
        raise NotImplementedError(self)

    def ok_if(self, ok, lines: Lines):
        """If not ok then raise a suitable `CommandException`"""
        if not ok:
            msg = f"{self} ->"
            if isinstance(lines, list):
                for line in lines:
                    msg += "\n    " + line.decode()
            else:
                msg += " " + lines.decode()
            raise CommandException(msg)


@dataclass
class Get(Command[Lines]):
    """Get the value of a field or star command.

    Args:
        field: The field, attribute, or star command to get

    For example::

        Get("PCAP.ACTIVE") -> b"1"
        Get("SEQ1.TABLE") -> [b"1048576", b"0", b"1000", b"1000"]
        Get("*IDN") -> b"PandA 1.1..."
    """

    field: str

    def lines(self) -> Lines:
        return f"{self.field}?".encode()

    def response(self, lines: Lines) -> Lines:
        """The value that was requested as a byte string. If it is multiline
        then it will be a list of byte strings"""
        if not isinstance(lines, list):
            # We got OK =value
            self.ok_if(lines.startswith(b"OK ="), lines)
            return lines[4:]
        else:
            return lines


@dataclass
class Put(Command[None]):
    """Put the value of a field.

    Args:
        field: The field, attribute, or star command to put
        value: The value, possibly multiline, to put

    For example::

        Put("PCAP.TRIG", "PULSE1.OUT")
        Put("SEQ1.TABLE", ["1048576", "0", "1000", "1000"])
    """

    field: str
    value: Union[str, List[str]] = ""

    def lines(self) -> Lines:
        if isinstance(self.value, list):
            # Multiline table with blank line to terminate
            return (
                [f"{self.field}<".encode()] + [v.encode() for v in self.value] + [b""]
            )
        else:
            return f"{self.field}={self.value}".encode()

    def response(self, lines: Lines):
        self.ok_if(lines == b"OK", lines)


class Arm(Command[None]):
    """Arm PCAP for an acquisition by sending ``*PCAP.ARM=``"""

    def lines(self) -> Lines:
        return b"*PCAP.ARM="

    def response(self, lines: Lines):
        self.ok_if(lines == b"OK", lines)


class GetBlockNumbers(Command[Dict[str, int]]):
    """Get the descriptions and field lists of the requested Blocks.

    For example::

        GetBlockNumbers() -> {"LUT": 8, "PCAP": 1, ...}
    """

    def lines(self) -> Lines:
        return b"*BLOCKS?"

    def response(self, lines: Lines) -> Dict[str, int]:
        """The name and number of each block type in a dictionary,
        alphabetically ordered"""
        blocks = {}
        assert isinstance(lines, list), f"Expected list of Blocks, got {lines!r}"
        for line in lines:
            block, num = line.split()
            blocks[block.decode()] = int(num)
        return {block: num for block, num in sorted(blocks.items())}


@dataclass
class GetFields(Command[Dict[str, FieldType]]):
    """Get the fields of a block, returning a `FieldType` for each one.

    Args:
        block: The name of the block type

    For example::

        GetFields("LUT") -> {"INPA": FieldType("bit_mux"), ...}
    """

    block: str

    def lines(self) -> Lines:
        return f"{self.block}.*?".encode()

    def response(self, lines: Lines) -> Dict[str, FieldType]:
        """The name and `FieldType` of each field in a dictionary, ordered
        to match the definition order in the PandA"""
        unsorted: Dict[int, Tuple[str, FieldType]] = {}
        assert isinstance(lines, list), f"Expected list of Fields, got {lines!r}"
        for line in lines:
            name, index, type_subtype = line.decode().split(maxsplit=2)
            unsorted[int(index)] = (name, FieldType(*type_subtype.split()))
        # Dict keeps insertion order, so insert in the order the server said
        fields = {name: field for _, (name, field) in sorted(unsorted.items())}
        return fields


class GetPcapBitsLabels(Command):
    """Get the labels for the bit fields in PCAP.

    For example::

        GetPcapBitsLabels() -> PcapBitsLabels()
    """


class GetChanges(Command):
    """Get the changes since the last time this was called.

    For example::

        GetChanges() -> Changes()
    """


# Checks whether the server will interpret cmd as a table command: search for
# first of '?', '=', '<', if '<' found first then it's a multiline command.
def is_multiline_command(cmd: str):
    for ch in cmd:
        if ch in "?=":
            return False
        if ch == "<":
            return True
    return False


@dataclass
class Raw(Command[List[str]]):
    """Send a raw command

    Args:
        inp: The input lines to send

    For example::

        Raw(["PCAP.ACTIVE?"]) -> ["OK =1"]
        Raw(["SEQ1.TABLE>", "1", "1", "0", "0"]) -> ["OK"]
        Raw(["SEQ1.TABLE?"]) -> ["!1", "!1", "!0", "!0", "."])
    """

    inp: List[str]

    def lines(self) -> Lines:
        return [line.encode() for line in self.inp]

    def response(self, lines: Lines) -> List[str]:
        """The lines that PandA responded, including the multiline markup"""
        if isinstance(lines, List):
            # Add the multiline markup back in...
            return [f"!{line.decode()}" for line in lines] + ["."]
        else:
            return [lines.decode()]
