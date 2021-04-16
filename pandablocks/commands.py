from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, TypeVar, Union

from .responses import FieldType

T = TypeVar("T")


@dataclass
class RawResponse:
    """A helper class representing a list of lines returned from PandA

    No newlines
    """

    lines: List[str]
    is_multiline: bool = False

    @property
    def line(self) -> str:
        """Check we are not multiline and return the line"""
        assert not self.is_multiline, f"{self} has many lines"
        return self.lines[0]

    @property
    def multiline(self) -> List[str]:
        """Return the lines, processed to remove markup"""
        assert self.is_multiline, f"{self} is not multiline"
        # Remove the ! and . markup
        return [line[1:] for line in self.lines[:-1]]


class CommandException(Exception):
    """Raised if a `Command` receives a mal-formed response"""


@dataclass
class Command(Generic[T]):
    """Abstract baseclass for all `ControlConnection` commands to be inherited from"""

    def lines(self) -> List[str]:
        """Return lines that should be sent to the PandA, with no newlines in them"""
        raise NotImplementedError(self)

    def response(self, raw: RawResponse) -> T:
        """Create a response from the lines received from the PandA"""
        raise NotImplementedError(self)


@dataclass
class Get(Command[Union[str, List[str]]]):
    """Get the value of a field or star command.

    Args:
        field: The field, attribute, or star command to get

    For example::

        Get("PCAP.ACTIVE") -> "1"
        Get("SEQ1.TABLE") -> ["1048576", "0", "1000", "1000"]
        Get("*IDN") -> "PandA 1.1..."
    """

    field: str

    def lines(self) -> List[str]:
        return [f"{self.field}?"]

    def response(self, raw: RawResponse) -> Union[str, List[str]]:
        """The value that was requested as a string. If it is multiline then it
        will be a list of strings"""
        if raw.is_multiline:
            return raw.multiline
        else:
            # We got OK =value
            assert raw.line.startswith("OK =")
            return raw.line[4:]


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

    def lines(self) -> List[str]:
        if isinstance(self.value, list):
            # Multiline table with blank line to terminate
            return [f"{self.field}<"] + self.value + [""]
        else:
            return [f"{self.field}={self.value}"]

    def response(self, raw: RawResponse):
        assert raw.line == "OK"


class Arm(Command[None]):
    """Arm PCAP for an acquisition by sending ``*PCAP.ARM=``"""

    def lines(self) -> List[str]:
        return ["*PCAP.ARM="]

    def response(self, raw: RawResponse):
        assert raw.line == "OK"


class Disarm(Command[None]):
    """Disarm PCAP, stopping acquisition by sending ``*PCAP.DISARM=``"""

    def lines(self) -> List[str]:
        return ["*PCAP.DISARM="]

    def response(self, raw: RawResponse):
        assert raw.line == "OK"


class GetBlockNumbers(Command[Dict[str, int]]):
    """Get the descriptions and field lists of the requested Blocks.

    For example::

        GetBlockNumbers() -> {"LUT": 8, "PCAP": 1, ...}
    """

    def lines(self) -> List[str]:
        return ["*BLOCKS?"]

    def response(self, raw: RawResponse) -> Dict[str, int]:
        """The name and number of each block type in a dictionary,
        alphabetically ordered"""
        blocks_list = []
        for line in raw.multiline:
            block, num = line.split()
            blocks_list.append((block, int(num)))
        return dict(sorted(blocks_list))


@dataclass
class GetFields(Command[Dict[str, FieldType]]):
    """Get the fields of a block, returning a `FieldType` for each one.

    Args:
        block: The name of the block type

    For example::

        GetFields("LUT") -> {"INPA": FieldType("bit_mux"), ...}
    """

    block: str

    def lines(self) -> List[str]:
        return [f"{self.block}.*?"]

    def response(self, raw: RawResponse) -> Dict[str, FieldType]:
        """The name and `FieldType` of each field in a dictionary, ordered
        to match the definition order in the PandA"""
        unsorted: Dict[int, Tuple[str, FieldType]] = {}
        for line in raw.multiline:
            name, index, type_subtype = line.split(maxsplit=2)
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

    def lines(self) -> List[str]:
        return self.inp

    def response(self, raw: RawResponse) -> List[str]:
        """The lines that PandA responded, including the multiline markup"""
        return raw.lines
