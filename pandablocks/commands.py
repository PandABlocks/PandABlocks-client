from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, TypeVar, Union

from .responses import FieldType

T = TypeVar("T")
# One or more lines to send
Lines = Union[bytes, List[bytes]]


class Command(Generic[T]):
    def lines(self) -> Lines:
        """Return lines that should be sent, with no newlines in them"""
        raise NotImplementedError(self)

    def response(self, lines: Lines) -> T:
        """Create a response from the lines received from the PandA"""
        raise NotImplementedError(self)

    def ok_if(self, ok, lines: Lines):
        """If not ok then raise a suitable error message"""
        if not ok:
            raise ValueError("Bad response to command {self}: '{lines}'")


@dataclass
class Get(Command[Lines]):
    """Get the value of a field, or star command.
    E.g.
        Get("PCAP.ACTIVE") -> b"1"
        Get("SEQ1.TABLE") -> [b"1048576", b"0", b"1000", b"1000"]
        Get("*IDN") -> b"PandA 1.1..."
    """

    field: str

    def lines(self) -> Lines:
        return f"{self.field}?".encode()

    def response(self, lines: Lines) -> Lines:
        if not isinstance(lines, list):
            # We got OK =value
            self.ok_if(lines.startswith(b"OK ="), lines)
            return lines[4:]
        else:
            return lines


@dataclass
class Put(Command[None]):
    """Put the value of a field.
    E.g.
        Put("PCAP.TRIG", "PULSE1.OUT")
        Put("SEQ1.TABLE", ["1048576", "0", "1000", "1000"])
    """

    field: str
    value: Union[str, List[str]]

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


class GetBlockNumbers(Command[Dict[str, int]]):
    """Get the descriptions and field lists of the requested Blocks
    E.g.
        GetBlockNumbers() -> {"PCAP": 1, "LUT": 8, ...}
    """

    def lines(self) -> Lines:
        return b"*BLOCKS?"

    def response(self, lines: Lines) -> Dict[str, int]:
        blocks = {}
        assert isinstance(lines, list), f"Expected list of Blocks, got {lines!r}"
        for line in lines:
            block, num = line.split()
            blocks[block.decode()] = int(num)
        return {block: num for block, num in sorted(blocks.items())}


@dataclass
class GetFields(Command[Dict[str, FieldType]]):
    """Get the fields of a block
    E.g.
        GetFields("LUT") -> {"INPA": FieldType("bit_mux"), ...}
    """

    block: str

    def lines(self) -> Lines:
        return f"{self.block}.*?".encode()

    def response(self, lines: Lines) -> Dict[str, FieldType]:
        unsorted: Dict[int, Tuple[str, FieldType]] = {}
        assert isinstance(lines, list), f"Expected list of Fields, got {lines!r}"
        for line in lines:
            name, index, type_subtype = line.decode().split(maxsplit=2)
            unsorted[int(index)] = (name, FieldType(*type_subtype.split()))
        # Dict keeps insertion order, so insert in the order the server said
        fields = {name: field for _, (name, field) in sorted(unsorted.items())}
        return fields


class GetPcapBitsLabels(Command):
    """Get the labels for the bit fields in PCAP
    E.g.
        GetPcapBitsLabels() -> PcapBitsLabels()
    """


class GetChanges(Command):
    """Get the changes since the last time this was called
    E.g.
        GetChanges() -> Changes()
    """
