from dataclasses import dataclass
from typing import List, Union

from .core import Command, Lines


@dataclass(frozen=True)
class Get(Command[Lines]):
    """Get the value of a field, or star command.
    E.g.
        Get("PCAP.ACTIVE") -> b"1"
        Get("SEQ1.TABLE") -> [b"1048576", b"0", b"1000", b"1000"]
        Get("*IDN") -> b"PandA 1.1..."
    """

    field: str
    strip_ok: bool = True

    def lines(self) -> Lines:
        return f"{self.field}?".encode()

    def response(self, lines: Lines) -> Lines:
        if not isinstance(lines, list) and self.strip_ok:
            # We got OK =value
            self.ok_if(lines.startswith(b"OK ="), lines)
            return lines[4:]
        else:
            return lines


@dataclass(frozen=True)
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


class GetBlocksData(Command):
    """Get the descriptions and field lists of the requested Blocks
    E.g.
        GetBlocksData() -> BlocksData()
    """


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
