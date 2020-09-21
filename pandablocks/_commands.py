from dataclasses import dataclass


class Command:
    """A command passed by the user to send()"""
    pass


@dataclass
class Get(Command):
    """Get the value of a field, or star command.
    E.g.
        Get("PCAP.ACTIVE") -> Value("1")
        Get("*IDN") -> Value("PandA 1.1...")
    """
    field: str


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


@dataclass
class Set(Command):
    """Set the value of a field.
    E.g.
        Set("PCAP.TRIG", "PULSE1.OUT")
    """
    field: str
    value: str
