import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    OrderedDict,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from ._exchange import Exchange, ExchangeGenerator
from .responses import Changes, FieldInfo

# Define the public API of this module
__all__ = [
    "Command",
    "CommandException",
    "Raw",
    "Get",
    "Put",
    "Arm",
    "Disarm",
    "GetBlockInfo",
    "GetFieldInfo",
    "GetPcapBitsLabels",
    "ChangeGroup",
    "GetChanges",
    "GetState",
    "SetState",
]


T = TypeVar("T")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")


# Checks whether the server will interpret cmd as a table command: search for
# first of '?', '=', '<', if '<' found first then it's a multiline command.
MULTILINE_COMMAND = re.compile(r"^[^?=]*<")


def is_multiline_command(cmd: str):
    return MULTILINE_COMMAND.match(cmd) is not None


@dataclass
class Command(Generic[T]):
    """Abstract baseclass for all ControlConnection commands to be inherited from"""

    def execute(self) -> ExchangeGenerator[T]:
        # A generator that sends lines to the PandA, gets lines back, and returns a
        # response
        raise NotImplementedError(self)


class CommandException(Exception):
    """Raised if a `Command` receives a mal-formed response"""


# `execute_commands()` actually returns a list with length equal to the number
# of tasks passed; however, Tuple is used similar to the annotation for
# zip() because typing does not support variadic type variables.  See
# typeshed PR #1550 for discussion.
@overload
def _execute_commands(c1: Command[T]) -> ExchangeGenerator[Tuple[T]]:
    ...


@overload
def _execute_commands(
    c1: Command[T], c2: Command[T2]
) -> ExchangeGenerator[Tuple[T, T2]]:
    ...


@overload
def _execute_commands(
    c1: Command[T], c2: Command[T2], c3: Command[T3]
) -> ExchangeGenerator[Tuple[T, T2, T3]]:
    ...


@overload
def _execute_commands(
    c1: Command[T], c2: Command[T2], c3: Command[T3], c4: Command[T4]
) -> ExchangeGenerator[Tuple[T, T2, T3, T4]]:
    ...


@overload
def _execute_commands(*commands: Command[Any]) -> ExchangeGenerator[Tuple[Any, ...]]:
    ...


def _execute_commands(*commands):
    """Call the `Command.execute` method on each of the commands to produce
    some `Exchange` generators. , then
    zip together the  to produce a"""
    # If we add type annotations to this function then mypy complains:
    # Overloaded function implementation does not accept all possible arguments
    # As we want to type check this, we put the logic in _zip_with_return
    ret = yield from _zip_with_return([command.execute() for command in commands])
    return ret


def _zip_with_return(
    generators: List[ExchangeGenerator[Any]],
) -> ExchangeGenerator[Tuple[Any, ...]]:
    # Sentinel to show what generators are not yet exhausted
    pending = object()
    returns = [pending] * len(generators)
    while True:
        yields: List[Exchange] = []
        for i, gen in enumerate(generators):
            # If we haven't exhausted the generator
            if returns[i] is pending:
                try:
                    # Get the exchanges that it wants to fill in
                    exchanges = next(gen)
                except StopIteration as e:
                    # Generator is exhausted, store its return value
                    returns[i] = e.value
                else:
                    # Add the exchanges to the list
                    if isinstance(exchanges, list):
                        yields += exchanges
                    else:
                        yields.append(exchanges)
        if yields:
            # There were some Exchanges yielded, so yield them all up
            # for the Connection to fill in
            yield yields
        else:
            # All the generators are exhausted, so return the tuple of all
            # their return values
            return tuple(returns)


@dataclass
class Raw(Command[List[str]]):
    """Send a raw command

    Args:
        inp: The input lines to send

    For example::

        Raw(["PCAP.ACTIVE?"]) -> ["OK =1"]
        Raw(["SEQ1.TABLE>", "1", "1", "0", "0", ""]) -> ["OK"]
        Raw(["SEQ1.TABLE?"]) -> ["!1", "!1", "!0", "!0", "."])
    """

    inp: List[str]

    def execute(self) -> ExchangeGenerator[List[str]]:
        ex = Exchange(self.inp)
        yield ex
        return ex.received


@dataclass
class Get(Command[Union[str, List[str]]]):
    """Get the value of a field or star command.

    Args:
        field: The field, attribute, or star command to get

    For example::

        Get("PCAP.ACTIVE?") -> "1"
        Get("SEQ1.TABLE?") -> ["1048576", "0", "1000", "1000"]
        Get("*IDN") -> "PandA 1.1..."
    """

    field: str

    def execute(self) -> ExchangeGenerator[Union[str, List[str]]]:
        ex = Exchange(f"{self.field}?")
        yield ex
        if ex.is_multiline:
            return ex.multiline
        else:
            # We got OK =value
            line = ex.line
            assert line.startswith("OK ="), "Line did not start with 'OK ='"
            return line[4:]


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

    def execute(self) -> ExchangeGenerator[None]:
        if isinstance(self.value, list):
            # Multiline table with blank line to terminate
            ex = Exchange([f"{self.field}<"] + self.value + [""])
        else:
            ex = Exchange(f"{self.field}={self.value}")
        yield ex
        assert ex.line == "OK"


class Arm(Command[None]):
    """Arm PCAP for an acquisition by sending ``*PCAP.ARM=``"""

    def execute(self) -> ExchangeGenerator[None]:
        ex = Exchange("*PCAP.ARM=")
        yield ex
        assert ex.line == "OK"


class Disarm(Command[None]):
    """Disarm PCAP, stopping acquisition by sending ``*PCAP.DISARM=``"""

    def execute(self) -> ExchangeGenerator[None]:
        ex = Exchange("*PCAP.DISARM=")
        yield ex
        assert ex.line == "OK"


@dataclass
class BlockInfo:
    number: int = 0
    description: str = ""


class GetBlockInfo(Command[Dict[str, BlockInfo]]):
    """Get the name, number, and description of each block type
    in a dictionary, alphabetically ordered

    For example::

         GetBlockInfo() ->
             {
                 "LUT": BlockInfo(number=8, description="Lookup table"),
                 "PCAP": BlockInfo(number=1, description="Position capture control"),
                 ...
             }
    """

    def execute(self) -> ExchangeGenerator[Dict[str, BlockInfo]]:
        ex = Exchange("*BLOCKS?")
        yield ex

        blocks_list, commands = [], []
        for line in ex.multiline:
            block, num = line.split()
            blocks_list.append((block, int(num)))
            commands.append(Get(f"*DESC.{block}"))

        description_values = yield from _execute_commands(*commands)

        blocks_info = {
            block[0]: BlockInfo(number=block[1], description=desc)
            for block, desc in zip(blocks_list, description_values)
        }

        return OrderedDict(sorted(blocks_info.items()))


@dataclass
class GetFieldInfo(Command[Dict[str, FieldInfo]]):
    """Get the fields of a block, returning a `FieldInfo` for each one, ordered
    to match the definition order in the PandA

    Args:
        block: The name of the block type

    For example::

    GetFieldInfo("LUT") -> {
        "INPA":
            FieldInfo(type='bit_mux',
                      subtype=None,
                      description='Input A',
                      label=['TTLIN1.VAL', 'TTLIN2.VAL' ...]),
        ...}
    """

    block: str

    def execute(self) -> ExchangeGenerator[Dict[str, FieldInfo]]:
        ex = Exchange(f"{self.block}.*?")
        yield ex
        unsorted: Dict[int, Tuple[str, FieldInfo]] = {}
        for line in ex.multiline:
            name, index, type_subtype = line.split(maxsplit=2)
            unsorted[int(index)] = (name, FieldInfo(*type_subtype.split()))
        # Dict keeps insertion order, so insert in the order the server said
        fields = {name: field for _, (name, field) in sorted(unsorted.items())}

        commands: List[Get] = []
        # Map from an index in the commands list to the associated field name
        field_mapping: Dict[int, str] = {}
        field: str
        field_info: FieldInfo
        for field, field_info in fields.items():
            commands.append(Get(f"*DESC.{self.block}.{field}"))
            field_mapping[len(commands) - 1] = field

            # TODO: enum'ize these strings?
            if (
                field_info.type in ("bit_mux", "pos_mux", "ext_out")
                or field_info.subtype == "enum"
            ):
                get_str = f"*ENUMS.{self.block}.{field}"
                if field_info.type == "ext_out":
                    get_str += ".CAPTURE"
                commands.append(Get(get_str))
                field_mapping[len(commands) - 1] = field

        returned_values = yield from _execute_commands(*commands)

        # Merge the returned information back into the existing fields dictionary
        for idx, value in enumerate(returned_values):
            command: Get = commands[idx]
            field_info = fields[field_mapping[idx]]
            # TODO: Confirm that there's no way we'll get misaligned index accessing
            if command.field.startswith("*DESC"):
                field_info.description = value
            elif command.field.startswith("*ENUMS"):
                field_info.label = value

        return fields


class GetPcapBitsLabels(Command):
    """Get the labels for the bit fields in PCAP.

    For example::

        GetPcapBitsLabels() -> {"BITS0" : ["TTLIN1.VAL", "TTLIN2.VAL", ...], ...}
    """

    def execute(self) -> ExchangeGenerator[Dict[str, List[str]]]:
        ex = Exchange("PCAP.*?")
        yield ex
        bits_fields = []
        for line in ex.multiline:
            split = line.split()
            if len(split) == 4:
                field_name, _, field_type, field_subtype = split
                # TODO: enum'ize these strings?
                if field_type == "ext_out" and field_subtype == "bits":
                    bits_fields.append("PCAP.%s" % field_name)

        exchanges = [Exchange(f"{field}.BITS?") for field in bits_fields]
        yield exchanges
        bits = {field: ex.multiline for field, ex in zip(bits_fields, exchanges)}
        return bits


class ChangeGroup(Enum):
    """Which group of values to ask for ``*CHANGES`` on:
    https://pandablocks-server.readthedocs.io/en/latest/commands.html#system-commands
    """

    #: All the groups below
    ALL = ""
    #: Configuration settings
    CONFIG = ".CONFIG"
    #: Bits on the system bus
    BITS = ".BITS"
    #: Positions
    POSN = ".POSN"
    #: Polled read values
    READ = ".READ"
    #: Attributes (included capture enable flags)
    ATTR = ".ATTR"
    #: Table changes
    TABLE = ".TABLE"
    #: Table changes
    METADATA = ".METADATA"


@dataclass
class GetChanges(Command[Changes]):
    """Get a `Changes` object showing which fields have changed since the last
    time this was called

    Args:
        group: Restrict to a particular `ChangeGroup`

    For example::

        GetChanges() -> Changes(
            value={"PCAP.TRIG": "PULSE1.OUT"},
            no_value=["SEQ1.TABLE"],
            in_error=["BAD.ENUM"],
        )
    """

    group: ChangeGroup = ChangeGroup.ALL

    def execute(self) -> ExchangeGenerator[Changes]:
        ex = Exchange(f"*CHANGES{self.group.value}?")
        yield ex
        changes = Changes({}, [], [])
        for line in ex.multiline:
            if line[-1] == "<":
                changes.no_value.append(line[:-1])
            elif line.endswith("(error)"):
                changes.in_error.append(line.split(" ", 1)[0])
            else:
                field, value = line.split("=", maxsplit=1)
                changes.values[field] = value
        return changes


@dataclass
class GetState(Command[List[str]]):
    """Get the state of all the fields in a PandA that should be saved as a
    list of raw lines that could be sent with `SetState`.

    For example::

        GetState() -> [
            "SEQ1.TABLE<B"
            "234fds0SDklnmnr"
            ""
            "PCAP.TRIG=PULSE1.OUT",
        ]
    """

    def execute(self) -> ExchangeGenerator[List[str]]:
        # TODO: explain in detail how this works
        # See: references/how-it-works
        attr, config, table, metadata = yield from _execute_commands(
            GetChanges(ChangeGroup.ATTR),
            GetChanges(ChangeGroup.CONFIG),
            GetChanges(ChangeGroup.TABLE),
            GetChanges(ChangeGroup.METADATA),
        )
        # Add the single line values
        line_values = dict(**attr.values, **config.values, **metadata.values)
        state = [f"{k}={v}" for k, v in line_values.items()]
        # Get the multiline values
        multiline_keys, commands = [], []
        for field in table.no_value:
            # Get tables as base64
            multiline_keys.append(f"{field}<B")
            commands.append(Get(f"{field}.B"))
        for field in metadata.no_value:
            # Get metadata as string list
            multiline_keys.append(f"{field}<")
            commands.append(Get(f"{field}"))
        multiline_values = yield from _execute_commands(*commands)
        for k, v in zip(multiline_keys, multiline_values):
            state += [k] + v + [""]
        return state


@dataclass
class SetState(Command[None]):
    """Set the state of all the fields in a PandA

    Args:
        state: A list of raw lines as produced by `GetState`

    For example::

        SetState([
            "SEQ1.TABLE<B"
            "234fds0SDklnmnr"
            ""
            "PCAP.TRIG=PULSE1.OUT",
        ])
    """

    state: List[str]

    def execute(self) -> ExchangeGenerator[None]:
        commands: List[Raw] = []
        command_lines: List[str] = []
        for line in self.state:
            command_lines.append(line)
            first_line = len(command_lines) == 1
            if (first_line and not is_multiline_command(line)) or not line:
                # If not a multiline command
                # Or blank line at the end of a multiline command
                commands.append(Raw(command_lines))
                command_lines = []
        returns = yield from _execute_commands(*commands)
        for command, ret in zip(commands, returns):
            if ret != ["OK"]:
                logging.warning(f"command {command.inp} failed with {ret}")
