import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from ._exchange import Exchange, ExchangeGenerator
from .responses import (
    BitMuxFieldInfo,
    BitOutFieldInfo,
    BlockInfo,
    Changes,
    EnumFieldInfo,
    ExtOutBitsFieldInfo,
    ExtOutFieldInfo,
    FieldInfo,
    PosMuxFieldInfo,
    PosOutFieldInfo,
    ScalarFieldInfo,
    SubtypeTimeFieldInfo,
    TableFieldDetails,
    TableFieldInfo,
    TimeFieldInfo,
    UintFieldInfo,
)

# Define the public API of this module
__all__ = [
    "Command",
    "CommandException",
    "Raw",
    "Get",
    "GetLine",
    "GetMultiline",
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
    some `Exchange` generators, which are yielded back to the connection,
    then zip together the responses to those exchanges into a tuple"""
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

    If the form of the expected return is known, consider using `GetLine`
    or `GetMultiline` instead.

    Args:
        field: The field, attribute, or star command to get

    For example::

        Get("PCAP.ACTIVE") -> "1"
        Get("SEQ1.TABLE") -> ["1048576", "0", "1000", "1000"]
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
            assert line.startswith("OK =")
            return line[4:]


@dataclass
class GetLine(Command[str]):
    """Get the value of a field or star command, when the result is expected to be a
    single line.

    Args:
        field: The field, attribute, or star command to get

    For example::

        GetLine("PCAP.ACTIVE") -> "1"
        GetLine("*IDN") -> "PandA 1.1..."
    """

    field: str

    def execute(self) -> ExchangeGenerator[str]:
        ex = Exchange(f"{self.field}?")
        yield ex
        # Expect "OK =value"
        line = ex.line
        assert line.startswith("OK =")
        return line[4:]


@dataclass
class GetMultiline(Get):
    """Get the value of a field or star command, when the result is expected to be a
    multiline response.

    Args:
        field: The field, attribute, or star command to get

    For example::

        GetMultiline("SEQ1.TABLE") -> ["1048576", "0", "1000", "1000"]
        GetMultiline("*METADATA.*") -> ["LABEL_FILTER1", "APPNAME", ...]
    """

    field: str

    def execute(self) -> ExchangeGenerator[List[str]]:
        ex = Exchange(f"{self.field}?")
        yield ex
        return ex.multiline


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
class GetBlockInfo(Command[Dict[str, BlockInfo]]):
    """Get the name, number, and description of each block type
    in a dictionary, alphabetically ordered

    Args:
        skip_description: If `True`, prevents retrieving the description
            for each Block. This will reduce network calls.

    For example::

         GetBlockInfo() ->
             {
                 "LUT": BlockInfo(number=8, description="Lookup table"),
                 "PCAP": BlockInfo(number=1, description="Position capture control"),
                 ...
             }
    """

    skip_description: bool = False

    def execute(self) -> ExchangeGenerator[Dict[str, BlockInfo]]:
        ex = Exchange("*BLOCKS?")
        yield ex

        blocks_list, commands = [], []
        for line in ex.multiline:
            block, num = line.split()
            blocks_list.append((block, int(num)))
            commands.append(GetLine(f"*DESC.{block}"))

        if self.skip_description:
            # Must use tuple() to match type returned by _execute_commands
            description_values = tuple(None for _ in commands)
        else:
            description_values = yield from _execute_commands(*commands)

        block_infos = {
            block: BlockInfo(number=num, description=desc)
            for (block, num), desc in sorted(zip(blocks_list, description_values))
        }

        return block_infos


# The type of the generators used for creating the Get commands for each field
# and setting the returned data into the FieldInfo structure
_FieldGeneratorType = Generator[
    Union[Exchange, List[Exchange]],
    # Tuple[Union[List[str], str], ...],
    None,
    Tuple[str, FieldInfo],
]


@dataclass
class GetFieldInfo(Command[Dict[str, FieldInfo]]):
    """Get the fields of a block, returning a `FieldInfo` (or appropriate subclass) for
    each one, ordered to match the definition order in the PandA
    TODO: Update this!

    Args:
        block: The name of the block type
        extended_metadata: If `True`, retrieves detailed metadata about a field and
            all of its attributes. This will cause an additional network round trip.
            If `False` only the field names and types will be returned. Default `True`.

    For example::

        GetFieldInfo("LUT") -> {
            "INPA":
                BitMuxFieldInfo(type='bit_mux',
                                subtype=None,
                                description='Input A',
                                max_delay=5
                                label=['TTLIN1.VAL', 'TTLIN2.VAL', ...]),
            ...}
    """

    block: str
    extended_metadata: bool = True

    def _param_uint(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = UintFieldInfo(field_type, field_subtype)

        max = yield from GetLine(f"{self.block}1.{field_name}.MAX").execute()

        field_info.max = int(max)
        return field_name, field_info

    def _scalar(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = ScalarFieldInfo(field_type, field_subtype)

        units, scale, offset = yield from _execute_commands(
            GetLine(f"{self.block}.{field_name}.UNITS"),
            GetLine(f"{self.block}.{field_name}.SCALE"),
            GetLine(f"{self.block}.{field_name}.OFFSET"),
        )

        field_info.units = units
        field_info.scale = float(scale)
        field_info.offset = int(offset)

        return field_name, field_info

    def _subtype_time(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = SubtypeTimeFieldInfo(field_type, field_subtype)

        field_info.units_labels = yield from GetMultiline(
            f"*ENUMS.{self.block}.{field_name}.UNITS"
        ).execute()

        return field_name, field_info

    def _enum(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = EnumFieldInfo(field_type, field_subtype)

        field_info.labels = yield from GetMultiline(
            f"*ENUMS.{self.block}.{field_name}"
        ).execute()

        return field_name, field_info

    def _time(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = TimeFieldInfo(field_type, field_subtype)

        units, min = yield from _execute_commands(
            GetMultiline(f"*ENUMS.{self.block}.{field_name}.UNITS"),
            GetLine(f"{self.block}1.{field_name}.MIN"),
        )

        field_info.units_labels = list(units)
        field_info.min = float(min)

        return field_name, field_info

    def _bit_out(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = BitOutFieldInfo(field_type, field_subtype)

        capture_word, offset = yield from _execute_commands(
            GetLine(f"{self.block}1.{field_name}.CAPTURE_WORD"),
            GetLine(f"{self.block}1.{field_name}.OFFSET"),
        )

        field_info.capture_word = str(capture_word)
        field_info.offset = int(offset)

        return field_name, field_info

    def _bit_mux(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = BitMuxFieldInfo(field_type, field_subtype)

        max_delay, labels = yield from _execute_commands(
            GetLine(f"{self.block}1.{field_name}.MAX_DELAY"),
            GetMultiline(f"*ENUMS.{self.block}.{field_name}"),
        )

        field_info.max_delay = int(max_delay)
        field_info.labels = list(labels)

        return field_name, field_info

    def _pos_mux(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = PosMuxFieldInfo(field_type, field_subtype)

        field_info.labels = yield from GetMultiline(
            f"*ENUMS.{self.block}.{field_name}"
        ).execute()

        return field_name, field_info

    def _table(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = TableFieldInfo(field_type, field_subtype)

        # Ignore the ROW_WORDS attribute as it's new and won't be present on all PandAs,
        # and there's no easy way to try it and catch an error while also running other
        # Get commands at the same time
        max_length, fields = yield from _execute_commands(
            GetLine(f"{self.block}1.{field_name}.MAX_LENGTH"),
            GetMultiline(f"{self.block}1.{field_name}.FIELDS"),
        )

        field_info.max_length = int(max_length)

        # Keep track of highest bit index
        max_bit_offset: int = 0

        desc_gets: List[GetLine] = []
        enum_field_gets: List[GetMultiline] = []
        enum_field_names: List[str] = []
        for field in fields:
            # Fields are of the form <bit_high>:<bit_low> <name> <subtype>
            bit_range, name, subtype = field.split()
            bit_high_str, bit_low_str = bit_range.split(":")
            bit_high = int(bit_high_str)
            bit_low = int(bit_low_str)

            if bit_high > max_bit_offset:
                max_bit_offset = bit_high

            if subtype == "enum":
                enum_field_gets.append(
                    GetMultiline(f"*ENUMS.{self.block}1.{field_name}[].{name}")
                )
                enum_field_names.append(name)

            info = TableFieldDetails(subtype, bit_low, bit_high)

            if field_info.fields is None:
                field_info.fields = {}

            field_info.fields[name] = info

            desc_gets.append(GetLine(f"*DESC.{self.block}1.{field_name}[].{name}"))

        # Calculate the number of 32 bit words that comprises one table row
        field_info.row_words = max_bit_offset // 32 + 1

        # The first len(enum_field_gets) items are enum labels, type List[str]
        # The second part of the list are descriptions, type str
        labels_and_descriptions = yield from _execute_commands(
            *enum_field_gets, *desc_gets
        )

        assert field_info.fields

        for name, labels in zip(
            enum_field_names, labels_and_descriptions[: len(enum_field_gets)]
        ):
            field_info.fields[name].labels = labels

        for name, desc in zip(
            field_info.fields.keys(), labels_and_descriptions[len(enum_field_gets) :]
        ):
            field_info.fields[name].description = desc

        return field_name, field_info

    def _pos_out(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = PosOutFieldInfo(field_type, field_subtype)

        field_info.capture_labels = yield from GetMultiline(
            f"*ENUMS.{self.block}.{field_name}.CAPTURE"
        ).execute()

        return field_name, field_info

    def _ext_out(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = ExtOutFieldInfo(field_type, field_subtype)

        field_info.capture_labels = yield from GetMultiline(
            f"*ENUMS.{self.block}.{field_name}.CAPTURE"
        ).execute()

        return (
            field_name,
            field_info,
        )

    def _ext_out_bits(
        self, field_name: str, field_type: str, field_subtype: Optional[str]
    ) -> _FieldGeneratorType:
        field_info = ExtOutBitsFieldInfo(field_type, field_subtype)
        bits, capture_labels = yield from _execute_commands(
            GetMultiline(f"{self.block}.{field_name}.BITS"),
            GetMultiline(f"*ENUMS.{self.block}.{field_name}.CAPTURE"),
        )
        field_info.bits = list(bits)
        field_info.capture_labels = list(capture_labels)
        return field_name, field_info

    def execute(self) -> ExchangeGenerator[Dict[str, FieldInfo]]:
        ex = Exchange(f"{self.block}.*?")
        yield ex
        unsorted: Dict[int, Tuple[str, FieldInfo]] = {}
        field_generators: List[ExchangeGenerator] = []
        desc_generators: List[ExchangeGenerator] = []

        for line in ex.multiline:
            field_name, index, type_subtype = line.split(maxsplit=2)

            # Append "None" to list below so there are always at least 2 elements
            # so we can always unpack into subtype, even if no split occurs.
            field_type, subtype, *_ = [*type_subtype.split(maxsplit=1), None]

            # Map a (type, subtype) to a method that returns the appropriate
            # subclasss of FieldInfo, and a list of all the Commands to request.
            # Note that fields that do not have additional attributes are not listed.
            # TODO: This is static so should live somewhere not inline - but it can't
            # live on the Dataclass itself due to Python restrictions...
            _commands_map: Dict[
                Tuple[str, Optional[str]],
                Callable[
                    [str, str, Optional[str]],
                    _FieldGeneratorType,
                ],
            ] = {
                # Order matches that of PandA server's Field Types docs
                ("time", None): self._time,
                ("bit_out", None): self._bit_out,
                ("pos_out", None): self._pos_out,
                ("ext_out", "timestamp"): self._ext_out,
                ("ext_out", "samples"): self._ext_out,
                ("ext_out", "bits"): self._ext_out_bits,
                ("bit_mux", None): self._bit_mux,
                ("pos_mux", None): self._pos_mux,
                ("table", None): self._table,
                ("param", "uint"): self._param_uint,
                ("read", "uint"): self._param_uint,
                ("write", "uint"): self._param_uint,
                ("param", "scalar"): self._scalar,
                ("read", "scalar"): self._scalar,
                ("write", "scalar"): self._scalar,
                ("param", "time"): self._subtype_time,
                ("read", "time"): self._subtype_time,
                ("write", "time"): self._subtype_time,
                ("param", "enum"): self._enum,
                ("read", "enum"): self._enum,
                ("write", "enum"): self._enum,
            }

            # Always create default FieldInfo. If necessary we will replace it later
            # with a more type-specific version.
            field_info = FieldInfo(field_type, subtype)

            if self.extended_metadata:
                try:
                    # Construct the list of type-specific generators
                    field_generators.append(
                        _commands_map[(field_type, subtype)](
                            field_name, field_type, subtype
                        )
                    )

                except KeyError:
                    # No type-specific commands to create
                    # Many fields have no attributes so this is frequently expected
                    # Also serves for future-proofing if new types/subtypes are defined
                    # TODO: Add tests for unknown types and subtypes
                    # TODO: Add a warning we encountered an unknown type
                    pass

                # Description is common to all fields
                # Note that we don't get the description for any attributes - these are
                # fixed strings and so not worth retrieving dynamically.
                desc_generators.append(
                    GetLine(f"*DESC.{self.block}.{field_name}").execute()
                )

            # Keep track of order of fields as returned by PandA. Important for later
            # matching descriptions back to their field.
            unsorted[int(index)] = (field_name, field_info)

        # Dict keeps insertion order, so insert in the order the server said
        # TODO: Confirm with Tom regarding the order of fields - the docs and the
        # comments imply the order of the server should be respected, but the
        # line below calls sorted().
        fields = {name: field for _, (name, field) in sorted(unsorted.items())}

        if self.extended_metadata is False:
            # Asked to not perform the requests for extra metadata.
            return fields

        # The first <len(fields)> elements are type Tuple[str, FieldInfo]
        # The second section of the elements are type str (field descriptions)
        infos_and_descriptions = yield from _zip_with_return(
            field_generators + desc_generators
        )

        field_name_info: Tuple[Tuple[str, FieldInfo], ...] = infos_and_descriptions[
            : len(field_generators)
        ]

        fields.update(field_name_info)

        desc: str
        for field_name, desc in zip(
            [item[0] for item in unsorted.values()],
            infos_and_descriptions[len(field_generators) :],
        ):
            fields[field_name].description = desc

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

                if field_type == "ext_out" and field_subtype == "bits":
                    bits_fields.append(f"PCAP.{field_name}")

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
        get_multiline: If `True`, return values of multiline fields in the
            `multiline_values` attribute. Note that this will invoke additional network
            requests.
            If `False` these fields will instead be returned in the `no_value`
            attribute. Default value is `False`.
        multiline_base64: If `True` any multiline values retrieved will use Base64
            encoding if possible, returning data as a single Base64 encoded value rather
            than a list of values. Only has an effect if ``get_multiline`` is `True`.
            This will typically increase data sent over the network.
            Default `False`.

    For example::

        GetChanges() -> Changes(
            value={"PCAP.TRIG": "PULSE1.OUT"},
            no_value=["SEQ1.TABLE"],
            in_error=["BAD.ENUM"],
            multiline_values={}
        )

        GetChanges(ChangeGroup.ALL, True) -> Changes(
            values={"PCAP.TRIG": "PULSE1.OUT"},
            no_value=[],
            in_error=["BAD.ENUM"],
            multiline_values={"SEQ1.TABLE" : ["1", "2", "3",...]}
        )
    """

    group: ChangeGroup = ChangeGroup.ALL
    get_multiline: bool = False
    multiline_base64: bool = False

    def execute(self) -> ExchangeGenerator[Changes]:
        ex = Exchange(f"*CHANGES{self.group.value}?")
        yield ex
        changes = Changes({}, [], [], {})
        multivalue_get_commands: List[Tuple[str, GetMultiline]] = []
        for line in ex.multiline:
            if line[-1] == "<":
                if self.get_multiline:
                    field = line[0:-1]

                    # The METADATA tables are strings, and cannot be requested in Base64
                    if self.multiline_base64 and "*METADATA" not in field:
                        base64 = ".B"
                    else:
                        base64 = ""

                    multivalue_get_commands.append(
                        (field, GetMultiline(field + base64))
                    )

                else:
                    changes.no_value.append(line[:-1])
            elif line.endswith("(error)"):
                changes.in_error.append(line.split(" ", 1)[0])
            else:
                field, value = line.split("=", maxsplit=1)
                changes.values[field] = value

        if self.get_multiline:
            multiline_vals = yield from _execute_commands(
                *[item[1] for item in multivalue_get_commands]
            )

            for field, value in zip(
                [item[0] for item in multivalue_get_commands], multiline_vals
            ):
                assert isinstance(value, list)
                changes.multiline_values[field] = value

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
            commands.append(GetMultiline(f"{field}.B"))
        for field in metadata.no_value:
            # Get metadata as string list
            multiline_keys.append(f"{field}<")
            commands.append(GetMultiline(f"{field}"))
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
