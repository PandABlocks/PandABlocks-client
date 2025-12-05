import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Define the public API of this module
__all__ = [
    "BlockInfo",
    "FieldInfo",
    "UintFieldInfo",
    "ScalarFieldInfo",
    "TimeFieldInfo",
    "SubtypeTimeFieldInfo",
    "EnumFieldInfo",
    "BitOutFieldInfo",
    "BitMuxFieldInfo",
    "PosMuxFieldInfo",
    "TableFieldDetails",
    "TableFieldInfo",
    "PosOutFieldInfo",
    "ExtOutFieldInfo",
    "ExtOutBitsFieldInfo",
    "Changes",
    "EndReason",
    "FieldCapture",
    "Data",
    "ReadyData",
    "StartData",
    "FrameData",
    "EndData",
]

# Control

# https://regex101.com/r/LZ71Ns/1
API_EXTRACT = re.compile(r"^(\d+).(\d+)")


@dataclass
class BlockInfo:
    """Block number and description as exposed by the TCP server

    Attributes:
        number: The index of this block
        description: The description for this block"""

    number: int = 0
    description: Optional[str] = None


@dataclass
class FieldInfo:
    """Field type, subtype, description and labels as exposed by TCP server:
    https://pandablocks-server.readthedocs.io/en/latest/fields.html#field-types

    Note that many fields will use a more specialised subclass of FieldInfo for
    their additional attributes.

    Attributes:
        type: Field type, like "param", "bit_out", "pos_mux", etc.
        subtype: Some types have subtype, like "uint", "scalar", "lut", etc.
        description: A description of the field
        labels: A list of the valid values for the field when there is a defined list
            of valid values, e.g. those with sub-type "enum"
    """

    type: str
    subtype: Optional[str]
    description: Optional[str]


@dataclass
class UintFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "param","read", or "write" and subtype
    "uint"""

    max_val: int


@dataclass
class ScalarFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "param","read", or "write" and subtype
    "scalar"""

    units: str
    scale: float
    offset: float


@dataclass
class TimeFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "time"""

    units_labels: list[str]


@dataclass
class SubtypeTimeFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "param","read", or "write" and subtype
    "time"""

    units_labels: list[str]


@dataclass
class EnumFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "param","read", or "write" and subtype
    "enum"""

    labels: list[str]


@dataclass
class BitOutFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "bit_out"""

    capture_word: str
    offset: int


@dataclass
class BitMuxFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "bit_mux"""

    max_delay: int
    labels: list[str]


@dataclass
class PosMuxFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "pos_mux"""

    labels: list[str]


@dataclass
class TableFieldDetails:
    """Info for each field in a table"""

    subtype: str
    bit_low: int
    bit_high: int
    description: Optional[str] = None
    labels: Optional[list[str]] = None


@dataclass
class TableFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "table"

    Attribute "has_mode" is set to True when created from `GetFieldInfo`
    with API version >= (4, 0)
    """

    max_length: int
    fields: dict[str, TableFieldDetails]
    row_words: int
    has_mode: bool = False


@dataclass
class PosOutFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "pos_out"""

    capture_labels: list[str]


@dataclass
class ExtOutFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "ext_out" and subtypes "timestamp"
    or "samples"""

    capture_labels: list[str]


@dataclass
class ExtOutBitsFieldInfo(ExtOutFieldInfo):
    """Extended `ExtOutFieldInfo` for fields with type "ext_out" and subtype "bits"""

    bits: list[str]


@dataclass
class Changes:
    """The changes returned from a ``*CHANGES`` command"""

    #: Map field -> value for single-line values that were returned
    values: dict[str, str]
    #: The fields that were present but without value
    no_value: list[str]
    #: The fields that were in error
    in_error: list[str]
    #: Map field -> value for multi-line values that were returned
    multiline_values: dict[str, list[str]]


# Data


class EndReason(Enum):
    """The reason that a PCAP acquisition completed"""

    #: Experiment completed by falling edge of ``PCAP.ENABLE```
    OK = "Ok"
    #: Client disconnect detected
    EARLY_DISCONNECT = "Early disconnect"
    #: Client not taking data quickly or network congestion, internal buffer overflow.
    #: In raw unscaled mode (i.e., no server-side scaling), the most recent
    #: `FrameData` is likely corrupted.
    DATA_OVERRUN = "Data overrun"
    #: Triggers too fast for configured data capture
    FRAMING_ERROR = "Framing error"
    #: Probable CPU overload on PandA, should not occur
    DRIVER_DATA_OVERRUN = "Driver data overrun"
    #: Data capture too fast for memory bandwidth
    DMA_DATA_ERROR = "DMA data error"
    # Reasons below this point are not from the server, they are generated in code
    #: An unknown exception occurred during HDF5 file processing
    UNKNOWN_EXCEPTION = "Unknown exception"
    #: StartData packets did not match when trying to continue printing to a file
    START_DATA_MISMATCH = "Start Data mismatched"
    #: Experiment manually completed by ``DATA:CAPTURE``
    MANUALLY_STOPPED = "Manually stopped"
    #: Experiment manually completed by ``*PCAP.DISARM=`` command
    DISARMED = "Disarmed"


@dataclass
class FieldCapture:
    """Information about a field that is being captured

    If scale, offset, and units are all `None`, then the field is a
    ``PCAP.BITS``.

    Attributes:
        name: Name of captured field
        type: Numpy data type of the field as transmitted
        capture: Value of CAPTURE field used to enable this field
        scale: Scaling factor
        offset: Offset
        units: Units string
    """

    name: str
    type: np.dtype
    capture: str
    scale: Optional[float] = field(default=None)
    offset: Optional[float] = field(default=None)
    units: Optional[str] = field(default=None)

    def __post_init__(self):
        sou = (self.scale, self.offset, self.units)
        if sou != (None, None, None) and None in sou:
            raise ValueError(
                f"If any of `scale={self.scale}`, `offset={self.offset}`"
                f", or `units={self.units}` is set, all must be set."
            )

    @property
    def raw_mode_dataset_dtype(self) -> np.dtype:
        """We use double for all dtypes that have scale and offset."""
        if self.scale is not None and self.offset is not None:
            return np.dtype("float64")
        return self.type

    @property
    def has_scale_or_offset(self) -> bool:
        """Return True if this field is a PCAP.BITS or PCAP.SAMPLES field"""
        return (self.scale is not None and self.offset is not None) and (
            self.scale != 1 or self.offset != 0
        )


class Data:
    """Baseclass for all responses yielded by a `DataConnection`"""


@dataclass
class ReadyData(Data):
    """Yielded once when the connection is established and ready to take data"""


@dataclass
class StartData(Data):
    """Yielded when a new PCAP acquisition starts.

    Attributes:
        fields: Information about each captured field as a `FieldCapture` object
        missed: Number of samples missed by late data port connection
        process: Data processing option, only "Scaled" or "Raw" are requested
        format: Data delivery formatting, only "Framed" is requested
        sample_bytes: Number of bytes in one sample
    """

    fields: list[FieldCapture]
    missed: int
    process: str
    format: str
    sample_bytes: int
    arm_time: Optional[str]
    start_time: Optional[str]
    hw_time_offset_ns: Optional[int]


@dataclass
class FrameData(Data):
    """Yielded when a new data frame is flushed.

    Attributes:
        data: A numpy `Structured Array <structured_arrays>`

    Data is structured into complete columns. Each column name is
    ``<name>.<capture>`` from the corresponding `FieldInfo`. Data
    can be accessed with these column names. For example::

        # Table view with 2 captured fields
        >>> import numpy
        >>> data = numpy.array([(0, 10),
        ...       (1, 11),
        ...       (2, 12)],
        ...      dtype=[('COUNTER1.OUT.Value', '<f8'), ('COUNTER2.OUT.Value', '<f8')])
        >>> fdata = FrameData(data)
        >>> (fdata.data[0]['COUNTER1.OUT.Value'], fdata.data[0]['COUNTER2.OUT.Value'])
        (np.float64(0.0), np.float64(10.0))
        >>> fdata.column_names # Column names
        ('COUNTER1.OUT.Value', 'COUNTER2.OUT.Value')
        >>> fdata.data['COUNTER1.OUT.Value'] # Column view
        array([0., 1., 2.])
    """

    data: np.ndarray

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return all the column names"""
        names = self.data.dtype.names
        assert names, f"No column names for {self.data.dtype}"
        return names


@dataclass
class EndData(Data):
    """Yielded when a PCAP acquisition ends.

    Attributes:
        samples: The total number of samples (rows) that were yielded
        reason: The `EndReason` for the end of acquisition
    """

    samples: int
    reason: EndReason


@dataclass
class Identification:
    """Versions of components as exposed by TCP server:
    https://pandablocks.github.io/PandABlocks-server/master/commands.html

    Attributes:
        software: Version of the TCP server (PandABlocks-server)
        fpga: Version of the FPGA firmware, build number, supporting
        firmware (PandABlocks-FPGA)
        rootfs: Version of the root filesystem (PandABlocks-rootfs)
    """

    software: str
    fpga: str
    rootfs: str

    def software_api(self) -> tuple[int, int]:
        match = API_EXTRACT.match(self.software)
        assert match, (
            f"PandA SW: {self.software} does not match expected pattern {API_EXTRACT}"
        )
        major, minor = match.groups()
        return (int(major), int(minor))
