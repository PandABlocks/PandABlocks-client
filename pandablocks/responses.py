from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

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

    units_labels: List[str]
    min_val: float


@dataclass
class SubtypeTimeFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "param","read", or "write" and subtype
    "time"""

    units_labels: List[str]


@dataclass
class EnumFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "param","read", or "write" and subtype
    "enum"""

    labels: List[str]


@dataclass
class BitOutFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "bit_out"""

    capture_word: str
    offset: int


@dataclass
class BitMuxFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "bit_mux"""

    max_delay: int
    labels: List[str]


@dataclass
class PosMuxFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "pos_mux"""

    labels: List[str]


@dataclass
class TableFieldDetails:
    """Info for each field in a table"""

    subtype: str
    bit_low: int
    bit_high: int
    description: Optional[str] = None
    labels: Optional[List[str]] = None


@dataclass
class TableFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "table"""

    max_length: int
    fields: Dict[str, TableFieldDetails]
    row_words: int


@dataclass
class PosOutFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "pos_out"""

    capture_labels: List[str]


@dataclass
class ExtOutFieldInfo(FieldInfo):
    """Extended `FieldInfo` for fields with type "ext_out" and subtypes "timestamp"
    or "samples"""

    capture_labels: List[str]


@dataclass
class ExtOutBitsFieldInfo(ExtOutFieldInfo):
    """Extended `ExtOutFieldInfo` for fields with type "ext_out" and subtype "bits"""

    bits: List[str]


@dataclass
class Changes:
    """The changes returned from a ``*CHANGES`` command"""

    #: Map field -> value for single-line values that were returned
    values: Dict[str, str]
    #: The fields that were present but without value
    no_value: List[str]
    #: The fields that were in error
    in_error: List[str]
    #: Map field -> value for multi-line values that were returned
    multiline_values: Dict[str, List[str]]


# Data


class EndReason(Enum):
    """The reason that a PCAP acquisition completed"""

    #: Experiment completed by falling edge of ``PCAP.ENABLE```
    OK = "Ok"
    #: Experiment manually completed by ``*PCAP.DISARM=`` command
    DISARMED = "Disarmed"
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


@dataclass
class FieldCapture:
    """Information about a field that is being captured

    Attributes:
        name: Name of captured field
        type: Numpy data type of the field as transmitted
        capture: Value of CAPTURE field used to enable this field
        scale: Scaling factor, default 1.0
        offset: Offset, default 0.0
        units: Units string, default ""
    """

    name: str
    type: np.dtype
    capture: str
    scale: float = 1.0
    offset: float = 0.0
    units: str = ""


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

    fields: List[FieldCapture]
    missed: int
    process: str
    format: str
    sample_bytes: int


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
        >>> fdata.data[0] # Row view
        (0., 10.)
        >>> fdata.column_names # Column names
        ('COUNTER1.OUT.Value', 'COUNTER2.OUT.Value')
        >>> fdata.data['COUNTER1.OUT.Value'] # Column view
        array([0., 1., 2.])
    """

    data: np.ndarray

    @property
    def column_names(self) -> Tuple[str, ...]:
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
