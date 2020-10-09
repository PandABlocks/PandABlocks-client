from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

# Control


@dataclass
class FieldType:
    type: str
    subtype: Optional[str] = None


# Data


class EndReason(Enum):
    # Experiment completed without intervention.
    OK = "Ok"
    # Experiment manually completed by *PCAP.DISARM= command.
    DISARMED = "Disarmed"
    # Client disconnect detected.
    EARLY_DISCONNECT = "Early disconnect"
    # Client not taking data quickly or network congestion, internal buffer overflow.
    DATA_OVERRUN = "Data overrun"
    # Triggers too fast for configured data capture.
    FRAMING_ERROR = "Framing error"
    # Probable CPU overload on PandA, should not occur.
    DRIVER_DATA_OVERRUN = "Driver data overrun"
    # Data capture too fast for memory bandwidth.
    DMA_DATA_ERROR = "DMA data error"


@dataclass
class DataField:
    name: str
    type: np.dtype
    capture: str
    scale: float = 1.0
    offset: float = 0.0
    units: str = ""


class Data:
    pass


@dataclass
class StartData(Data):
    fields: List[DataField]
    missed: int
    process: str  # Raw or Scaled
    format: str  # Framed
    sample_bytes: int


@dataclass
class FrameData(Data):
    data: np.ndarray


@dataclass
class EndData(Data):
    samples: int
    reason: EndReason
