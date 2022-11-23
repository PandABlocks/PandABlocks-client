# Various new or derived types/classes and helper functions for the IOC module
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, NewType, Optional, Union

from softioc import builder
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.responses import FieldInfo


class InErrorException(Exception):
    """Placeholder exception to mark a field as being in error as reported by PandA"""


# Custom type aliases and new types
ScalarRecordValue = Union[str, InErrorException]
TableRecordValue = List[str]
RecordValue = Union[ScalarRecordValue, TableRecordValue]
# EPICS format, i.e. ":" dividers
EpicsName = NewType("EpicsName", str)
# PandA format, i.e. "." dividers
PandAName = NewType("PandAName", str)


def panda_to_epics_name(field_name: PandAName) -> EpicsName:
    """Convert PandA naming convention to EPICS convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return EpicsName(field_name.replace(".", ":"))


def epics_to_panda_name(field_name: EpicsName) -> PandAName:
    """Convert EPICS naming convention to PandA convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return PandAName(field_name.replace(":", "."))


def device_and_record_to_panda_name(field_name: EpicsName) -> PandAName:
    """Convert an EPICS naming convention (including Device prefix) to PandA
    convention."""
    _, record_name = field_name.split(":", maxsplit=1)
    return epics_to_panda_name(EpicsName(record_name))


def check_num_labels(labels: List[str], record_name: str):
    """Check that the number of labels can fit into an mbbi/mbbo record"""
    assert (
        len(labels) <= 16
    ), f"Too many labels ({len(labels)}) to create record {record_name}"


def trim_string_value(value: str, record_name: str) -> str:
    """Record value for string records is a maximum of 40 characters long. Ensure any
    string is shorter than that before setting it."""
    if len(value) > 39:
        logging.error(
            f"Value for {record_name} longer than EPICS limit of 40 characters."
            f"It will be truncated. Value: {value}"
        )
        value = value[:39]
    return value


def trim_description(description: Optional[str], record_name: str) -> Optional[str]:
    """Record description field is a maximum of 40 characters long. Ensure any string
    is shorter than that before setting it."""
    # TODO: Trim leading and trailing spaces?
    if description and len(description) > 39:
        # As per Tom Cobb, it's unlikely we'll ever re-write descriptions to be shorter,
        # so we'll hide this message in low level logging only
        logging.info(
            f"Description for {record_name} longer than EPICS limit of "
            f"40 characters. It will be truncated. Description: {description}"
        )
        description = description[:39]
    return description


# Constants used in bool records
ZNAM_STR = "0"
ONAM_STR = "1"

# The list of all OUT record types
OUT_RECORD_FUNCTIONS = [
    builder.aOut,
    builder.boolOut,
    builder.Action,
    builder.mbbOut,
    builder.longOut,
    builder.longStringOut,
    builder.stringOut,
    builder.WaveformOut,
]


@dataclass
class RecordInfo:
    """A container for a record and extra information needed to later update
    the record.

    `record`: The PythonSoftIOC RecordWrapper instance. Must be provided
        via the add_record() method.
    `record_prefix`: The device prefix the record uses.
    `data_type_func`: Function to convert string data to form appropriate for the record
    `labels`: List of valid labels for the record. By setting this field to non-None,
        the `record` is assumed to be mbbi/mbbo type.
    `is_in_record`: Flag for whether the `record` is an "In" record type.
    `on_changes_func`: Function called during processing of *CHANGES? for this record
    `_pending_change`: Marks whether this record was just Put data to PandA, and so is
        expecting to see the same value come back from a *CHANGES? request.
    `_field_info`: The FieldInfo structure associated with this record. May be a
        subclass of FieldInfo."""

    record: RecordWrapper = field(init=False)
    data_type_func: Callable
    labels: Optional[List[str]] = None
    # PythonSoftIOC issues #52 or #54 may remove need for is_in_record
    is_in_record: bool = True
    on_changes_func: Optional[Callable[[Any], Awaitable[None]]] = None
    _pending_change: bool = field(default=False, init=False)
    _field_info: Optional[FieldInfo] = field(default=None, init=False)

    def add_record(self, record: RecordWrapper) -> None:
        self.record = record
