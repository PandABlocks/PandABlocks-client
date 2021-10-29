# Various new or derived types/classes and helper functions for the IOC module
# Mostly exists to avoid circular dependencies.
import logging
from dataclasses import dataclass
from typing import Callable, List, NewType, Optional, Union

from softioc.pythonSoftIoc import RecordWrapper

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


def check_num_labels(labels: List[str], record_name: str):
    """Check that the number of labels can fit into an mbbi/mbbo record"""
    assert (
        len(labels) <= 16
    ), f"Too many labels ({len(labels)}) to create record {record_name}"


def trim_description(description: Optional[str], record_name: str) -> Optional[str]:
    """Record description field is a maximum of 40 characters long. Ensure any string
    is shorter than that before setting it."""
    if description and len(description) > 40:
        # As per Tom Cobb, it's unlikely the descriptions will ever be truncated so
        # we'll hide this message in low level logging only
        logging.info(
            f"Description for {record_name} longer than EPICS limit of "
            f"40 characters. It will be truncated. Description: {description}"
        )
        description = description[:40]
    return description


# Constants used in bool records
ZNAM_STR = "0"
ONAM_STR = "1"


class InErrorException(Exception):
    """Placeholder exception to mark a field as being in error as reported by PandA"""


# Custom type aliases and new types
ScalarRecordValue = Union[str, InErrorException]
TableRecordValue = List[str]
RecordValue = Union[ScalarRecordValue, TableRecordValue]


@dataclass
class RecordInfo:
    """A container for a record and extra information needed to later update
    the record.

    `record`: The PythonSoftIOC RecordWrapper instance
    `data_type_func`: Function to convert string data to form appropriate for the record
    `labels`: List of valid labels for the record. By setting this field to non-None,
        the `record` is assumed to be mbbi/mbbo type.
    `is_in_record`: Flag for whether the `record` is an "In" record type."""

    record: RecordWrapper
    data_type_func: Callable
    labels: Optional[List[str]] = None
    # PythonSoftIOC issues #52 or #54 may remove need for is_in_record
    is_in_record: bool = True
