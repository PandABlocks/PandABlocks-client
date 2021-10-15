from dataclasses import dataclass

# Various new or derived types/classes for the IOC module
# Mostly exists to avoid circular dependencies.
from typing import Callable, List, NewType, Optional, Union

from softioc.pythonSoftIoc import RecordWrapper

# EPICS format, i.e. ":" dividers
EpicsName = NewType("EpicsName", str)
# PAndA format, i.e. "." dividers
PandAName = NewType("PandAName", str)


def _panda_to_epics_name(field_name: PandAName) -> EpicsName:
    """Convert PandA naming convention to EPICS convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return EpicsName(field_name.replace(".", ":"))


def _epics_to_panda_name(field_name: EpicsName) -> PandAName:
    """Convert EPICS naming convention to PandA convention. This module defaults to
    EPICS names internally, only converting back to PandA names when necessary."""
    return PandAName(field_name.replace(":", "."))


class _InErrorException(Exception):
    """Placeholder exception to mark a field as being in error as reported by PandA"""


# Custom type aliases and new types
ScalarRecordValue = Union[str, _InErrorException]
TableRecordValue = List[str]
RecordValue = Union[ScalarRecordValue, TableRecordValue]


@dataclass
class _RecordInfo:
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
    # TODO: Removing this for the moment to work out circular dependency
    # table_updater: Optional[_TableUpdater] = None
    # PythonSoftIOC issues #52 or #54 may remove need for is_in_record
    is_in_record: bool = True
