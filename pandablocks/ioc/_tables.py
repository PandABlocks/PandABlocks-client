# IOC Table record support

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from softioc import alarm
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetMultiline, Put
from pandablocks.ioc._types import (
    EpicsName,
    RecordValue,
    _epics_to_panda_name,
    _InErrorException,
    _RecordInfo,
)
from pandablocks.responses import TableFieldDetails, TableFieldInfo


@dataclass
class TableRecordWrapper:
    """Replacement RecordWrapper for controlling Tables.
    This is only expected to be used for MODE records."""

    record: RecordWrapper
    table_updater: "_TableUpdater"

    def set(self, values: List[str]) -> None:
        """Set the given values into the table records"""
        self.table_updater.update_table(values)

    def __getattr__(self, name):
        """Forward all requests for other attributes to the underlying Record"""
        return getattr(self.record, name)


class TablePacking:
    """Class to handle packing and unpacking Table data to and from a PandA"""

    @staticmethod
    def unpack(
        row_words: int,
        table_fields: Dict[str, TableFieldDetails],
        table_data: List[str],
    ) -> List[np.ndarray]:
        """Unpacks the given `packed` data based on the fields provided.
        Returns the unpacked data in column-indexed format

        Args:
            row_words: The number of 32-bit words per row
            table_fields: The list of fields present in the packed data. Must be ordered
                in bit-ascending order (i.e. lowest bit_low field first)
            table_data: The list of data for this table, from PandA. Each item is
                expected to be the string representation of a uint32.

        Returns:
            numpy array: A list of 1-D numpy arrays, one item per field. Each item
            will have length equal to the PandA table's number of rows.
        """

        data = np.array(table_data, dtype=np.uint32)
        # Convert 1-D array into 2-D, one row element per row in the PandA table
        data = data.reshape(len(data) // row_words, row_words)
        packed = data.T

        unpacked = []
        for name, field_details in table_fields.items():
            offset = field_details.bit_low
            bit_len = field_details.bit_high - field_details.bit_low + 1

            # The word offset indicates which column this field is in
            # (column is exactly one 32-bit word)
            word_offset = offset // 32

            # bit offset is location of field inside the word
            bit_offset = offset & 0x1F

            # Mask to remove every bit that isn't in the range we want
            mask = (1 << bit_len) - 1

            # Can't use proper numpy types, that's only available in 1.21+
            val: Any = (packed[word_offset] >> bit_offset) & mask

            if field_details.subtype == "int":
                # First convert from 2's complement to offset, then add in offset.
                # TODO: Test this with extreme values - int_max, int_min, etc.
                val = np.int32((val ^ (1 << (bit_len - 1))) + (-1 << (bit_len - 1)))
            else:
                # Use shorter types, as these are used in waveform creation
                if bit_len <= 8:
                    val = np.uint8(val)
                elif bit_len <= 16:
                    val = np.uint16(val)

            unpacked.append(val)

        return unpacked

    @staticmethod
    def pack(
        row_words: int,
        table_records: Dict[EpicsName, _RecordInfo],
        table_fields: Dict[str, TableFieldDetails],
    ) -> List[str]:
        """Pack the records based on the field definitions into the format PandA expects
        for table writes.
        Args:
            row_words: The number of 32-bit words per row
            table_records: The list of fields and their associated _RecordInfo
                structure, used to access the value of each record.
            table_fields: The list of fields present in the packed data. Must be ordered
                in bit-ascending order (i.e. lowest bit_low field first)

        Returns:
            List[str]: The list of data ready to be sent to PandA
        """

        packed = None

        # Iterate over the zipped fields and their associated records to construct the
        # packed array. Note that the MODE record is at the end of the table_records,
        # and so will be ignored by zip() as it has no associated value in table_fields
        for field_details, record_info in zip(
            table_fields.values(), table_records.values()
        ):
            curr_val = record_info.record.get()
            assert isinstance(curr_val, np.ndarray)  # Check no SCALAR records here
            # PandA always handles tables in uint32 format
            curr_val = np.uint32(curr_val)

            if packed is None:
                # Create 1-D array sufficiently long to exactly hold the entire table
                packed = np.zeros((len(curr_val), row_words), dtype=np.uint32)
            else:
                assert len(packed) == len(curr_val), (
                    f"Table record {record_info.record.name} has mismatched length "
                    "compared to other records, cannot pack data"
                )

            offset = field_details.bit_low

            # The word offset indicates which column this field is in
            # (each column is one 32-bit word)
            word_offset = offset // 32
            # bit offset is location of field inside the word
            bit_offset = offset & 0x1F

            # Slice to get the column to apply the values to.
            # bit shift the value to the relevant bits of the word
            packed[:, word_offset] |= curr_val << bit_offset

        assert isinstance(packed, np.ndarray)  # Squash mypy warning

        # 2-D array -> 1-D array -> list[int] -> list[str]
        return [str(x) for x in packed.flatten().tolist()]


class TableModeEnum(Enum):
    """Operation modes for the MODES record on PandA table fields"""

    VIEW = 0  # Discard all EPICS record updates, process all PandA updates (default)
    EDIT = 1  # Process all EPICS record updates, discard all PandA updates
    SUBMIT = 2  # Push EPICS records to PandA, overriding current PandA data
    DISCARD = 3  # Discard all EPICS records, re-fetch from PandA


@dataclass
class _TableUpdater:
    """Class to handle updating table records, batching PUTs into the smallest number
    possible to avoid unnecessary/unwanted processing on the PandA

    Args:
        client: The client to be used to read/write to the PandA
        table_name: The name of the table, in EPICS format, e.g. "SEQ1:TABLE"
        field_info: The TableFieldInfo structure for this table
        table_fields: The list of all fields in the table. During initialization they
            will be sorted into bit-ascending order
        table_records: The list of records that comprises this table
        all_values_dict: The dictionary containing the most recent value of all records
            as returned from GetChanges. This dict will be dynamically updated by other
            methods."""

    client: AsyncioClient
    table_name: EpicsName
    field_info: TableFieldInfo
    table_fields: Dict[str, TableFieldDetails]
    table_records: Dict[EpicsName, _RecordInfo]
    all_values_dict: Dict[EpicsName, RecordValue]

    def __post_init__(self):
        # Called at the end of dataclass's __init__
        # The field order will be whatever was configured in the PandA.
        # Ensure fields in bit order from lowest to highest so we can parse data
        self.table_fields = dict(
            sorted(self.table_fields.items(), key=lambda item: item[1].bit_low)
        )

    def validate_waveform(self, record: RecordWrapper, new_val) -> bool:
        """Controls whether updates to the waveform records are processed, based on the
        value of the MODE record.

        Args:
            record: The record currently being validated
            new_val: The new value attempting to be written

        Returns:
            bool: `True` to allow record update, `False` otherwise.
        """

        record_val = self._mode_record_info.record.get()

        if record_val == TableModeEnum.VIEW.value:
            logging.debug(
                f"{self.table_name} MODE record is VIEW, stopping update "
                f"to {record.name}"
            )
            return False
        elif record_val == TableModeEnum.EDIT.value:
            logging.debug(
                f"{self.table_name} MODE record is EDIT, allowing update "
                f"to {record.name}"
            )
            return True
        elif record_val == TableModeEnum.SUBMIT.value:
            # SUBMIT only present when currently writing out data to PandA.
            logging.warning(
                f"Update of record {record.name} attempted while MODE was SUBMIT."
                "New value will be discarded"
            )
            return False
        elif record_val == TableModeEnum.DISCARD.value:
            # DISCARD only present when currently overriding local data with PandA data
            logging.warning(
                f"Update of record {record.name} attempted while MODE was DISCARD."
                "New value will be discarded"
            )
            return False
        else:
            logging.error("MODE record has unknown value: " + str(record_val))
            # In case it isn't already, set an alarm state on the record
            self._mode_record_info.record.set_alarm(
                alarm.INVALID_ALARM, alarm.UDF_ALARM
            )
            return False

    async def update_waveform(self, new_val: int, record_name: str) -> None:
        """Handles updates to a specific waveform record to update its associated
        scalar value record"""
        self._update_scalar(record_name)

    async def update_mode(self, new_val: int):
        """Controls the behaviour when the MODE record is updated.
        Controls Put'ting data back to PandA, or re-Get'ting data from Panda
        and replacing record data."""

        assert self._mode_record_info.labels

        new_label = self._mode_record_info.labels[new_val]

        if new_label == TableModeEnum.SUBMIT.name:
            try:
                # Send all EPICS data to PandA
                logging.info(f"Sending table data for {self.table_name} to PandA")
                assert self.field_info.row_words
                packed_data = []
                packed_data = TablePacking.pack(
                    self.field_info.row_words,
                    self._get_table_field_records(),
                    self.table_fields,
                )

                panda_field_name = _epics_to_panda_name(self.table_name)
                await self.client.send(Put(panda_field_name, packed_data))

            except Exception:
                logging.exception(
                    f"Unable to Put record {self.table_name}, value {packed_data},"
                    "to PandA. Rolling back to last value from PandA.",
                )

                # Reset value of all table records to last values returned from
                # GetChanges
                assert self.field_info.row_words
                assert self.table_name in self.all_values_dict
                old_val = self.all_values_dict[self.table_name]

                if isinstance(old_val, _InErrorException):
                    # If PythonSoftIOC issue #53 is fixed we could put some error state.
                    logging.error(
                        f"Cannot restore previous value to table {self.table_name}, "
                        "PandA marks this field as in error."
                    )
                    return

                assert isinstance(old_val, list)
                field_data = TablePacking.unpack(
                    self.field_info.row_words, self.table_fields, old_val
                )
                # Table records are never In type, so can always disable processing
                for record_info, data in zip(
                    self._get_table_field_records().values(), field_data
                ):
                    record_info.record.set(data, process=False)
            finally:
                # Already in on_update of this record, so disable processing to
                # avoid recursion
                self._mode_record_info.record.set(
                    TableModeEnum.VIEW.value, process=False
                )

        elif new_label == TableModeEnum.DISCARD.name:
            # Recreate EPICS data from PandA data
            logging.info(f"Re-fetching table {self.table_name} data from PandA")
            panda_field_name = _epics_to_panda_name(self.table_name)
            panda_vals = await self.client.send(GetMultiline(f"{panda_field_name}"))

            assert self.field_info.row_words
            field_data = TablePacking.unpack(
                self.field_info.row_words, self.table_fields, panda_vals
            )

            for record_info, data in zip(self.table_records.values(), field_data):
                record_info.record.set(data, process=False)

            # Already in on_update of this record, so disable processing to
            # avoid recursion
            self._mode_record_info.record.set(TableModeEnum.VIEW.value, process=False)

    def update_table(self, new_values: List[str]) -> None:
        """Update the waveform records with the given values from the PandA, depending
        on the value of the table's MODE record.
        Note: This is NOT a method called through a record's `on_update`.

        Args:
            new_values: The list of new values from the PandA
        """

        curr_mode = TableModeEnum(self._mode_record_info.record.get())

        if curr_mode == TableModeEnum.VIEW:
            assert self.field_info.row_words
            field_data = TablePacking.unpack(
                self.field_info.row_words, self.table_fields, list(new_values)
            )

            table_field_records = self._get_table_field_records()
            for record_info, data in zip(table_field_records.values(), field_data):
                # Must skip processing as the validate method would reject the update
                record_info.record.set(data, process=False)
                self._update_scalar(record_info.record.name)
        else:
            # No other mode allows PandA updates to EPICS records
            logging.warning(
                f"Update of table {self.table_name} attempted when MODE "
                "was not VIEW. New value will be discarded"
            )

    async def update_index(self, new_val) -> None:
        """Update the SCALAR record for every column in the table based on the new
        index and/or new waveform data."""
        for record_info in self._get_table_field_records().values():
            self._update_scalar(record_info.record.name)

    def _update_scalar(self, waveform_record_name: str) -> None:
        """Update the column's SCALAR record based on the new index and/or new waveform
        data.

        Args:
            waveform_record_name: The name of the column record including leading
            namespace, e.g. "<namespace>:SEQ1:TABLE:POSITION"
        """

        # Remove namespace from record name
        _, waveform_record_name = waveform_record_name.split(":", maxsplit=1)

        waveform_record = self.table_records[EpicsName(waveform_record_name)].record
        waveform_data = waveform_record.get()

        scalar_record = self.table_records[
            EpicsName(waveform_record_name + ":SCALAR")
        ].record

        index_record_name = waveform_record_name.rsplit(":", maxsplit=1)[0] + ":INDEX"
        index_record = self.table_records[EpicsName(index_record_name)].record
        index = index_record.get()

        try:
            scalar_val = waveform_data[index]
            sev = alarm.NO_ALARM
        except IndexError as e:
            logging.warning(
                f"Index {index} of record {waveform_record_name} is out of bounds.",
                exc_info=e,
            )
            scalar_val = 0
            sev = alarm.INVALID_ALARM

        # alarm value is ignored if severity = NO_ALARM. Softioc also defaults
        # alarm value to UDF_ALARM, but I'm specifying it for clarity.
        scalar_record.set(scalar_val, severity=sev, alarm=alarm.UDF_ALARM)

    def _get_table_field_records(self) -> Dict[EpicsName, _RecordInfo]:
        """Filter the list of all table records to only return those of the table
        fields, removing all SCALAR, INDEX, and MODE records."""
        return {
            EpicsName(k): v
            for k, v in self.table_records.items()
            if not k.endswith(("SCALAR", "INDEX", "MODE"))
        }

    def set_mode_record_info(self, record_info: _RecordInfo) -> None:
        """Set the special MODE record that controls the behaviour of this table"""
        self._mode_record_info = record_info
