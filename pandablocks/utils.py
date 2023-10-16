from typing import Dict, Iterable, List, Union, cast

import numpy as np
import numpy.typing as npt

from pandablocks.responses import TableFieldInfo

UnpackedArray = Union[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint16],
    npt.NDArray[np.int32],
    npt.NDArray[np.uint32],
    List[str],
]


def words_to_table(
    words: Iterable[str],
    table_field_info: TableFieldInfo,
    convert_enum_indices: bool = False,
) -> Dict[str, UnpackedArray]:
    """Unpacks the given `packed` data based on the fields provided.
    Returns the unpacked data in {column_name: column_data} column-indexed format

    Args:
        words: An iterable of data for this table, from PandA. Each item is
            expected to be the string representation of a uint32.
        table_fields_info: The info for tables, containing the number of words per row,
            and the bit information for fields.
        convert_enum_indices: If True, convert all enum values to their string
            representation. Otherwise return enums as integer values
    Returns:
        unpacked: A dict containing record information, where keys are field names
            and values are numpy arrays or a sequence of strings of record values
            in that column.
    """

    row_words = table_field_info.row_words
    data = np.array(words, dtype=np.uint32)
    # Convert 1-D array into 2-D, one row element per row in the PandA table
    data = data.reshape(len(data) // row_words, row_words)
    packed = data.T

    unpacked: Dict[str, UnpackedArray] = {}

    for field_name, field_info in table_field_info.fields.items():
        offset = field_info.bit_low
        bit_length = field_info.bit_high - field_info.bit_low + 1

        # The word offset indicates which column this field is in
        # (column is exactly one 32-bit word)
        word_offset = offset // 32

        # bit offset is location of field inside the word
        bit_offset = offset & 0x1F

        # Mask to remove every bit that isn't in the range we want
        mask = (1 << bit_length) - 1

        value: npt.NDArray[np.uint32] = (packed[word_offset] >> bit_offset) & mask
        packing_value: UnpackedArray

        if field_info.subtype == "int":
            # First convert from 2's complement to offset, then add in offset.
            temp = (value ^ (1 << (bit_length - 1))) + (-1 << (bit_length - 1))
            packing_value = temp.astype(np.int32)
        elif field_info.subtype == "enum" and convert_enum_indices:
            assert field_info.labels, f"Enum field {field_name} has no labels"
            packing_value = [field_info.labels[x] for x in value]
        else:
            # Use shorter types, as these are used in waveform creation
            if bit_length <= 8:
                packing_value = value.astype(np.uint8)
            elif bit_length <= 16:
                packing_value = value.astype(np.uint16)
            else:
                packing_value = value.astype(np.uint32)  # already uint32

        unpacked.update({field_name: packing_value})

    return unpacked


def table_to_words(
    table: Dict[str, UnpackedArray], table_field_info: TableFieldInfo
) -> List[str]:
    """Convert records based on the field definitions into the format PandA expects
    for table writes.

    Args:
        table: A dict containing record information, where keys are field names
            and values are iterables of record values in that column.
        table_field_info: The info for tables, containing the dict `fields` for
            information on each field, and the number of words per row.
    Returns:
        List[str]: The list of data ready to be sent to PandA
    """
    row_words = table_field_info.row_words

    # Iterate over the zipped fields and their associated records to construct the
    # packed array.
    packed = None

    for column_name, column in table.items():
        field_details = table_field_info.fields[column_name]
        if field_details.labels and len(column) and isinstance(column[0], str):
            # Must convert the list of strings to list of ints
            column_value = np.array(
                [field_details.labels.index(x) for x in column], dtype=np.uint32
            )
        else:
            # PandA always handles tables in uint32 format
            column_value = np.array(column, dtype=np.uint32)

        if packed is None:
            # Create 1-D array sufficiently long to exactly hold the entire table, cast
            # to prevent type error, this will still work if column is another iterable
            # e.g numpy array
            packed = np.zeros((len(column), row_words), dtype=np.uint32)
        else:
            assert len(packed) == len(column), (
                f"Table record {column_name} has mismatched length {len(column)} "
                f"compared to other records {len(packed)}, cannot pack data. "
                "If setting values through the ioc, ensure all values are given "
                "before submitting."
            )

        offset = field_details.bit_low

        # The word offset indicates which column this field is in
        # (each column is one 32-bit word)
        word_offset = offset // 32
        # bit offset is location of field inside the word
        bit_offset = offset & 0x1F

        # Slice to get the column to apply the values to.
        # bit shift the value to the relevant bits of the word

        packed[:, word_offset] |= cast(np.unsignedinteger, column_value) << bit_offset

    assert isinstance(packed, np.ndarray), "Table has no columns"  # Squash mypy warning

    # 2-D array -> 1-D array -> list[int] -> list[str]
    return [str(x) for x in packed.flatten().tolist()]
