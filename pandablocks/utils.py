from typing import Dict, List, Sequence, Union

import numpy as np
import numpy.typing as npt

from pandablocks.responses import TableFieldInfo

UnpackedArray = Union[
    npt.NDArray[np.int32], npt.NDArray[np.uint8], npt.NDArray[np.uint16]
]


def words_to_table(
    words: Sequence[str], table_field_info: TableFieldInfo
) -> Dict[str, List]:
    """Unpacks the given `packed` data based on the fields provided.
    Returns the unpacked data in {column_name: column_data} column-indexed format

    Args:
        words: The list of data for this table, from PandA. Each item is
            expected to be the string representation of a uint32.
        table_fields_info: The info for tables, containing the number of words per row,
            and the bit information for fields.
    Returns:
        unpacked: A dict of lists, one item per field.
    """

    row_words = table_field_info.row_words
    data = np.array(words, dtype=np.uint32)
    # Convert 1-D array into 2-D, one row element per row in the PandA table
    data = data.reshape(len(data) // row_words, row_words)
    packed = data.T

    # Ensure fields are in bit-order
    table_fields = dict(
        sorted(
            table_field_info.fields.items(),
            key=lambda item: item[1].bit_low,
        )
    )

    unpacked: Dict[str, List] = {}

    for field_name, field_info in table_fields.items():
        offset = field_info.bit_low
        bit_length = field_info.bit_high - field_info.bit_low + 1

        # The word offset indicates which column this field is in
        # (column is exactly one 32-bit word)
        word_offset = offset // 32

        # bit offset is location of field inside the word
        bit_offset = offset & 0x1F

        # Mask to remove every bit that isn't in the range we want
        mask = (1 << bit_length) - 1

        value: UnpackedArray = (packed[word_offset] >> bit_offset) & mask

        if field_info.subtype == "int":
            # First convert from 2's complement to offset, then add in offset.
            value = (value ^ (1 << (bit_length - 1))) + (-1 << (bit_length - 1))
            value = value.astype(np.int32)
        else:
            # Use shorter types, as these are used in waveform creation
            if bit_length <= 8:
                value = value.astype(np.uint8)
            elif bit_length <= 16:
                value = value.astype(np.uint16)

        value_list = value.tolist()
        # Convert back to label from integer
        if field_info.labels:
            value_list = [field_info.labels[x] for x in value_list]

        unpacked.update({field_name: value_list})

    return unpacked


def table_to_words(
    table: Dict[str, Sequence], table_field_info: TableFieldInfo
) -> List[str]:
    """Pack the records based on the field definitions into the format PandA expects
    for table writes.

    Args:
        row_words: The number of 32-bit words per row
        table_fields_info: The info for tables, containing the number of words per row,
            and the bit information for fields.
    Returns:
        List[str]: The list of data ready to be sent to PandA
    """
    row_words = table_field_info.row_words

    # Ensure fields are in bit-order
    table_fields = dict(
        sorted(
            table_field_info.fields.items(),
            key=lambda item: item[1].bit_low,
        )
    )

    # Iterate over the zipped fields and their associated records to construct the
    # packed array.
    packed = None

    for column_name, column in table.items():
        field_details = table_fields[column_name]
        if field_details.labels:
            # Must convert the list of ints into strings
            column = [field_details.labels.index(x) for x in column]

        # PandA always handles tables in uint32 format
        column_value = np.uint32(np.array(column))

        if packed is None:
            # Create 1-D array sufficiently long to exactly hold the entire table
            packed = np.zeros((len(column), row_words), dtype=np.uint32)
        else:
            assert len(packed) == len(column), (
                f"Table record {column_name} has mismatched length "
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
        packed[:, word_offset] |= column_value << bit_offset

    assert isinstance(packed, np.ndarray)  # Squash mypy warning

    # 2-D array -> 1-D array -> list[int] -> list[str]
    return [str(x) for x in packed.flatten().tolist()]
