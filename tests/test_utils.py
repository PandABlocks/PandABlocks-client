from typing import Dict, List, OrderedDict

import numpy as np
import pytest

from pandablocks.responses import TableFieldDetails, TableFieldInfo
from pandablocks.utils import UnpackedArray, table_to_words, words_to_table


@pytest.fixture
def table_fields() -> Dict[str, TableFieldDetails]:
    """Table field definitions, taken from a SEQ.TABLE instance.
    Associated with table_data and table_field_info fixtures"""
    return {
        "REPEATS": TableFieldDetails(
            subtype="uint",
            bit_low=0,
            bit_high=15,
            description="Number of times the line will repeat",
            labels=None,
        ),
        "TRIGGER": TableFieldDetails(
            subtype="enum",
            bit_low=16,
            bit_high=19,
            description="The trigger condition to start the phases",
            labels=[
                "Immediate",
                "BITA=0",
                "BITA=1",
                "BITB=0",
                "BITB=1",
                "BITC=0",
                "BITC=1",
                "POSA>=POSITION",
                "POSA<=POSITION",
                "POSB>=POSITION",
                "POSB<=POSITION",
                "POSC>=POSITION",
                "POSC<=POSITION",
            ],
        ),
        "POSITION": TableFieldDetails(
            subtype="int",
            bit_low=32,
            bit_high=63,
            description="The position that can be used in trigger condition",
            labels=None,
        ),
        "TIME1": TableFieldDetails(
            subtype="uint",
            bit_low=64,
            bit_high=95,
            description="The time the optional phase 1 should take",
            labels=None,
        ),
        "OUTA1": TableFieldDetails(
            subtype="uint",
            bit_low=20,
            bit_high=20,
            description="Output A value during phase 1",
            labels=None,
        ),
        "OUTB1": TableFieldDetails(
            subtype="uint",
            bit_low=21,
            bit_high=21,
            description="Output B value during phase 1",
            labels=None,
        ),
        "OUTC1": TableFieldDetails(
            subtype="uint",
            bit_low=22,
            bit_high=22,
            description="Output C value during phase 1",
            labels=None,
        ),
        "OUTD1": TableFieldDetails(
            subtype="uint",
            bit_low=23,
            bit_high=23,
            description="Output D value during phase 1",
            labels=None,
        ),
        "OUTE1": TableFieldDetails(
            subtype="uint",
            bit_low=24,
            bit_high=24,
            description="Output E value during phase 1",
            labels=None,
        ),
        "OUTF1": TableFieldDetails(
            subtype="uint",
            bit_low=25,
            bit_high=25,
            description="Output F value during phase 1",
            labels=None,
        ),
        "TIME2": TableFieldDetails(
            subtype="uint",
            bit_low=96,
            bit_high=127,
            description="The time the mandatory phase 2 should take",
            labels=None,
        ),
        "OUTA2": TableFieldDetails(
            subtype="uint",
            bit_low=26,
            bit_high=26,
            description="Output A value during phase 2",
            labels=None,
        ),
        "OUTB2": TableFieldDetails(
            subtype="uint",
            bit_low=27,
            bit_high=27,
            description="Output B value during phase 2",
            labels=None,
        ),
        "OUTC2": TableFieldDetails(
            subtype="uint",
            bit_low=28,
            bit_high=28,
            description="Output C value during phase 2",
            labels=None,
        ),
        "OUTD2": TableFieldDetails(
            subtype="uint",
            bit_low=29,
            bit_high=29,
            description="Output D value during phase 2",
            labels=None,
        ),
        "OUTE2": TableFieldDetails(
            subtype="uint",
            bit_low=30,
            bit_high=30,
            description="Output E value during phase 2",
            labels=None,
        ),
        "OUTF2": TableFieldDetails(
            subtype="uint",
            bit_low=31,
            bit_high=31,
            description="Output F value during phase 2",
            labels=None,
        ),
    }


@pytest.fixture
def table_field_info(table_fields) -> TableFieldInfo:
    """Table data associated with table_fields and table_data fixtures"""
    return TableFieldInfo(
        "table", None, "Sequencer table of lines", 16384, table_fields, 4
    )


@pytest.fixture
def table_1_np_arrays() -> OrderedDict[str, UnpackedArray]:
    # Intentionally not in panda order. Whatever types the np arrays are,
    # the outputs from words_to_table will be uint32 or int32.
    return OrderedDict(
        {
            "REPEATS": np.array([5, 0, 50000], dtype=np.uint32),
            "TRIGGER": ["Immediate", "BITC=1", "Immediate"],
            "POSITION": np.array([-5, 678, 0], dtype=np.int32),
            "TIME1": np.array([100, 0, 9], dtype=np.uint32),
            "OUTA1": np.array([0, 1, 1], dtype=np.uint8),
            "OUTB1": np.array([0, 0, 1], dtype=np.uint8),
            "OUTD1": np.array([1, 0, 1], dtype=np.uint8),
            "OUTE1": np.array([0, 0, 1], dtype=np.uint8),
            "OUTC1": np.array([0, 0, 1], dtype=np.uint8),
            "OUTF1": np.array([1, 0, 1], dtype=np.uint8),
            "TIME2": np.array([0, 55, 9999], dtype=np.uint32),
            "OUTA2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTB2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTC2": np.array([1, 1, 1], dtype=np.uint8),
            "OUTD2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTE2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTF2": np.array([1, 0, 1], dtype=np.uint8),
        }
    )


@pytest.fixture
def table_1_np_arrays_int_enums() -> OrderedDict[str, UnpackedArray]:
    # Intentionally not in panda order. Whatever types the np arrays are,
    # the outputs from words_to_table will be uint32 or int32.
    return OrderedDict(
        {
            "POSITION": np.array([-5, 678, 0], dtype=np.int32),
            "TIME1": np.array([100, 0, 9], dtype=np.uint32),
            "OUTA1": np.array([0, 1, 1], dtype=np.uint8),
            "OUTB1": np.array([0, 0, 1], dtype=np.uint8),
            "OUTD1": np.array([1, 0, 1], dtype=np.uint8),
            "OUTE1": np.array([0, 0, 1], dtype=np.uint8),
            "OUTC1": np.array([0, 0, 1], dtype=np.uint8),
            "OUTF1": np.array([1, 0, 1], dtype=np.uint8),
            "TIME2": np.array([0, 55, 9999], dtype=np.uint32),
            "OUTA2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTB2": np.array([0, 0, 1], dtype=np.uint8),
            "REPEATS": np.array([5, 0, 50000], dtype=np.uint32),
            "OUTC2": np.array([1, 1, 1], dtype=np.uint8),
            "OUTD2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTE2": np.array([0, 0, 1], dtype=np.uint8),
            "OUTF2": np.array([1, 0, 1], dtype=np.uint8),
            "TRIGGER": np.array([0, 6, 0], dtype=np.uint8),
        }
    )


@pytest.fixture
def table_data_1() -> List[str]:
    return [
        "2457862149",
        "4294967291",
        "100",
        "0",
        "269877248",
        "678",
        "0",
        "55",
        "4293968720",
        "0",
        "9",
        "9999",
    ]


@pytest.fixture
def table_2_np_arrays() -> Dict[str, UnpackedArray]:
    table: Dict[str, UnpackedArray] = {
        "REPEATS": np.array([1, 0], dtype=np.uint32),
        "TRIGGER": ["Immediate", "Immediate"],
        "POSITION": np.array([-20, 2**31 - 1], dtype=np.int32),
        "TIME1": np.array([12, 2**32 - 1], dtype=np.uint32),
        "TIME2": np.array([32, 1], dtype=np.uint32),
    }

    table["OUTA1"] = np.array([0, 1], dtype=np.uint8)
    table["OUTA2"] = np.array([1, 0], dtype=np.uint8)
    for key in "BCDEF":
        table[f"OUT{key}1"] = table[f"OUT{key}2"] = np.array([0, 0], dtype=np.uint8)

    return table


@pytest.fixture
def table_data_2() -> List[str]:
    return [
        "67108865",
        "4294967276",
        "12",
        "32",
        "1048576",
        "2147483647",
        "4294967295",
        "1",
    ]


def test_table_packing_pack_length_mismatched(
    table_1_np_arrays: OrderedDict[str, UnpackedArray],
    table_field_info: TableFieldInfo,
):
    assert table_field_info.row_words

    # Adjust one of the record lengths so it mismatches
    field_info = table_field_info.fields[("OUTC1")]
    assert field_info
    table_1_np_arrays["OUTC1"] = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    with pytest.raises(AssertionError):
        table_to_words(table_1_np_arrays, table_field_info)


@pytest.mark.parametrize(
    "table_fixture_name,table_data_fixture_name",
    [
        ("table_2_np_arrays", "table_data_2"),
        ("table_1_np_arrays", "table_data_1"),
    ],
)
def test_table_to_words_and_words_to_table(
    table_fixture_name: str,
    table_data_fixture_name: str,
    table_field_info: TableFieldInfo,
    request,
):
    table: Dict[str, UnpackedArray] = request.getfixturevalue(table_fixture_name)
    table_data: List[str] = request.getfixturevalue(table_data_fixture_name)

    output_data = table_to_words(table, table_field_info)
    assert output_data == table_data
    output_table = words_to_table(
        output_data, table_field_info, convert_enum_indices=True
    )

    # Test the correct keys are outputted
    assert output_table.keys() == table.keys()

    # Check the items have been inserted in panda order
    sorted_output_table = OrderedDict({key: table[key] for key in output_table.keys()})
    assert sorted_output_table != OrderedDict(table)

    # Check the values are the same
    for output_key in output_table.keys():
        np.testing.assert_equal(output_table[output_key], table[output_key])


def test_table_packing_unpack(
    table_1_np_arrays: OrderedDict[str, np.ndarray],
    table_field_info: TableFieldInfo,
    table_data_1: List[str],
):
    assert table_field_info.row_words
    output_table = words_to_table(
        table_data_1, table_field_info, convert_enum_indices=True
    )

    actual: UnpackedArray
    for field_name, actual in output_table.items():
        expected = table_1_np_arrays[str(field_name)]
        np.testing.assert_array_equal(actual, expected)


def test_table_packing_unpack_no_convert_enum(
    table_1_np_arrays_int_enums: OrderedDict[str, UnpackedArray],
    table_field_info: TableFieldInfo,
    table_data_1: List[str],
):
    assert table_field_info.row_words
    output_table = words_to_table(table_data_1, table_field_info)

    actual: UnpackedArray
    for field_name, actual in output_table.items():
        expected = table_1_np_arrays_int_enums[str(field_name)]
        np.testing.assert_array_equal(actual, expected)


def test_table_packing_pack(
    table_1_np_arrays: Dict[str, UnpackedArray],
    table_field_info: TableFieldInfo,
    table_data_1: List[str],
):
    assert table_field_info.row_words
    unpacked = table_to_words(table_1_np_arrays, table_field_info)

    for actual, expected in zip(unpacked, table_data_1):
        assert actual == expected


def test_table_packing_give_default_values(
    table_1_np_arrays: Dict[str, UnpackedArray],
    table_field_info: TableFieldInfo,
):
    # We should have a complete table at the point of unpacking
    table_1_np_arrays["TRIGGER"] = np.array([])

    with pytest.raises(AssertionError) as e:
        table_to_words(table_1_np_arrays, table_field_info)
        assert (
            "Table record TRIGGER has mismatched length 0 compared "
            "to other records 3" in str(e.value)
        )
