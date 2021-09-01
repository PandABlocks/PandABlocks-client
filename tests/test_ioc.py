import pytest

from pandablocks.asyncio import AsyncioClient
from pandablocks.ioc import IocRecordFactory
from pandablocks.responses import BitOutFieldInfo, PosOutFieldInfo, TimeFieldInfo

TEST_PREFIX = "TEST-PREFIX"
counter = 0


@pytest.fixture
def ioc_record_factory():
    global counter
    counter += 1
    # Ensure we pass a different test prefix each time, so each test uses its
    # own device name prefix
    return IocRecordFactory(TEST_PREFIX + str(counter), AsyncioClient("123"))


TEST_RECORD = "TEST:RECORD"

# TODO: A test for every known type-subtype pairing
@pytest.mark.parametrize(
    "field_info, values, expected_records",
    [
        (
            TimeFieldInfo(
                "time",
                units_labels=["s", "ms", "min"],
                min=8e-09,
            ),
            {
                f"{TEST_RECORD}": "0.1",
                f"{TEST_RECORD}:UNITS": "s",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:UNITS", f"{TEST_RECORD}:MIN"],
        ),
        (
            BitOutFieldInfo(
                "bit_out",
                capture_word="ABC.DEF",
                offset=10,
            ),
            {
                f"{TEST_RECORD}": "0",
            },
            [f"{TEST_RECORD}", f"{TEST_RECORD}:CAPTURE_WORD", f"{TEST_RECORD}:OFFSET"],
        ),
        (
            PosOutFieldInfo("pos_out", capture_labels=["No", "Diff"]),
            {
                f"{TEST_RECORD}": "0",
                f"{TEST_RECORD}:CAPTURE": "Diff",
                f"{TEST_RECORD}:OFFSET": "5",
                f"{TEST_RECORD}:SCALE": "0.5",
                f"{TEST_RECORD}:UNITS": "MyUnits",
            },
            [
                f"{TEST_RECORD}",
                f"{TEST_RECORD}:CAPTURE",
                f"{TEST_RECORD}:OFFSET",
                f"{TEST_RECORD}:SCALE",
                f"{TEST_RECORD}:UNITS",
            ],
        ),
    ],
)
def test_create_record(
    ioc_record_factory: IocRecordFactory, field_info, values, expected_records
):
    returned_records = ioc_record_factory.create_record(TEST_RECORD, field_info, values)
    assert len(returned_records) == len(expected_records)
    assert all(key in returned_records for key in expected_records)
