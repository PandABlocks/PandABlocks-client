import queue
from pathlib import Path

import numpy as np
import pytest

from pandablocks.hdf import HDFWriter, Pipeline, create_default_pipeline, stop_pipeline
from pandablocks.responses import EndData, EndReason, FieldCapture, FrameData, StartData


def test_pipeline_returns_number_written(tmp_path):
    NUMBER_OF_FRAMES_WRITTEN = 10000

    num_written_queue = queue.Queue()

    class DummyDownstream(Pipeline):
        def __init__(self):
            self.what_to_do = {int: num_written_queue.put_nowait}
            super().__init__()

    file_counter = DummyDownstream()

    try:
        pipeline = create_default_pipeline(
            iter([Path(tmp_path / "1.h5")]), {}, file_counter
        )

        pipeline[0].queue.put_nowait(
            StartData(
                [
                    FieldCapture(
                        name="COUNTER1.OUT",
                        type=np.dtype("float64"),
                        capture="Value",
                        scale=1,
                        offset=0,
                        units="",
                    )
                ],
                0,
                "Scaled",
                "Framed",
                52,
                "2024-03-05T20:27:12.607841574Z",
                "2024-03-05T20:27:12.608875498Z",
                100555,
            ),
        )
        pipeline[0].queue.put_nowait(
            FrameData(
                np.array(
                    NUMBER_OF_FRAMES_WRITTEN * [(1,)],
                    dtype=[("COUNTER1.OUT.Value", "<f8")],
                )
            )
        )
        pipeline[0].queue.put_nowait(EndData(5, EndReason.DISARMED))

        assert num_written_queue.get() == NUMBER_OF_FRAMES_WRITTEN
    finally:
        stop_pipeline(pipeline)


@pytest.mark.parametrize(
    "capture_record_hdf_names,expected_names",
    [
        (
            {},
            {
                "/COUNTER1.OUT.Value",
                "/COUNTER2.OUT.Mean",
                "/COUNTER2.OUT.Max",
                "/COUNTER2.OUT.Min",
            },
        ),
        (
            {"COUNTER1.OUT": {"Value": "scientific-name"}},
            {
                "/scientific-name",
                "/COUNTER2.OUT.Mean",
                "/COUNTER2.OUT.Max",
                "/COUNTER2.OUT.Min",
            },
        ),
        (
            {
                "COUNTER2.OUT": {
                    "Mean": "scientific-name",
                    "Max": "scientific-name-max",
                    "Min": "scientific-name-min",
                }
            },
            {
                "/COUNTER1.OUT.Value",
                "/scientific-name",
                "/scientific-name-max",
                "/scientific-name-min",
            },
        ),
        (
            {
                "COUNTER1.OUT": {"Value": "scientific-name1"},
                "COUNTER2.OUT": {
                    "Mean": "scientific-name2",
                    "Max": "scientific-name2-max",
                    "Min": "scientific-name2-min",
                },
            },
            {
                "/scientific-name1",
                "/scientific-name2",
                "/scientific-name2-max",
                "/scientific-name2-min",
            },
        ),
    ],
)
def test_hdf_writer_uses_alternative_dataset_names(
    tmp_path, capture_record_hdf_names, expected_names
):
    hdf_writer = HDFWriter(
        iter([str(tmp_path / "test_file.h5")]), capture_record_hdf_names
    )

    start_data = StartData(
        fields=[
            FieldCapture(
                name="COUNTER1.OUT",
                type=np.dtype("float64"),
                capture="Value",
                scale=1,
                offset=0,
                units="",
            ),
            FieldCapture(
                name="COUNTER2.OUT",
                type=np.dtype("float64"),
                capture="Min",
                scale=1,
                offset=0,
                units="",
            ),
            FieldCapture(
                name="COUNTER2.OUT",
                type=np.dtype("float64"),
                capture="Max",
                scale=1,
                offset=0,
                units="",
            ),
            FieldCapture(
                name="COUNTER2.OUT",
                type=np.dtype("float64"),
                capture="Mean",
                scale=1,
                offset=0,
                units="",
            ),
        ],
        missed=0,
        process="Scaled",
        format="Framed",
        sample_bytes=52,
        arm_time="2024-03-05T20:27:12.607841574Z",
        start_time="2024-03-05T20:27:12.608875498Z",
        hw_time_offset_ns=100555,
    )

    hdf_writer.open_file(start_data)

    assert {dataset.name for dataset in hdf_writer.datasets} == expected_names
