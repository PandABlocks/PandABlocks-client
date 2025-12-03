import queue
from pathlib import Path

import numpy as np
import pytest

from pandablocks.hdf import HDFWriter, Pipeline, create_default_pipeline, stop_pipeline
from pandablocks.responses import EndData, EndReason, FieldCapture, FrameData, StartData


def test_pipeline_returns_number_written(tmp_path):
    number_of_frames_written = 10000

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
                    number_of_frames_written * [(1,)],
                    dtype=[("COUNTER1.OUT.Value", "<f8")],
                )
            )
        )
        pipeline[0].queue.put_nowait(EndData(5, EndReason.DISARMED))

        assert num_written_queue.get() == number_of_frames_written
    finally:
        stop_pipeline(pipeline)


def test_field_capture_pcap_bits():
    pcap_bits_frame_data = FieldCapture(
        name="PCAP.BITS",
        type=np.dtype("uint32"),
        capture="Value",
        scale=None,
        offset=None,
        units=None,
    )

    assert not pcap_bits_frame_data.has_scale_or_offset
    assert pcap_bits_frame_data.raw_mode_dataset_dtype is np.dtype("uint32")

    frame_data_without_scale_offset = FieldCapture(
        name="frame_data_without_scale_offset",
        type=np.dtype("uint32"),
        capture="Value",
        scale=1.0,
        offset=0.0,
        units="",
    )

    assert not frame_data_without_scale_offset.has_scale_or_offset
    assert frame_data_without_scale_offset.raw_mode_dataset_dtype is np.dtype("float64")

    with pytest.raises(
        ValueError,
        match=(
            "If any of `scale=None`, `offset=0.0`, or `units=` is set, all must be set"
        ),
    ):
        _ = FieldCapture(
            name="malformed_frame_data",
            type=np.dtype("uint32"),
            capture="Value",
            scale=None,
            offset=0.0,
            units="",
        )

    frame_data_with_offset = FieldCapture(
        name="frame_data_with_offset",
        type=np.dtype("float64"),
        capture="Value",
        scale=1.0,
        offset=1.0,
        units="",
    )
    frame_data_with_scale = FieldCapture(
        name="frame_data_with_scale",
        type=np.dtype("float64"),
        capture="Value",
        scale=1.1,
        offset=0.0,
        units="",
    )

    assert frame_data_with_offset.has_scale_or_offset
    assert frame_data_with_offset.raw_mode_dataset_dtype is np.dtype("float64")
    assert frame_data_with_scale.has_scale_or_offset
    assert frame_data_with_scale.raw_mode_dataset_dtype is np.dtype("float64")

    frame_data_with_scale_and_offset = FieldCapture(
        name="frame_data_with_scale_and_offset",
        type=np.dtype("float64"),
        capture="Value",
        scale=1.1,
        offset=0.0,
        units="",
    )

    assert frame_data_with_scale_and_offset.has_scale_or_offset
    assert frame_data_with_scale_and_offset.raw_mode_dataset_dtype is np.dtype(
        "float64"
    )


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
