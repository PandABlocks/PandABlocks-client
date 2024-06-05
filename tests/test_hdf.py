import queue
from pathlib import Path

import numpy as np

from pandablocks.hdf import Pipeline, create_default_pipeline, stop_pipeline
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
