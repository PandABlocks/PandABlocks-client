import sys

import h5py
import pandas as pd

if __name__ == "__main__":
    with h5py.File(sys.argv[1], "r") as f:
        arm_time = f.attrs.get("arm_time", None)
        start_time = f.attrs.get("start_time", None)
        hw_time_offset_ns = f.attrs.get("hw_time_offset_ns", None)

        print(f"Arm time: {arm_time!r}")
        print(f"Start time: {start_time!r}")
        print(f"Hardware time offset: {hw_time_offset_ns!r} ns")

        if start_time:
            # Compute and print the start time that includes the offset
            ts_start = pd.Timestamp(start_time)
            if hw_time_offset_ns:
                ts_start += pd.Timedelta(nanoseconds=hw_time_offset_ns)
            print(f"Start TS including the offset: {ts_start}")


# Expected output:
#
# Arm time: '2024-03-05T20:27:12.607841574Z'
# Start time: '2024-03-05T20:27:12.605729480Z'
# Hardware time offset: 2155797 ns
# Start TS including the offset: 2024-03-05T20:27:08.605729480Z
