from pathlib import Path

import numpy as np
import pytest

from pandablocks.commands import Get, Put
from pandablocks.core import (
    ControlConnection,
    DataConnection,
    DataField,
    EndData,
    EndReason,
    FrameData,
    StartData,
)


def test_connection_gets_split_value():
    conn = ControlConnection()
    cmd = Get("PCAP.ACTIVE")
    assert conn.send(cmd) == b"PCAP.ACTIVE?\n"
    assert list(conn.receive_data(b"OK =1")) == []
    assert list(conn.receive_data(b"\nAnySpamWeLike")) == [(cmd, b"1")]


def test_connection_gets_muliline():
    conn = ControlConnection()
    cmd = Get("SEQ1.TABLE")
    assert conn.send(cmd) == b"SEQ1.TABLE?\n"
    assert list(conn.receive_data(b"!1048576\n!0\n!10")) == []
    assert list(conn.receive_data(b"00\n!1000\n.\n")) == [
        (cmd, [b"1048576", b"0", b"1000", b"1000"])
    ]


def test_connect_put_single_line():
    conn = ControlConnection()
    cmd = Put("PCAP.TRIG", "PULSE1.OUT")
    assert conn.send(cmd) == b"PCAP.TRIG=PULSE1.OUT\n"
    assert list(conn.receive_data(b"OK\n")) == [(cmd, None)]


def test_connect_put_multi_line():
    conn = ControlConnection()
    cmd = Put("SEQ1.TABLE", ["1048576", "0", "1000", "1000"])
    assert conn.send(cmd) == b"SEQ1.TABLE<\n1048576\n0\n1000\n1000\n\n"
    assert list(conn.receive_data(b"OK\n")) == [(cmd, None)]


DUMP_FIELDS = [
    DataField(
        name="PCAP.BITS2", type=np.uint32, capture="Value", scale=1, offset=0, units="",
    ),
    DataField(
        name="COUNTER1.OUT", type=np.double, capture="Min", scale=1, offset=0, units="",
    ),
    DataField(
        name="COUNTER1.OUT", type=np.double, capture="Max", scale=1, offset=0, units="",
    ),
    DataField(
        name="COUNTER3.OUT",
        type=np.double,
        capture="Value",
        scale=1,
        offset=0,
        units="",
    ),
    DataField(
        name="PCAP.TS_START",
        type=np.double,
        capture="Value",
        scale=8e-09,
        offset=0,
        units="s",
    ),
    DataField(
        name="COUNTER1.OUT",
        type=np.double,
        capture="Mean",
        scale=1,
        offset=0,
        units="",
    ),
    DataField(
        name="COUNTER2.OUT",
        type=np.double,
        capture="Mean",
        scale=1,
        offset=0,
        units="",
    ),
]


class Rows:
    def __init__(self, *rows):
        self.rows = rows

    def __eq__(self, o):
        same = o.tolist() == [pytest.approx(row) for row in self.rows]
        return same


def test_slow_data_collection():
    conn = DataConnection()
    assert conn.connect() == b"XML FRAMED SCALED\n"
    with open(Path(__file__).parent / "slow_dump.txt", "rb") as f:
        events = list(conn.receive_data(f.read()))
    assert [
        StartData(DUMP_FIELDS, 0, "Scaled", "Framed", 52),
        FrameData(Rows([0, 1, 1, 3, 5.6e-08, 1, 2])),
        FrameData(Rows([8, 2, 2, 6, 1.000000056, 2, 4])),
        FrameData(Rows([0, 3, 3, 9, 2.000000056, 3, 6])),
        FrameData(Rows([8, 4, 4, 12, 3.000000056, 4, 8])),
        FrameData(Rows([0, 5, 5, 15, 4.000000056, 5, 10])),
        EndData(5, EndReason.DISARMED),
    ] == events


def test_fast_data_collection():
    conn = DataConnection()
    assert conn.connect() == b"XML FRAMED SCALED\n"
    with open(Path(__file__).parent / "fast_dump.txt", "rb") as f:
        events = list(conn.receive_data(f.read()))
    assert [
        StartData(DUMP_FIELDS, 0, "Scaled", "Framed", 52),
        FrameData(
            Rows(
                [0, 1, 1, 3, 5.6e-08, 1, 2],
                [0, 2, 2, 6, 0.010000056, 2, 4],
                [0, 3, 3, 9, 0.020000056, 3, 6],
                [0, 4, 4, 12, 0.030000056, 4, 8],
                [0, 5, 5, 15, 0.040000056, 5, 10],
                [0, 6, 6, 18, 0.050000056, 6, 12],
                [0, 7, 7, 21, 0.060000056, 7, 14],
                [0, 8, 8, 24, 0.070000056, 8, 16],
                [0, 9, 9, 27, 0.080000056, 9, 18],
                [0, 10, 10, 30, 0.090000056, 10, 20],
            )
        ),
        FrameData(
            Rows(
                [0, 11, 11, 33, 0.100000056, 11, 22],
                [0, 12, 12, 36, 0.110000056, 12, 24],
                [0, 13, 13, 39, 0.120000056, 13, 26],
                [0, 14, 14, 42, 0.130000056, 14, 28],
                [0, 15, 15, 45, 0.140000056, 15, 30],
                [0, 16, 16, 48, 0.150000056, 16, 32],
                [0, 17, 17, 51, 0.160000056, 17, 34],
                [0, 18, 18, 54, 0.170000056, 18, 36],
                [0, 19, 19, 57, 0.180000056, 19, 38],
                [0, 20, 20, 60, 0.190000056, 20, 40],
                [0, 21, 21, 63, 0.200000056, 21, 42],
            )
        ),
        FrameData(
            Rows(
                [0, 22, 22, 66, 0.210000056, 22, 44],
                [0, 23, 23, 69, 0.220000056, 23, 46],
                [0, 24, 24, 72, 0.230000056, 24, 48],
                [0, 25, 25, 75, 0.240000056, 25, 50],
                [0, 26, 26, 78, 0.250000056, 26, 52],
                [0, 27, 27, 81, 0.260000056, 27, 54],
                [0, 28, 28, 84, 0.270000056, 28, 56],
                [0, 29, 29, 87, 0.280000056, 29, 58],
                [0, 30, 30, 90, 0.290000056, 30, 60],
                [0, 31, 31, 93, 0.300000056, 31, 62],
            )
        ),
        FrameData(
            Rows(
                [0, 32, 32, 96, 0.310000056, 32, 64],
                [0, 33, 33, 99, 0.320000056, 33, 66],
                [0, 34, 34, 102, 0.330000056, 34, 68],
                [0, 35, 35, 105, 0.340000056, 35, 70],
                [0, 36, 36, 108, 0.350000056, 36, 72],
                [0, 37, 37, 111, 0.360000056, 37, 74],
                [0, 38, 38, 114, 0.370000056, 38, 76],
                [0, 39, 39, 117, 0.380000056, 39, 78],
                [0, 40, 40, 120, 0.390000056, 40, 80],
                [0, 41, 41, 123, 0.400000056, 41, 82],
            )
        ),
        FrameData(
            Rows(
                [0, 42, 42, 126, 0.410000056, 42, 84],
                [0, 43, 43, 129, 0.420000056, 43, 86],
                [0, 44, 44, 132, 0.430000056, 44, 88],
                [0, 45, 45, 135, 0.440000056, 45, 90],
                [0, 46, 46, 138, 0.450000056, 46, 92],
                [0, 47, 47, 141, 0.460000056, 47, 94],
                [0, 48, 48, 144, 0.470000056, 48, 96],
                [0, 49, 49, 147, 0.480000056, 49, 98],
                [0, 50, 50, 150, 0.490000056, 50, 100],
                [0, 51, 51, 153, 0.500000056, 51, 102],
            )
        ),
        FrameData(
            Rows(
                [0, 52, 52, 156, 0.510000056, 52, 104],
                [0, 53, 53, 159, 0.520000056, 53, 106],
                [0, 54, 54, 162, 0.530000056, 54, 108],
                [0, 55, 55, 165, 0.540000056, 55, 110],
                [0, 56, 56, 168, 0.550000056, 56, 112],
                [0, 57, 57, 171, 0.560000056, 57, 114],
                [0, 58, 58, 174, 0.570000056, 58, 116],
                [0, 59, 59, 177, 0.580000056, 59, 118],
                [0, 60, 60, 180, 0.590000056, 60, 120],
                [0, 61, 61, 183, 0.600000056, 61, 122],
            )
        ),
        EndData(61, EndReason.DISARMED),
    ] == events
    assert events[1].data["COUNTER1.OUT.Mean"] == pytest.approx(range(1, 11))
