import h5py
import pytest

from pandablocks.hdf import write_hdf_files


@pytest.mark.asyncio
async def test_writing_fast_hdf(dummy_server, fast_dump, tmp_path):
    dummy_server.data = fast_dump
    await write_hdf_files("localhost", str(tmp_path / "%d.h5"), 1)
    hdf_file = h5py.File(tmp_path / "1.h5", "r")
    assert hdf_file["/COUNTER1.OUT.Min"][:] == pytest.approx(range(1, 59))
