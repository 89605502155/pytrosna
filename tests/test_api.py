"""The high-level API: File, Transaction and module functions."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import pytrosna
from pytrosna import DeviceSchema, File
from pytrosna.errors import (
    InvalidArgumentError,
    PointNotFoundError,
    TypeMismatchError,
    UnknownAnnotationError,
    UnknownDeviceError,
)

ROOM = DeviceSchema.build(
    "room1", "s", {"temperature": "float64", "door": "bool"}, timezone="Europe/Moscow"
)


def room_frame(start: str = "2026-10-08 10:00", periods: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=periods, freq="10s", tz="Europe/Moscow", unit="s"),
            "temperature": [21.5, 21.6, 21.7, 21.6, 21.8][:periods],
            "door": [False] * periods,
        }
    )


@pytest.fixture
def room(path: Path) -> File:
    pytrosna.write(path, room_frame(), "room1", message="import", author="tests")
    return pytrosna.open(path)


def test_write_and_read(room: File) -> None:
    assert room.finalized
    assert room.format_version == (1, 0)
    assert list(room.devices) == ["room1"]
    assert room.device().timezone == "Europe/Moscow"
    df = room.read_pandas()
    assert df["temperature"].tolist() == [21.5, 21.6, 21.7, 21.6, 21.8]
    assert str(df["time"].dt.tz) == "Europe/Moscow"
    assert room.count() == 5
    assert room.count(start="2026-10-08 10:00:15") == 3
    assert room.count(start="2026-10-08 10:00:15", end="2026-10-08 10:00:30") == 2
    head = room.head
    assert head is not None
    assert (head.message, head.author) == ("import", "tests")
    assert repr(room) == f"pytrosna.File({room.path!r}, devices=['room1'])"


def test_read_variants(room: File) -> None:
    batch = room.read(columns=["temperature"], start="2026-10-08 10:00:10")
    assert len(batch) == 4
    assert room.read_arrow().num_rows == 5
    assert room.read_polars(end="2026-10-08 10:00:10").height == 2
    reader = room.read_batches(columns=["temperature"])
    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema.names == ["time", "temperature"]
    assert reader.read_all().num_rows == 5
    assert sum(len(b) for b in room.iter_batches()) == 5
    assert sum(len(b) for b in pytrosna.iter_batches(room.path)) == 5
    assert pytrosna.read(room.path).time.size == 5
    assert pytrosna.read_batches(room.path).read_all().num_rows == 5
    assert room.time_range() == (
        room.to_raw("2026-10-08 10:00"),
        room.to_raw("2026-10-08 10:00:40"),
    )
    assert room.to_datetime(room.to_raw("2026-10-08 10:00")).hour == 10
    nullable = room.read_pandas(dtype_backend="numpy_nullable")
    assert str(nullable["temperature"].dtype) == "Float64"


def test_edit_transaction(room: File) -> None:
    with room.edit(message="door was open", author="analyst") as tx:
        tx.update("room1", "2026-10-08 10:00:20", temperature=21.65)
        tx.delete("room1", "2026-10-08 10:00:10")
        tx.insert("room1", "2026-10-08 10:01:00", temperature=22.0, door=True)
        tx.annotate(
            "room1", "2026-10-08 10:00:15", "2026-10-08 10:00:25", "door open", "ventilation"
        )
        tx.set_metadata("site", "Брянск")
    assert tx.commit is not None
    assert tx.commit.number == 2
    assert tx.annotation_ids == [1]
    df = room.read_pandas()
    assert df["temperature"].tolist() == [21.5, 21.65, 21.6, 21.8, 22.0]
    assert df["door"].tolist() == [False, False, False, False, True]
    assert room.metadata == {"site": "Брянск"}
    assert room.read_pandas(as_of=1)["temperature"].tolist() == [21.5, 21.6, 21.7, 21.6, 21.8]
    changes = room.diff(1)
    assert [p.kind.value for p in changes.points] == ["removed", "changed", "added"]
    assert [a.label for a in room.annotations()] == ["door open"]
    assert room.annotations("room1", as_of=1) == []
    assert [c.number for c in room.commits()] == [1, 2]
    assert room.commit(2).author == "analyst"


def test_failed_transactions_change_nothing(room: File) -> None:
    size = os.path.getsize(room.path)
    with pytest.raises(RuntimeError), room.edit() as tx:
        tx.delete("room1", "2026-10-08 10:00:10")
        raise RuntimeError
    with pytest.raises(PointNotFoundError), room.edit() as tx:
        tx.delete("room1", "2026-10-08 10:00:10")
        tx.update("room1", "2026-10-08 11:00", temperature=1.0)
    assert os.path.getsize(room.path) == size
    assert room.count() == 5
    assert pytrosna.verify(room.path).ok
    with room.edit() as tx:
        pass
    assert tx.commit is None


def test_transaction_validates_times_and_values(room: File) -> None:
    with room.edit() as tx:
        with pytest.raises(InvalidArgumentError, match="more precise"):
            tx.insert("room1", "2026-10-08 10:00:00.5", temperature=1.0)
        with pytest.raises(UnknownDeviceError):
            tx.insert("nope", 1, temperature=1.0)
    with pytest.raises(TypeMismatchError), room.edit() as tx:
        tx.insert("room1", "2026-10-08 10:05", temperature="hot")


def test_annotation_editing(room: File) -> None:
    with room.edit() as tx:
        tx.annotate("room1", "2026-10-08 10:00", "2026-10-08 10:00:10", "a")
        tx.annotate("room1", dt.datetime(2026, 10, 8, 10, 0, 20), "2026-10-08 10:00:20", "b")
    with room.edit() as tx:
        tx.update_annotation(1, label="renamed", end="2026-10-08 10:00:30")
        tx.remove_annotation(2)
    annotations = room.annotations()
    assert [(a.id, a.label) for a in annotations] == [(1, "renamed")]
    assert annotations[0].end == room.to_raw("2026-10-08 10:00:30")
    with room.edit() as tx, pytest.raises(UnknownAnnotationError):
        tx.update_annotation(7, label="x")


def test_transaction_write_and_delete_range(room: File) -> None:
    with room.edit(message="bulk") as tx:
        tx.delete_range("room1", "2026-10-08 10:00:10", "2026-10-08 10:00:30")
        tx.write("room1", room_frame("2026-10-08 11:00", 2))
        tx.remove_metadata("missing")
    assert room.count() == 4


def test_modes(path: Path) -> None:
    pytrosna.write(path, room_frame(), "room1", mode="x")
    with pytest.raises(FileExistsError):
        pytrosna.write(path, room_frame(), "room1", mode="x")
    pytrosna.write(path, room_frame("2026-10-08 12:00", 2), "room1", mode="a")
    assert pytrosna.open(path).count() == 7
    pytrosna.write(path, room_frame("2026-10-08 13:00", 1), "room1")  # "w" replaces the file
    assert pytrosna.open(path).count() == 1
    assert len(pytrosna.open(path).commits()) == 1
    with pytest.raises(InvalidArgumentError, match="mode"):
        pytrosna.write(path, room_frame(), "room1", mode="r")


def test_failed_write_leaves_the_file_untouched(room: File) -> None:
    before = Path(room.path).read_bytes()
    bad = pd.DataFrame({"time": [1, 2], "temperature": ["a", "b"]})
    with pytest.raises(TypeMismatchError):
        pytrosna.write(room.path, bad, "room1", unit="s", mode="a")
    with pytest.raises(pytrosna.UnsupportedError):  # mode "w" would replace the file
        pytrosna.write(room.path, pd.DataFrame({"time": [1], "v": [object()]}), "room1", unit="s")
    assert Path(room.path).read_bytes() == before
    assert not [p for p in os.listdir(Path(room.path).parent) if p.endswith(".tmp")]


def test_append_localizes_naive_times(room: File) -> None:
    naive = pd.DataFrame({"time": pd.to_datetime(["2026-10-08 10:02:00"]), "temperature": [20.0]})
    room.write(naive, "room1")
    assert room.read_pandas()["time"].iloc[-1] == pd.Timestamp(
        "2026-10-08 10:02", tz="Europe/Moscow"
    )


def test_integer_times_need_a_unit(path: Path) -> None:
    with pytest.raises(pytrosna.UnsupportedError, match="unit"):
        pytrosna.write(path, {"time": [1, 2], "v": [1.0, 2.0]}, "d")
    pytrosna.write(
        path, {"time": [1, 2], "v": [1.0, 2.0]}, "d", unit="s", codec="lz4", rows_per_block=1
    )
    f = pytrosna.open(path)
    assert f.device().time_unit.label == "s"
    assert len(f.blocks()) == 2
    assert f.blocks()[0].segments[1].codec.label in ("none", "lz4")


def test_create(path: Path) -> None:
    f = pytrosna.create(path, ROOM, metadata={"site": "lab"}, message="setup")
    assert f.devices == {"room1": ROOM}
    assert f.metadata == {"site": "lab"}
    assert f.count() == 0
    assert f.time_range() is None
    with pytest.raises(FileExistsError):
        pytrosna.create(path, ROOM)
    pytrosna.create(path, overwrite=True)
    assert pytrosna.open(path).devices == {}


def test_device_selection(path: Path) -> None:
    pytrosna.create(path, ROOM, DeviceSchema.build("room2", "s", {"x": "int64"}))
    f = pytrosna.open(path)
    with pytest.raises(InvalidArgumentError, match="several devices"):
        f.device()
    with pytest.raises(UnknownDeviceError):
        f.device("room3")
    assert f.device("room2").name == "room2"
    pytrosna.create(path, overwrite=True)
    with pytest.raises(KeyError, match="no devices"):
        pytrosna.open(path).device()


def test_history_helpers(room: File) -> None:
    with room.edit() as tx:
        tx.delete("room1", "2026-10-08 10:00")
    assert room.commit_at(dt.datetime(2000, 1, 1, tzinfo=dt.UTC)) == 0
    assert room.commit_at(dt.datetime.now(dt.UTC) + dt.timedelta(days=1)) == 2
    assert [b.rows for b in room.blocks()] == [5]


def test_maintenance(room: File, tmp_path: Path) -> None:
    assert room.verify().ok
    assert pytrosna.recover(room.path).was_finalized
    compacted = room.compact(tmp_path / "c.trosna")
    assert compacted.count() == 5
    report = pytrosna.compact(room.path, tmp_path / "c.trosna", overwrite=True)
    assert report.rows == 5


def test_writer_access(room: File) -> None:
    with room.writer(sync=False) as w:
        w.write_row("room1", room.to_raw("2026-10-08 11:00"), temperature=1.0)
    room.reload()
    assert room.count() == 6


def test_module_exports() -> None:
    assert pytrosna.EXTENSION == "trosna"
    assert pytrosna.FORMAT_VERSION == (1, 0)
    assert pytrosna.__version__ == "0.1.0"
    for name in pytrosna.__all__:
        assert hasattr(pytrosna, name), name


def test_numpy_mapping_round_trip(path: Path) -> None:
    rng = np.random.default_rng(0)
    data = {
        "time": np.arange(1000, dtype=np.int64),
        "x": rng.normal(size=1000),
        "n": rng.integers(0, 3, 1000),
    }
    pytrosna.write(path, data, "d", unit="us")
    batch = pytrosna.read(path)
    assert np.array_equal(batch.time, data["time"])
    assert np.array_equal(batch["x"].values, data["x"])
    assert np.array_equal(batch["n"].values, data["n"])
