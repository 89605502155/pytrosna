"""Writer: creating, appending, editing, committing and recovering."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from pytrosna import (
    Batch,
    Codec,
    Column,
    DeviceSchema,
    Reader,
    WriteOptions,
    Writer,
)
from pytrosna.errors import (
    CorruptedError,
    DeviceExistsError,
    InvalidArgumentError,
    LockedError,
    NotTrosnaError,
    PointNotFoundError,
    SchemaError,
    TrosnaError,
    TypeMismatchError,
    UnknownAnnotationError,
    UnknownColumnError,
    UnknownDeviceError,
)
from pytrosna.writer import to_column

from .conftest import SCHEMA, sample_columns

SIMPLE = DeviceSchema.build("d", "ms", {"v": "int64", "s": "string"})


def read_all(path: Path, device: str = "d", **kw: object) -> Batch:
    with Reader(path) as r:
        return r.read(device, **kw)  # type: ignore[arg-type]


def test_create_write_read(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SCHEMA)
        assert w.write("vm01", sample_columns(100)) == 100
        commit = w.commit(message="m", author="a", time_ns=5)
        assert commit is not None
        assert (commit.number, commit.message, commit.author, commit.time_ns) == (1, "m", "a", 5)
        assert commit.changes.rows_written == 100
        assert w.head == 1
    batch = read_all(path, "vm01")
    expected = sample_columns(100)
    assert batch.time.tolist() == expected["time"].tolist()  # type: ignore[union-attr]
    assert batch["swap"].to_list() == expected["swap"]
    assert batch["state"].to_list() == expected["state"]


def test_create_refuses_existing_files(path: Path) -> None:
    Writer.create(path).close()
    with pytest.raises(FileExistsError):
        Writer.create(path)
    Writer.create(path, overwrite=True).close()
    with pytest.raises(InvalidArgumentError):
        Writer.create(path, WriteOptions(), overwrite=True)


def test_open_requires_a_trosna_file(tmp_path: Path) -> None:
    other = tmp_path / "x.csv"
    other.write_text("time,v\n")
    with pytest.raises(NotTrosnaError):
        Writer.open(other)
    with pytest.raises(FileNotFoundError):
        Writer.open(tmp_path / "missing.trosna")


def test_lock_prevents_two_writers(path: Path) -> None:
    w = Writer.create(path)
    with pytest.raises(LockedError):
        Writer.open(path)
    w.close()
    Writer.open(path).close()


@pytest.mark.parametrize(
    "options",
    [
        {"rows_per_block": 0},
        {"rows_per_block": (1 << 24) + 1},
        {"max_block_bytes": 0},
        {"codec": "brotli"},
        {"encoding": "best"},
    ],
)
def test_invalid_options(options: dict[str, object]) -> None:
    with pytest.raises(InvalidArgumentError):
        WriteOptions(**options)  # type: ignore[arg-type]


@pytest.mark.parametrize("codec", list(Codec))
def test_blocks_are_split(path: Path, codec: Codec) -> None:
    with Writer.create(path, rows_per_block=7, codec=codec, sync=False) as w:
        w.create_device(SCHEMA)
        w.write("vm01", sample_columns(30))
    with Reader(path) as r:
        blocks = r.blocks("vm01")
        assert [b.rows for b in blocks] == [7, 7, 7, 7, 2]
        assert len(r.read("vm01")) == 30


def test_byte_limit_flushes_buffers(path: Path) -> None:
    with Writer.create(path, max_block_bytes=200, sync=False) as w:
        w.create_device(SIMPLE)
        for i in range(10):
            w.write("d", {"time": [i * 10, i * 10 + 1], "v": [i, i], "s": ["x" * 40, "y"]})
    with Reader(path) as r:
        assert len(r.blocks("d")) > 1
        assert len(r.read("d")) == 20


def test_out_of_order_rows_and_duplicates(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write("d", {"time": [30, 10, 20, 10], "v": [3, 1, 2, 100]})
        w.write_row("d", 20, v=200)
    batch = read_all(path)
    assert batch.time.tolist() == [10, 20, 30]
    assert batch["v"].to_list() == [100, 200, 3]  # the last write of a time stamp wins
    assert batch["s"].to_list() == [None, None, None]


def test_write_accepts_batches_and_columns(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write("d", Batch([1, 2], {"v": Column.from_values("int64", [1, None])}))
        w.write("d", {"time": np.array([3]), "v": np.ma.masked_array([9], mask=[True])})
        w.write("d", {"time": [], "v": []})
        assert w.write("d", Batch([], {})) == 0
    assert read_all(path)["v"].to_list() == [1, None, None]


def test_write_type_errors(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        with pytest.raises(UnknownColumnError):
            w.write("d", {"time": [1], "x": [1]})
        with pytest.raises(UnknownDeviceError):
            w.write("e", {"time": [1]})
        with pytest.raises(TypeMismatchError):
            w.write("d", {"time": [1], "v": ["text"]})
        with pytest.raises(TypeMismatchError):
            w.write("d", {"time": [1.5], "v": [1]})
        with pytest.raises(InvalidArgumentError):
            w.write("d", {"time": [1, 2], "v": [1]})
        with pytest.raises(InvalidArgumentError):
            w.write("d", {"v": [1]})
        with pytest.raises(TypeMismatchError):
            w.write("d", Batch([1], {"v": Column.from_values("int32", [1])}))
        with pytest.raises(UnknownColumnError):
            w.write_row("d", 1, x=1)
        with pytest.raises(TypeMismatchError):
            w.write_row("d", 1, v="no")
        with pytest.raises(InvalidArgumentError):
            w.write_row("d", 2**63, v=1)


def test_to_column_conversions() -> None:
    assert to_column(np.array([1, 2], dtype=np.uint8), SIMPLE.types[0], "v").to_list() == [1, 2]
    assert to_column(np.array([1.0, 2.0]), SIMPLE.types[0], "v").to_list() == [1, 2]
    with pytest.raises(TypeMismatchError):
        to_column(np.array([1.5]), SIMPLE.types[0], "v")
    with pytest.raises(TypeMismatchError):
        to_column(np.array([2**63], dtype=np.uint64), SIMPLE.types[0], "v")
    with pytest.raises(TypeMismatchError):
        to_column(np.array(["a"]), SIMPLE.types[0], "v")
    assert to_column(np.array(["a", "b"]), SIMPLE.types[1], "s").to_list() == ["a", "b"]
    assert to_column(np.array([b"a"]), SIMPLE.types[1], "s").to_list() == ["a"]
    with pytest.raises(TypeMismatchError):
        to_column(np.array([1]), SIMPLE.types[1], "s")
    bool_type = SCHEMA.columns[3].data_type
    assert to_column(np.array([True]), bool_type, "ok").to_list() == [True]
    with pytest.raises(TypeMismatchError):
        to_column(np.array([1]), bool_type, "ok")
    float_type = SCHEMA.columns[0].data_type
    assert to_column(np.array([1, 2]), float_type, "cpu").to_list() == [1.0, 2.0]
    with pytest.raises(TypeMismatchError):
        to_column(np.array([True]), float_type, "cpu")
    with pytest.raises(InvalidArgumentError):
        to_column(np.zeros((1, 1)), float_type, "cpu")
    nan = to_column(np.array([math.nan]), float_type, "cpu")
    assert nan.null_count == 0


def test_devices(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        with pytest.raises(DeviceExistsError):
            w.create_device(SIMPLE)
        with pytest.raises(SchemaError):
            w.create_device(DeviceSchema("bad", "s"))
        assert w.ensure_device(SIMPLE) == SIMPLE
        other = DeviceSchema.build("e", "s", {"x": "bool"})
        assert w.ensure_device(other) == other
        with pytest.raises(SchemaError):
            w.ensure_device(DeviceSchema.build("d", "s", {"v": "int64", "s": "string"}))
        assert [d.name for d in w.devices] == ["d", "e"]
        assert w.has_device("e")
        assert w.device("e") == other


def test_update_get_and_delete(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write("d", {"time": [1, 2, 3], "v": [10, 20, 30], "s": ["a", "b", "c"]})
        w.commit()
        assert w.get("d", 2) == {"v": 20, "s": "b"}
        w.update("d", 2, v=21)  # from the file
        assert w.get("d", 2) == {"v": 21, "s": "b"}  # from the buffer
        w.update("d", 2, {"s": None})
        assert w.get("d", 2) == {"v": 21, "s": None}
        w.delete("d", 3)
        assert w.get("d", 3) is None  # pending tombstone
        with pytest.raises(PointNotFoundError):
            w.update("d", 3, v=1)
        w.write_row("d", 3, v=33)  # a point written after a deletion is visible again
        w.delete_range("d", 5, 4)  # empty range: nothing happens
        w.commit()
        assert w.get("d", 1) == {"v": 10, "s": "a"}
        assert w.get("d", 99) is None
    batch = read_all(path)
    assert batch.time.tolist() == [1, 2, 3]
    assert batch["v"].to_list() == [10, 21, 33]
    assert batch["s"].to_list() == ["a", None, None]


def test_delete_range_open_ends(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write("d", {"time": list(range(10)), "v": list(range(10))})
        w.commit()
        w.delete_range("d", None, 2)
        w.delete_range("d", 8, None)
    assert read_all(path).time.tolist() == [3, 4, 5, 6, 7]


def test_annotations(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        first = w.annotate("d", 1, 5, "spike", "note")
        second = w.annotate("d", 7, 7, "event")
        assert (first, second) == (1, 2)
        w.update_annotation(first, label="peak")
        w.update_annotation(first, end=6, note="")
        w.remove_annotation(second)
        assert w.annotations() == {1: ("d", 1, 6, "peak", None)}
        with pytest.raises(UnknownAnnotationError):
            w.remove_annotation(second)
        with pytest.raises(UnknownAnnotationError):
            w.update_annotation(99, label="x")
        with pytest.raises(InvalidArgumentError):
            w.annotate("d", 5, 1, "backwards")
        with pytest.raises(InvalidArgumentError):
            w.update_annotation(first, start=100)
        assert w.annotate("d", 0, 0, "after removal") == 3
    with Reader(path) as r:
        assert [(a.id, a.label) for a in r.annotations()] == [(1, "peak"), (3, "after removal")]


def test_metadata(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.set_metadata("a", "1")
        w.update_metadata({"b": "2", "c": "3"})
        w.remove_metadata("c")
        w.remove_metadata("missing")
        assert w.metadata == {"a": "1", "b": "2"}
    with Reader(path) as r:
        assert r.metadata == {"a": "1", "b": "2"}


def test_commit_without_changes(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        assert w.commit() is None
        assert not w.has_pending_changes
        w.create_device(SIMPLE)
        assert w.has_pending_changes
        w.commit()
        w.write_row("d", 1, v=1)
        assert w.has_pending_changes
        w.delete("d", 5)
        w.rollback()
        assert not w.has_pending_changes
    with Reader(path) as r:
        assert len(r.commits()) == 1
        assert len(r.read("d")) == 0


def test_rollback_discards_frames_and_devices(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write("d", {"time": [1], "v": [1]})
        w.commit()
        size = path.stat().st_size
        w.create_device(DeviceSchema.build("e", "s", {"x": "bool"}))
        w.write("d", {"time": list(range(100)), "v": list(range(100))})
        w.annotate("d", 1, 2, "x")
        w.set_metadata("k", "v")
        w.rollback()
        assert path.stat().st_size == size
        assert [d.name for d in w.devices] == ["d"]
        assert w.annotations() == {}
        assert w.metadata == {}
        w.write_row("d", 2, v=2)
    assert read_all(path).time.tolist() == [1, 2]


def test_exception_in_with_block_discards_changes(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
    with pytest.raises(RuntimeError), Writer.open(path) as w:
        w.write_row("d", 1, v=1)
        raise RuntimeError
    with Reader(path) as r:
        assert r.finalized
        assert len(r.read("d")) == 0
        assert len(r.commits()) == 1


def test_closed_writer(path: Path) -> None:
    w = Writer.create(path)
    w.close()
    w.close()  # idempotent
    assert w.closed
    assert "closed" in repr(w)
    with pytest.raises(TrosnaError, match="closed"):
        w.write_row("d", 1)


def test_garbage_collected_writer_finalizes_the_file(path: Path) -> None:
    w = Writer.create(path, sync=False)
    w.create_device(SIMPLE)
    w.write_row("d", 1, v=1)
    del w
    import gc

    gc.collect()
    with Reader(path) as r:
        assert r.finalized
        assert len(r.read("d")) == 1


def test_appending_keeps_history(path: Path) -> None:
    for i in range(3):
        with Writer.create(path, sync=False) if i == 0 else Writer.open(path, sync=False) as w:
            if i == 0:
                w.create_device(SIMPLE)
            w.write_row("d", i, v=i)
            w.commit(message=f"session {i}")
    with Reader(path) as r:
        assert [c.message for c in r.commits()] == ["session 0", "session 1", "session 2"]
        assert r.read("d")["v"].to_list() == [0, 1, 2]
        commits = r.commits()
        assert all(b.prev_hash == a.hash for a, b in itertools.pairwise(commits))


def test_chain_origin(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.set_chain_origin(b"\x07" * 32)
        w.create_device(SIMPLE)
        w.commit()
        with pytest.raises(InvalidArgumentError):
            w.set_chain_origin(b"\x07" * 32)
    with Reader(path) as r:
        assert r.commits()[0].prev_hash == "07" * 32
    with Writer.create(path, overwrite=True) as w, pytest.raises(InvalidArgumentError):
        w.set_chain_origin(b"short")


def test_recovery_of_an_unfinished_file(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write_row("d", 1, v=1)
    finalized = path.read_bytes()
    w = Writer.open(path, sync=False)
    w.write_row("d", 2, v=2)
    w.commit()
    w.write_row("d", 3, v=3)
    w._flush_device(0)  # an uncommitted Data frame reaches the file
    w._file.flush()
    crashed = path.read_bytes()
    w._release()  # simulate a crash: no commit, no index
    assert len(crashed) > len(finalized)
    with Reader(path) as r:
        assert not r.finalized
        assert r.read("d")["v"].to_list() == [1, 2]
    with Writer.open(path) as w2:
        assert w2.recovery is not None
        assert w2.recovery.discarded_frames == 1
        assert w2.recovery.reason == "the file was not finalized"
    with Reader(path) as r:
        assert r.finalized
        assert r.read("d")["v"].to_list() == [1, 2]


def test_truncated_tail_is_removed(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SIMPLE)
        w.write_row("d", 1, v=1)
    data = path.read_bytes()
    path.write_bytes(data + b"TBLK\x03\x00")
    with Writer.open(path) as w:
        assert w.recovery is not None
        assert w.recovery.reason == "the last frame is incomplete"
        assert w.recovery.truncated_bytes > 0


def test_damage_in_the_middle_is_not_truncated(sample_file: Path) -> None:
    # Without the footer the writer has to scan the frames and finds the damage.
    data = bytearray(sample_file.read_bytes()[:-24])
    data[20] ^= 0xFF  # inside the first frame
    sample_file.write_bytes(bytes(data))
    with pytest.raises(CorruptedError, match="refusing to truncate"):
        Writer.open(sample_file)
    assert sample_file.read_bytes() == bytes(data)
    with Writer.open(sample_file, repair_corruption=True) as w:
        assert w.recovery is not None
    with Reader(sample_file) as r:
        assert r.finalized
        assert r.devices == []


class FakeMsvcrt:
    """Stands in for the Windows ``msvcrt`` module."""

    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self, refuse_offsets: tuple[int, ...] = ()) -> None:
        self.locked: dict[int, int] = {}  # offset -> fd
        self.refuse = refuse_offsets
        self.calls: list[tuple[int, int]] = []

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        import errno
        import os

        offset = os.lseek(fd, 0, os.SEEK_CUR)
        self.calls.append((mode, offset))
        assert nbytes == 1
        if mode == self.LK_UNLCK:
            self.locked.pop(offset, None)
            return
        if offset in self.refuse:
            raise OSError(errno.EINVAL, "Invalid argument")
        if offset in self.locked:
            raise OSError(errno.EACCES, "Permission denied")
        self.locked[offset] = fd


@pytest.fixture
def fake_windows(monkeypatch: pytest.MonkeyPatch) -> FakeMsvcrt:
    import sys

    import pytrosna.writer as writer_module

    fake = FakeMsvcrt()
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(writer_module, "_WINDOWS", True)
    return fake


def test_windows_lock_far_beyond_the_data(path: Path, fake_windows: FakeMsvcrt) -> None:
    w = Writer.create(path, sync=False)
    assert fake_windows.locked == {1 << 62: w._file.fileno()}
    with pytest.raises(LockedError):
        Writer.open(path)
    w.create_device(SIMPLE)
    w.write_row("d", 1, v=1)
    w.close()
    assert fake_windows.locked == {}  # unlocked before closing
    with Writer.open(path, sync=False) as again:
        again.write_row("d", 2, v=2)
    assert read_all(path)["v"].to_list() == [1, 2]
    assert fake_windows.locked == {}


def test_windows_lock_falls_back_to_a_32_bit_offset(path: Path, fake_windows: FakeMsvcrt) -> None:
    fake_windows.refuse = (1 << 62,)
    w = Writer.create(path, sync=False)
    assert list(fake_windows.locked) == [(1 << 31) - 2]
    w.close()
    assert fake_windows.locked == {}
    fake_windows.refuse = (1 << 62, (1 << 31) - 2)
    with pytest.raises(OSError, match="Invalid argument"):
        Writer.open(path)
