"""History: commit summaries and differences between versions."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pytrosna import ChangeKind, DeviceSchema, Reader, Writer
from pytrosna.errors import InvalidArgumentError, UnknownCommitError

from .conftest import T0

SCHEMA = DeviceSchema.build("d", "s", {"v": "float64", "n": "int32"})


@pytest.fixture
def history(path: Path) -> Path:
    with Writer.create(path, sync=False) as w:
        w.create_device(SCHEMA)
        w.create_device(DeviceSchema.build("other", "s", {"x": "bool"}))
        w.write("d", {"time": [1, 2, 3, 4], "v": [1.0, 2.0, math.nan, 4.0], "n": [1, 2, 3, 4]})
        w.annotate("d", 1, 2, "first")
        w.annotate("other", 1, 1, "elsewhere")
        w.commit(message="one")  # 1
        w.update("d", 2, v=20.0)
        w.update("d", 3, v=math.nan)  # same bits: not a change
        w.delete("d", 4)
        w.write_row("d", 5, v=5.0)
        w.update_annotation(1, label="changed")
        w.annotate("d", 3, 3, "second")
        w.commit(message="two")  # 2
        w.remove_annotation(2)
        w.remove_annotation(1)
        w.set_metadata("k", "v")
        w.commit(message="three")  # 3
    return path


def test_commit_changes(history: Path) -> None:
    with Reader(history) as r:
        one, two, three = (c.changes for c in r.commits())
    assert one.devices_created == ("d", "other")
    assert (one.blocks_written, one.rows_written, one.annotation_ops) == (1, 4, 2)
    assert (two.ranges_deleted, two.rows_written, two.annotation_ops) == (1, 3, 2)
    assert three.metadata_changed
    assert three.annotation_ops == 2


def test_diff_points(history: Path) -> None:
    with Reader(history) as r:
        d = r.diff("d", 1, 2)
    assert (d.device, d.columns, d.from_commit, d.to_commit) == ("d", ("v", "n"), 1, 2)
    kinds = [(p.time, p.kind) for p in d.points]
    assert kinds == [(2, ChangeKind.CHANGED), (4, ChangeKind.REMOVED), (5, ChangeKind.ADDED)]
    changed = d.points[0]
    assert changed.before == {"v": 2.0, "n": 2}
    assert changed.after == {"v": 20.0, "n": 2}
    assert d.points[1].after is None
    assert d.points[2].before is None
    assert str(ChangeKind.ADDED) == "added"
    assert d


def test_diff_annotations(history: Path) -> None:
    with Reader(history) as r:
        two = r.diff("d", 1, 2)
        three = r.diff("d", 2)
        nothing = r.diff("other", 1, 2)
        removed_elsewhere = r.diff("other", 2, 3)
        from_empty = r.diff("d", 0, 1)
    assert [(a.id, a.kind) for a in two.annotations] == [
        (1, ChangeKind.CHANGED),
        (3, ChangeKind.ADDED),
    ]
    # annotation 2 belongs to another device, 3 is kept
    assert [(a.id, a.kind) for a in three.annotations] == [(1, ChangeKind.REMOVED)]
    assert three.points == ()
    assert not nothing
    assert [a.id for a in removed_elsewhere.annotations] == [2]
    assert [p.kind for p in from_empty.points] == [ChangeKind.ADDED] * 4


def test_diff_errors(history: Path) -> None:
    with Reader(history) as r:
        with pytest.raises(InvalidArgumentError):
            r.diff("d", 2, 1)
        with pytest.raises(UnknownCommitError):
            r.diff("d", 1, 9)


def test_large_diff_reads_only_touched_ranges(path: Path) -> None:
    with Writer.create(path, sync=False, rows_per_block=1000) as w:
        w.create_device(SCHEMA)
        w.write("d", {"time": list(range(10_000)), "v": [float(i) for i in range(10_000)]})
        w.commit()
        w.update("d", T0 % 10_000, v=-1.0)
    with Reader(path) as r:
        d = r.diff("d", 1)
    assert [p.time for p in d.points] == [T0 % 10_000]
