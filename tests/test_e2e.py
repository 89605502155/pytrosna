"""End-to-end scenarios: long workflows across many features, a model-based
test of the editing semantics, crash simulation and corruption."""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import pytrosna
from pytrosna import DeviceSchema, Reader, Writer, cli
from pytrosna.errors import CorruptedError, TrosnaError

# ====================================================================== lifecycle


def test_sensor_archive_lifecycle(tmp_path: Path) -> None:
    """A laboratory archive: CSV import, daily appends from pandas and Polars,
    corrections in transactions, labels, time travel, diff, compaction and
    verification of provenance."""
    archive = tmp_path / "lab.trosna"
    csv = tmp_path / "day1.csv"
    times = pd.date_range("2026-10-01", periods=1440, freq="min", tz="Europe/Moscow", unit="s")
    rng = np.random.default_rng(42)
    temperature = np.round(20 + np.cumsum(rng.normal(0, 0.05, times.size)), 2)
    pd.DataFrame(
        {"time": times.strftime("%Y-%m-%dT%H:%M:%S%z"), "temperature": temperature}
    ).to_csv(csv, index=False)

    # Day 1 arrives as CSV through the command line.
    assert cli.main(["convert", str(csv), str(archive), "--device", "room1"]) == 0
    f = pytrosna.open(archive)
    assert f.count() == 1440
    day1_hash = f.head.hash  # type: ignore[union-attr]

    # Days 2 and 3 arrive from pandas and Polars.
    for day, maker in ((2, "pandas"), (3, "polars")):
        stamps = pd.date_range(
            f"2026-10-0{day}", periods=1440, freq="min", tz="Europe/Moscow", unit="s"
        )
        values = np.round(20 + np.cumsum(rng.normal(0, 0.05, stamps.size)), 2)
        frame = pd.DataFrame({"time": stamps, "temperature": values})
        data = frame if maker == "pandas" else pl.from_pandas(frame)
        f.write(data, "room1", message=f"day {day} from {maker}", author="collector")
    assert f.count() == 3 * 1440
    assert len(f.commits()) == 3

    # A technician corrects a faulty sensor reading and labels the incident.
    with f.edit(message="sensor glitch", author="technician") as tx:
        tx.update("room1", "2026-10-02 12:00", temperature=21.0)
        tx.delete_range("room1", "2026-10-02 12:01", "2026-10-02 12:09")
        tx.annotate(
            "room1", "2026-10-02 12:00", "2026-10-02 12:10", "maintenance", "probe replaced"
        )
    assert f.count() == 3 * 1440 - 9
    point = f.read(start="2026-10-02 12:00", end="2026-10-02 12:00")
    assert point["temperature"].to_list() == [21.0]

    # Time travel: version 3 still shows the original readings.
    original = f.read(start="2026-10-02 12:00", end="2026-10-02 12:10", as_of=3)
    assert len(original) == 11
    diff = f.diff(3)
    assert [p.kind.value for p in diff.points].count("removed") == 9
    assert [p.kind.value for p in diff.points].count("changed") == 1
    assert [a.after.label for a in diff.annotations] == ["maintenance"]  # type: ignore[union-attr]

    # The history is a hash chain; any version can be named by its hash.
    commits = f.commits()
    assert commits[0].hash == day1_hash
    assert all(b.prev_hash == a.hash for a, b in itertools.pairwise(commits))
    assert f.verify().ok

    # Compaction keeps the latest state and links to the history it replaces.
    compacted = f.compact(tmp_path / "lab-compacted.trosna")
    assert compacted.count() == f.count()
    assert compacted.commits()[0].prev_hash == f.head.hash  # type: ignore[union-attr]
    assert compacted.annotations() == f.annotations()
    assert compacted.metadata["trosna.compacted_from_commit"] == "4"
    assert compacted.size < f.size
    left = f.read_pandas()
    right = compacted.read_pandas()
    pd.testing.assert_frame_equal(left, right)

    # Export the corrected data back to CSV and check it round-trips.
    out = tmp_path / "export.csv"
    assert cli.main(["convert", str(compacted.path), str(out)]) == 0
    exported = pd.read_csv(out)
    assert len(exported) == compacted.count()
    assert exported["temperature"].tolist() == right["temperature"].tolist()


def test_multi_device_streaming(tmp_path: Path) -> None:
    """Several devices, many blocks, every codec; streamed reads equal full reads."""
    for codec in ("zstd", "lz4", "none"):
        path = tmp_path / f"multi-{codec}.trosna"
        rng = np.random.default_rng(7)
        with Writer.create(path, codec=codec, rows_per_block=5000, sync=False) as w:
            for k in range(3):
                w.create_device(
                    DeviceSchema.build(
                        f"vm{k}",
                        "ms",
                        {"cpu": "float32", "ram": "int64", "state": "string", "up": "bool"},
                    )
                )
            for chunk in range(4):
                for k in range(3):
                    n = 6000
                    start = chunk * n * 1000
                    w.write(
                        f"vm{k}",
                        {
                            "time": start + np.arange(n, dtype=np.int64) * 1000 + k,
                            "cpu": rng.random(n).astype(np.float32),
                            "ram": rng.integers(1 << 20, 1 << 22, n),
                            "state": rng.choice(["ok", "busy", "idle"], n),
                            "up": rng.random(n) < 0.99,
                        },
                    )
                w.commit(message=f"chunk {chunk}")
        f = pytrosna.open(path)
        for k in range(3):
            whole = f.read(f"vm{k}")
            assert len(whole) == 24_000
            streamed = list(f.iter_batches(f"vm{k}"))
            assert len(streamed) > 1
            assert pytrosna.Batch.concat(streamed) == whole
            arrow = pytrosna.read_batches(path, f"vm{k}").read_all()
            assert arrow.num_rows == 24_000
        assert pytrosna.verify(path).ok


# ====================================================================== model-based test


class Model:
    """A reference model of SPEC §10: per version, a dict of points and annotations."""

    def __init__(self) -> None:
        self.points: dict[int, dict[str, object]] = {}
        self.annotations: dict[int, tuple[int, int, str]] = {}
        # Identifiers are never reused, even after a removal: a new one is one
        # more than the largest identifier ever used (SPEC §5.5).
        self.max_id = 0
        self.versions: list[
            tuple[dict[int, dict[str, object]], dict[int, tuple[int, int, str]]]
        ] = [({}, {})]
        self._snapshot = self._copy()

    def _copy(self) -> tuple[dict[int, dict[str, object]], dict[int, tuple[int, int, str]], int]:
        return {t: dict(v) for t, v in self.points.items()}, dict(self.annotations), self.max_id

    def rollback(self) -> None:
        self.points, self.annotations, self.max_id = self._snapshot
        self._snapshot = self._copy()

    def commit(self) -> None:
        points, annotations, _ = self._copy()
        self.versions.append((points, annotations))
        self._snapshot = self._copy()


def equal_values(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


@pytest.mark.parametrize("seed", range(6))
def test_editing_semantics_against_a_model(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    schema = DeviceSchema.build("d", "ms", {"x": "float64", "n": "int32", "s": "string"})
    model = Model()
    w = Writer.create(path, sync=False, rows_per_block=rng.choice([3, 16, 1000]))
    w.create_device(schema)
    w.commit()
    model.commit()
    span = 60

    def random_row() -> dict[str, object]:
        return {
            "x": rng.choice([None, float(rng.randint(-5, 5)), math.nan]),
            "n": rng.choice([None, rng.randint(-100, 100)]),
            "s": rng.choice([None, "a", "bb", "ccc"]),
        }

    for step in range(120):
        action = rng.random()
        if action < 0.30:  # a batch of rows, possibly out of order and duplicated
            times = [rng.randrange(span) for _ in range(rng.randint(1, 8))]
            rows = [random_row() for _ in times]
            w.write("d", {"time": times, **{k: [r[k] for r in rows] for k in ("x", "n", "s")}})
            for t, r in zip(times, rows, strict=True):
                model.points[t] = r
        elif action < 0.45:
            t = rng.randrange(span)
            r = random_row()
            w.write_row("d", t, r)
            model.points[t] = r
        elif action < 0.55:
            t = rng.randrange(span)
            change = {"n": rng.randint(0, 9)}
            if t in model.points:
                w.update("d", t, change)
                model.points[t] = {**model.points[t], **change}
            else:
                with pytest.raises(pytrosna.PointNotFoundError):
                    w.update("d", t, change)
        elif action < 0.70:
            lo = rng.randrange(span)
            hi = lo + rng.randrange(8)
            w.delete_range("d", lo, hi)
            for t in [t for t in model.points if lo <= t <= hi]:
                del model.points[t]
        elif action < 0.78:
            lo = rng.randrange(span)
            hi = lo + rng.randrange(5)
            label = f"label{step}"
            annotation_id = w.annotate("d", lo, hi, label)
            model.max_id += 1
            assert annotation_id == model.max_id
            model.annotations[annotation_id] = (lo, hi, label)
        elif action < 0.82 and model.annotations:
            annotation_id = rng.choice(sorted(model.annotations))
            w.remove_annotation(annotation_id)
            del model.annotations[annotation_id]
        elif action < 0.90:
            if w.commit(message=f"step {step}") is not None:
                model.commit()
        elif action < 0.93:
            w.rollback()
            model.rollback()
        elif action < 0.96:
            # closing commits what is pending
            if w.has_pending_changes:
                model.commit()
            w.close()
            w = Writer.open(path, sync=False)
        else:
            t = rng.randrange(span)
            got = w.get("d", t)
            expected = model.points.get(t)
            assert (got is None) == (expected is None)
            if got is not None and expected is not None:
                assert all(equal_values(got[k], expected[k]) for k in expected)
    if w.commit() is not None:
        model.commit()
    w.close()

    with Reader(path) as r:
        assert len(r.commits()) == len(model.versions) - 1
        for version, (points, annotations) in enumerate(model.versions):
            batch = r.read("d", as_of=version) if version else r.read("d", as_of=0)
            assert batch.time.tolist() == sorted(points), f"version {version}"
            for t, row in batch.rows():
                assert all(equal_values(row[k], points[t][k]) for k in row), (version, t)
            got = {a.id: (a.start, a.end, a.label) for a in r.annotations(as_of=version)}
            assert got == annotations, f"annotations of version {version}"
    assert pytrosna.verify(path).ok


# ====================================================================== crashes and corruption


def build_history(path: Path) -> list[int]:
    """Three commits; returns the number of points after each commit."""
    with Writer.create(path, sync=False, rows_per_block=4) as w:
        w.create_device(DeviceSchema.build("d", "s", {"v": "int64", "s": "string"}))
        w.write("d", {"time": list(range(10)), "v": list(range(10)), "s": ["x"] * 10})
        w.commit()
        w.delete_range("d", 2, 4)
        w.annotate("d", 0, 1, "label")
        w.commit()
        w.write("d", {"time": [20, 21], "v": [1, 2]})
        w.commit()
    return [10, 7, 9]


def test_truncation_at_every_byte(tmp_path: Path) -> None:
    """A crash can cut a file anywhere; the reader then returns the state of the
    last complete commit, and the next writer continues from it."""
    original = tmp_path / "full.trosna"
    counts = build_history(original)
    data = original.read_bytes()
    with Reader(original) as r:
        commit_ends = [
            c.offset + 16 + len(r._catalog.commits[i].record.encode())
            for i, c in enumerate(r._catalog.commits)
        ]
    cut = tmp_path / "cut.trosna"
    for size in range(len(data) + 1):
        cut.write_bytes(data[:size])
        complete = sum(1 for end in commit_ends if end <= size)
        try:
            with Reader(cut) as r:
                assert len(r.commits()) == complete
                if complete:
                    assert r.query("d").count() == counts[complete - 1]
        except CorruptedError:
            assert size < 8  # only a file without a complete header is unreadable
            continue
        if size >= 8 and size % 7 == 0:
            with Writer.open(cut, sync=False) as w:
                if not w.has_device("d"):
                    w.create_device(DeviceSchema.build("d", "s", {"v": "int64", "s": "string"}))
                w.write_row("d", 100, v=1)
            with Reader(cut) as r:
                assert r.finalized
                assert r.query("d").time_range(100, 100).count() == 1
            assert pytrosna.verify(cut).ok


def test_random_corruption_never_crashes(tmp_path: Path) -> None:
    """Flipped bytes give an error report or correct data, never another exception."""
    original = tmp_path / "full.trosna"
    build_history(original)
    data = original.read_bytes()
    with Reader(original) as r:
        expected = r.read("d")
    damaged = tmp_path / "damaged.trosna"
    rng = random.Random(1)
    for _ in range(300):
        mutated = bytearray(data)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(mutated))
            mutated[i] ^= 1 << rng.randrange(8)
        damaged.write_bytes(bytes(mutated))
        try:
            report = pytrosna.verify(damaged)
            assert isinstance(report.ok, bool)
            with Reader(damaged) as r:
                if "d" in r.device_names:
                    batch = r.read("d")
                    if report.ok and r.finalized:
                        assert batch == expected
        except (TrosnaError, KeyError):
            pass


def test_writer_refuses_damage_but_repairs_on_request(tmp_path: Path) -> None:
    path = tmp_path / "damaged.trosna"
    build_history(path)
    data = bytearray(path.read_bytes()[:-24])
    data[30] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CorruptedError):
        Writer.open(path)
    report = pytrosna.recover(path, force=True)
    assert report.removed_bytes > 0
    assert pytrosna.verify(path).ok
