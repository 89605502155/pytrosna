"""Interoperability with the reference Rust implementation.

These tests run the reference ``trosna`` command (set ``TROSNA_CLI`` to its
path) and are skipped when it is not available.
"""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pytrosna
from pytrosna import DeviceSchema, Reader, Writer

from .conftest import SCHEMA, sample_columns

pytestmark = pytest.mark.interop


def trosna(cli: str, *args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli, *map(str, args)], capture_output=True, text=True, check=check, timeout=120
    )


def rust_rows(cli: str, path: Path, *args: str) -> list[dict[str, str]]:
    out = trosna(cli, "cat", path, "--raw-time", *args).stdout
    return list(csv.DictReader(io.StringIO(out)))


def assert_same_as_rust(cli: str, path: Path, device: str) -> None:
    """The Rust tool reads the same points as pytrosna."""
    ours = pytrosna.read(path, device)
    theirs = rust_rows(cli, path, "--device", device)
    assert [int(r[ours.schema.time_name]) for r in theirs] == ours.time.tolist()  # type: ignore[union-attr]
    for name, column in ours.columns.items():
        values = column.to_list()
        for row, value in zip(theirs, values, strict=True):
            text = row[name]
            if value is None:
                assert text == ""
            elif isinstance(value, bool):
                assert text == ("true" if value else "false")
            elif isinstance(value, float):
                assert (
                    np.float32(float(text)) == np.float32(value)
                    or float(text) == value
                    or (np.isnan(value) and text.lower() == "nan")
                )
            else:
                assert text == str(value)


@pytest.mark.parametrize("codec", ["zstd", "lz4", "none"])
def test_rust_reads_python_files(rust_cli: str, path: Path, codec: str) -> None:
    with Writer.create(path, codec=codec, rows_per_block=500, sync=False) as w:
        w.set_metadata("source", "pytrosna")
        w.create_device(SCHEMA)
        w.create_device(DeviceSchema.build("room", "s", {"t": "float64"}, timezone="Europe/Moscow"))
        w.write("vm01", sample_columns(3000, seed=3))
        w.write(
            "room",
            {
                "time": [1_791_442_800 + i for i in range(100)],
                "t": [20.0 + i / 10 for i in range(100)],
            },
        )
        w.commit(message="import", author="python")
        w.update("vm01", 1_700_000_005_000, cpu=1.5, state="правка")
        w.delete_range("vm01", 1_700_000_100_000, 1_700_000_200_000)
        w.annotate("vm01", 1_700_000_000_000, 1_700_000_009_000, "spike", "заметка")
        w.commit(message="edit")
    result = trosna(rust_cli, "verify", path)
    assert "OK" in result.stdout or result.returncode == 0
    head = pytrosna.open(path).head
    assert head is not None
    assert head.hash in result.stdout
    assert_same_as_rust(rust_cli, path, "vm01")
    assert_same_as_rust(rust_cli, path, "room")
    log = trosna(rust_cli, "log", path).stdout
    assert "import" in log
    assert "edit" in log
    annotations = trosna(rust_cli, "annotations", path).stdout
    assert "spike" in annotations
    old = rust_rows(rust_cli, path, "--device", "vm01", "--as-of", "1")
    assert len(old) == 3000


def test_python_reads_rust_files(rust_cli: str, tmp_path: Path) -> None:
    source = tmp_path / "sensors.csv"
    rng = np.random.default_rng(5)
    lines = ["time,temperature,count,ok,label"]
    for i in range(5000):
        lines.append(
            f"2026-10-08T10:{(i // 60) % 60:02d}:{i % 60:02d}+03:00,"
            f"{20 + rng.normal():.3f},{rng.integers(0, 9)},{'true' if i % 5 else 'false'},"
            f"{'' if i % 7 == 0 else 'l' + str(i % 3)}"
        )
    source.write_text("\n".join(lines[:3601]) + "\n")  # one hour of seconds
    target = tmp_path / "rust.trosna"
    trosna(rust_cli, "convert", source, target, "--device", "room1", "-m", "rust import")
    f = pytrosna.open(target)
    assert f.finalized
    assert f.device().timezone == "+03:00"
    assert f.count() == 3600
    assert f.verify().ok
    assert f.head is not None
    assert f.head.message == "rust import"
    assert_same_as_rust(rust_cli, target, "room1")
    df = f.read_pandas()
    assert df["label"].isna().sum() == sum(1 for i in range(3600) if i % 7 == 0)


def test_mixed_editing_history(rust_cli: str, path: Path) -> None:
    pytrosna.write(
        path, {"time": list(range(0, 100, 10)), "v": [float(i) for i in range(10)]}, "d", unit="s"
    )
    # Rust edits a Python file ...
    trosna(rust_cli, "update", path, "--time", "1970-01-01T00:00:20Z", "v=2.5", "-m", "rust update")
    trosna(rust_cli, "delete", path, "--time", "1970-01-01T00:00:30Z", "-m", "rust delete")
    trosna(
        rust_cli,
        "annotate",
        path,
        "--start",
        "1970-01-01T00:00:00Z",
        "--end",
        "1970-01-01T00:00:10Z",
        "--label",
        "rust label",
    )
    # ... then Python edits it again.
    f = pytrosna.open(path)
    with f.edit(message="python edit") as tx:
        tx.insert("d", 35, v=3.5)
        tx.update_annotation(1, label="renamed in python")
    assert [c.message for c in f.commits()] == [
        None,
        "rust update",
        "rust delete",
        None,
        "python edit",
    ]
    assert f.read()["v"].to_list() == [0.0, 1.0, 2.5, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert f.verify().ok
    result = trosna(rust_cli, "verify", path)
    assert result.returncode == 0
    assert "renamed in python" in trosna(rust_cli, "annotations", path).stdout
    diff = trosna(rust_cli, "diff", path, "--from", "1").stdout
    assert "2.5" in diff


def test_compaction_both_ways(rust_cli: str, sample_file: Path, tmp_path: Path) -> None:
    by_rust = tmp_path / "by-rust.trosna"
    trosna(rust_cli, "compact", sample_file, by_rust)
    by_python = tmp_path / "by-python.trosna"
    pytrosna.compact(sample_file, by_python)
    with Reader(by_rust) as a, Reader(by_python) as b:
        assert a.read("vm01") == b.read("vm01")
        assert a.annotations() == b.annotations()
        assert a.metadata == b.metadata
        assert a.commits()[0].prev_hash == b.commits()[0].prev_hash
    assert trosna(rust_cli, "verify", by_python).returncode == 0


def test_rust_recovers_python_files(rust_cli: str, sample_file: Path) -> None:
    sample_file.write_bytes(sample_file.read_bytes()[:-24] + b"TBLK\x03")
    trosna(rust_cli, "recover", sample_file)
    assert pytrosna.verify(sample_file).finalized
    assert pytrosna.open(sample_file).count() == 17
