"""Reader: catalogue, queries, versions and block information."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pytrosna import DataType, Encoding, Reader, Writer
from pytrosna.errors import (
    CorruptedError,
    NotFinalizedError,
    NotTrosnaError,
    UnknownColumnError,
    UnknownCommitError,
    UnknownDeviceError,
    UnsupportedVersionError,
)

from .conftest import SCHEMA, T0


def test_catalogue(sample_file: Path) -> None:
    with Reader(sample_file) as r:
        assert r.finalized
        assert r.format_version == (1, 0)
        assert r.metadata == {"title": "sample"}
        assert r.metadata_as_of(0) == {}
        assert r.device_names == ["vm01"]
        assert r.device("vm01") == SCHEMA
        assert "vm01" in repr(r)
        with pytest.raises(UnknownDeviceError):
            r.device("nope")


def test_queries(sample_file: Path) -> None:
    with Reader(sample_file) as r:
        q = r.query("vm01")
        assert q.count() == 17
        assert q.as_of(1).count() == 20
        assert q.as_of(0).count() == 0
        assert q.time_range(T0 + 5000, T0 + 14_000).count() == 7
        assert q.time_range(T0 + 5000, T0 + 4000).count() == 0
        assert q.time_range(None, T0 + 1000).count() == 2
        assert q.time_range(T0 + 19_000).count() == 1
        batch = q.columns(["temp", "cpu"]).time_range(T0, T0 + 3000).collect()
        assert batch.names == ["temp", "cpu"]
        assert batch["cpu"].to_list()[3] == 0.99
        assert q.device == SCHEMA
        assert sum(len(b) for b in q.batches()) == 17
        with pytest.raises(UnknownColumnError):
            q.columns(["missing"]).collect()
        with pytest.raises(UnknownCommitError):
            q.as_of(3).collect()
        old = r.read("vm01", as_of=1, columns=["cpu"], start=T0 + 3000, end=T0 + 3000)
        assert old["cpu"].to_list() != [0.99]
        assert len(r.read("vm01", columns=[])) == 17
        empty = r.read("vm01", start=0, end=10)
        assert len(empty) == 0
        assert empty.names == SCHEMA.column_names


def test_history(sample_file: Path) -> None:
    with Reader(sample_file) as r:
        commits = r.commits()
        assert [c.number for c in commits] == [1, 2]
        assert r.head == commits[-1]
        assert r.commit(1) == commits[0]
        with pytest.raises(UnknownCommitError):
            r.commit(0)
        assert r.commit_at(0) == 0
        assert r.commit_at(1_500) == 1
        assert r.commit_at(10**18) == 2
        assert r.device_exists_at("vm01", 1)
        assert not r.device_exists_at("vm01", 0)
        assert not r.device_exists_at("other", 1)
        assert not r.device_exists_at("vm01", 7)
        assert [a.label for a in r.annotations("vm01")] == ["overload"]
        assert r.annotations(as_of=1) == []
        assert commits[0].short_hash == commits[0].hash[:12]
        assert commits[0].time.year == 1970


def test_blocks(sample_file: Path) -> None:
    with Reader(sample_file) as r:
        blocks = r.blocks("vm01")
        assert [b.rows for b in blocks] == [8, 8, 4, 1]
        assert [b.commit for b in blocks] == [1, 1, 1, 2]
        first = blocks[0].segments
        assert [s.column for s in first] == ["time", *SCHEMA.column_names]
        assert first[0].data_type is None
        assert first[1].data_type is DataType.FLOAT64
        assert first[1].encoding is Encoding.XOR
        assert first[3].null_count == 3  # swap: rows 0, 3, 6
        assert first[0].statistics is not None
        assert first[0].statistics.min == T0


def test_empty_file(path: Path) -> None:
    Writer.create(path).close()
    with Reader(path) as r:
        assert r.head is None
        assert r.commits() == []
        assert r.devices == []
        assert r.finalized


def test_not_trosna_files(tmp_path: Path) -> None:
    small = tmp_path / "small.trosna"
    small.write_bytes(b"TRO")
    with pytest.raises(NotTrosnaError):
        Reader(small)
    other = tmp_path / "other.trosna"
    other.write_bytes(b"PAR1" * 10)
    with pytest.raises(NotTrosnaError):
        Reader(other)
    future = tmp_path / "future.trosna"
    future.write_bytes(b"TROSNA\x02\x00")
    with pytest.raises(UnsupportedVersionError):
        Reader(future)


def test_strict_mode(sample_file: Path) -> None:
    data = sample_file.read_bytes()
    sample_file.write_bytes(data[:-24])  # remove the footer
    with Reader(sample_file) as r:
        assert not r.finalized
        assert r.query("vm01").count() == 17
    with pytest.raises(NotFinalizedError):
        Reader(sample_file, strict=True)


def test_damaged_index_falls_back_to_scanning(sample_file: Path) -> None:
    data = bytearray(sample_file.read_bytes())
    index_offset = int.from_bytes(data[-12:-4], "little")
    data[index_offset + 20] ^= 0xFF
    sample_file.write_bytes(bytes(data))
    with Reader(sample_file) as r:
        assert not r.finalized
        assert r.query("vm01").count() == 17
    with pytest.raises(CorruptedError):
        Reader(sample_file, strict=True)


def test_checksums_can_be_skipped(sample_file: Path) -> None:
    with Reader(sample_file) as r:
        segment = r.blocks("vm01")[0]
    data = bytearray(sample_file.read_bytes())
    # flip a bit in the stored bytes of the first block's ok column (plain bitmap)
    with Reader(sample_file) as r:
        entry = r._catalog.data[0]
        position = entry.segment_offsets[4]
    data[position] ^= 0b1
    sample_file.write_bytes(bytes(data))
    with Reader(sample_file) as r, pytest.raises(CorruptedError, match="checksum"):
        r.read("vm01")
    with Reader(sample_file, verify_checksums=False) as r:
        assert len(r.read("vm01")) == 17
    assert segment.rows == 8


def test_reader_sees_a_snapshot(path: Path) -> None:
    with Writer.create(path, sync=False) as w:
        w.create_device(SCHEMA)
        w.write_row("vm01", 1, cpu=1.0)
    r = Reader(path)
    with Writer.open(path, sync=False) as w:
        w.write_row("vm01", 2, cpu=2.0)
    assert r.query("vm01").count() == 1
    r.close()
    with Reader(path) as fresh:
        assert np.array_equal(fresh.read("vm01").time, [1, 2])
