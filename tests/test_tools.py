"""Verification, recovery and compaction."""

from __future__ import annotations

from pathlib import Path

import pytest

import pytrosna
from pytrosna import Reader, Writer, compact, recover, verify
from pytrosna.errors import CorruptedError, NotTrosnaError

from .conftest import GOLDEN_HEAD


def test_verify_intact_file(sample_file: Path) -> None:
    report = verify(sample_file)
    assert report.ok
    assert report
    assert report.finalized
    assert (report.commits, report.blocks) == (2, 4)
    assert report.frames > 0
    assert report.head is not None
    assert report.origin is None
    assert report.problems == ()
    assert report.warnings == ()


def test_verify_detects_modified_history(sample_file: Path) -> None:
    data = bytearray(sample_file.read_bytes())
    with Reader(sample_file) as r:
        entry = r._catalog.data[1]
    position = entry.segment_offsets[1]
    data[position] ^= 0x01
    # Recompute the frame CRC, so that only the hash chain can tell.
    from pytrosna import _format as fmt

    start = entry.offset
    header = fmt.parse_frame_header(bytes(data[start : start + 12]))
    end = start + header.frame_len
    payload = bytes(data[start + 12 : end - 4])
    data[end - 4 : end] = fmt.frame_crc(bytes(data[start : start + 12]), payload).to_bytes(
        4, "little"
    )
    sample_file.write_bytes(bytes(data))
    report = verify(sample_file)
    assert not report.ok
    assert any("content hash mismatch" in p for p in report.problems)
    assert any("checksum" in p for p in report.problems)


def test_verify_damaged_frame_in_the_middle(sample_file: Path) -> None:
    data = bytearray(sample_file.read_bytes())
    data[40] ^= 0xFF
    sample_file.write_bytes(bytes(data))
    report = verify(sample_file)
    assert not report.ok
    assert any("valid frames follow" in p for p in report.problems)


def test_verify_unfinished_file(sample_file: Path) -> None:
    data = sample_file.read_bytes()
    sample_file.write_bytes(data[:-24] + b"TBLK\x01")
    report = verify(sample_file)
    assert report.ok
    assert not report.finalized
    assert any("incomplete" in w for w in report.warnings)
    assert any("not finalized" in w for w in report.warnings)


def test_verify_garbage_at_the_end(sample_file: Path) -> None:
    sample_file.write_bytes(sample_file.read_bytes() + b"\x00" * 40)
    report = verify(sample_file)
    assert report.ok
    assert any("at the end of the file" in w for w in report.warnings)


def test_verify_damaged_index(sample_file: Path) -> None:
    data = bytearray(sample_file.read_bytes())
    index_offset = int.from_bytes(data[-12:-4], "little")
    data[index_offset + 30] ^= 0xFF
    sample_file.write_bytes(bytes(data))
    report = verify(sample_file)
    assert any("damaged index" in p for p in report.problems)


def test_verify_rejects_other_files(tmp_path: Path) -> None:
    p = tmp_path / "a.trosna"
    p.write_bytes(b"abc")
    with pytest.raises(NotTrosnaError):
        verify(p)


def test_verify_golden_file() -> None:
    from .conftest import GOLDEN

    report = verify(GOLDEN)
    assert report.ok
    assert report.head == GOLDEN_HEAD


def test_recover(sample_file: Path) -> None:
    assert recover(sample_file).was_finalized
    data = sample_file.read_bytes()
    sample_file.write_bytes(data[:-24])
    report = recover(sample_file)
    assert not report.was_finalized
    assert report.recovery is not None
    # the dead Index frame after the last commit is cut off; no data is lost
    index_offset = int.from_bytes(data[-12:-4], "little")
    assert report.removed_bytes == len(data) - 24 - index_offset
    assert report.removed_frames == 0
    assert verify(sample_file).finalized


def test_recover_with_force(sample_file: Path) -> None:
    data = bytearray(sample_file.read_bytes()[:-24])
    data[40] ^= 0xFF
    sample_file.write_bytes(bytes(data))
    with pytest.raises(CorruptedError):
        recover(sample_file)
    report = recover(sample_file, force=True)
    assert report.removed_bytes > 0
    assert verify(sample_file).ok


def test_compact(sample_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "small.trosna"
    report = compact(sample_file, target)
    assert (report.version, report.rows, report.annotations) == (2, 17, 1)
    assert report.bytes_after < report.bytes_before
    with Reader(sample_file) as a, Reader(target) as b:
        assert a.read("vm01") == b.read("vm01")
        assert a.annotations() == b.annotations()
        assert len(b.commits()) == 1
        assert b.commits()[0].prev_hash == a.head.hash  # type: ignore[union-attr]
        assert b.metadata["trosna.compacted_from"] == a.head.hash  # type: ignore[union-attr]
        assert b.metadata["trosna.compacted_from_commit"] == "2"
        assert b.metadata["title"] == "sample"
        assert len(b.blocks("vm01")) == 1
    assert verify(target).origin == report.origin
    with pytest.raises(FileExistsError):
        compact(sample_file, target)


def test_compact_an_older_version(sample_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "v1.trosna"
    report = compact(sample_file, target, as_of=1)
    assert (report.version, report.rows, report.annotations) == (1, 20, 0)
    report = compact(sample_file, target, as_of=0, overwrite=True)
    assert report.rows == 0
    assert report.origin == "0" * 64
    with Reader(target) as r:
        assert r.devices == []


def test_compact_failure_leaves_no_half_file(
    sample_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "broken.trosna"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError

    monkeypatch.setattr(Writer, "restore_annotation", fail)
    with pytest.raises(RuntimeError):
        compact(sample_file, target)
    with Reader(target) as r:
        assert r.devices == []
    assert pytrosna.verify(target).ok
