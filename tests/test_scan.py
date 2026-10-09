"""Catalogue building: index loading, scanning and recovery helpers (SPEC §8-§9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pytrosna import _format as fmt
from pytrosna._catalog import Catalog
from pytrosna._scan import END_OF_FILE, TRUNCATED, find_valid_frame_after, load_index, read_at, scan
from pytrosna.errors import CorruptedError, UnsupportedError


def scan_bytes(tmp_path: Path, data: bytes, *, full: bool = True) -> object:
    p = tmp_path / "x.trosna"
    p.write_bytes(data)
    with p.open("rb") as f:
        return scan(f, len(data), full=full)


def test_scan_of_a_finalized_file(sample_file: Path) -> None:
    data = sample_file.read_bytes()
    events = []
    with sample_file.open("rb") as f:
        outcome = scan(f, len(data), full=True, visit=events.append)
        index = load_index(f, len(data))
    assert outcome.stop == END_OF_FILE
    assert outcome.ends_with_footer
    assert not outcome.invalid
    assert outcome.discarded_frames == 0
    assert len(outcome.catalog.commits) == 2
    assert index is not None
    assert index.index_entries() == outcome.catalog.index_entries()
    assert events[-1].kind is fmt.FrameKind.FOOTER
    with sample_file.open("rb") as f:
        light = scan(f, len(data), full=False)
    assert light.catalog.index_entries() == outcome.catalog.index_entries()


def test_scan_stops_at_damage(tmp_path: Path, sample_file: Path) -> None:
    data = bytearray(sample_file.read_bytes())
    data[8] = ord("X")  # sync marker of the first frame
    outcome = scan_bytes(tmp_path, bytes(data))
    assert outcome.invalid  # type: ignore[attr-defined]
    assert outcome.stop == "missing frame synchronisation marker"  # type: ignore[attr-defined]
    assert outcome.committed_end == 8  # type: ignore[attr-defined]


def test_scan_of_a_truncated_file(tmp_path: Path, sample_file: Path) -> None:
    data = sample_file.read_bytes()
    outcome = scan_bytes(tmp_path, data[:-30])
    assert outcome.stop == TRUNCATED  # type: ignore[attr-defined]
    assert len(outcome.catalog.commits) == 2  # type: ignore[attr-defined]
    outcome = scan_bytes(tmp_path, data[:-5])
    assert outcome.stop == TRUNCATED  # type: ignore[attr-defined]


def test_unknown_frames(tmp_path: Path) -> None:
    ancillary = fmt.encode_frame(0x90, b"vendor data")
    commit = fmt.CommitRecord(1, 0, 8, 1, fmt.ZERO_HASH, b"\x00" * 32).seal()
    data = fmt.file_header() + ancillary + fmt.encode_frame(fmt.FrameKind.COMMIT, commit.encode())
    outcome = scan_bytes(tmp_path, data)
    assert outcome.stop == END_OF_FILE  # type: ignore[attr-defined]
    assert outcome.catalog.index_entries()[0].body == b"vendor data"  # type: ignore[attr-defined]
    critical = fmt.file_header() + fmt.encode_frame(0x09, b"")
    with pytest.raises(UnsupportedError):
        scan_bytes(tmp_path, critical)


def test_data_frames_shorter_than_their_header(tmp_path: Path) -> None:
    short = fmt.file_header() + fmt.encode_frame(fmt.FrameKind.DATA, b"\x01")
    assert scan_bytes(tmp_path, short, full=False).stop == "Data frame too short"  # type: ignore[attr-defined]
    with pytest.raises(CorruptedError, match="too short"):
        scan_bytes(tmp_path, short, full=True)
    wrong = fmt.file_header() + fmt.encode_frame(fmt.FrameKind.DATA, b"\xff\x00\x00\x00")
    assert "exceeds" in scan_bytes(tmp_path, wrong, full=False).stop  # type: ignore[attr-defined]
    with pytest.raises(CorruptedError, match="exceeds"):
        scan_bytes(tmp_path, wrong, full=True)


def test_catalog_consistency_checks() -> None:
    catalog = Catalog()
    catalog.apply(fmt.FrameKind.META, 8, fmt.encode_meta({}), None)
    with pytest.raises(CorruptedError, match="file order"):
        catalog.apply(fmt.FrameKind.META, 8, fmt.encode_meta({}), None)
    with pytest.raises(CorruptedError, match="derived"):
        catalog.apply(fmt.FrameKind.INDEX, 100, b"", None)
    with pytest.raises(CorruptedError, match="out of sequence"):
        catalog.apply(
            fmt.FrameKind.COMMIT,
            200,
            fmt.CommitRecord(2, 0, 8, 1, fmt.ZERO_HASH, fmt.ZERO_HASH).seal().encode(),
            None,
        )
    with pytest.raises(CorruptedError, match="describe"):
        catalog.apply(
            fmt.FrameKind.COMMIT,
            300,
            fmt.CommitRecord(1, 0, 9, 1, fmt.ZERO_HASH, fmt.ZERO_HASH).seal().encode(),
            None,
        )
    first = fmt.CommitRecord(1, 0, 8, 1, fmt.ZERO_HASH, fmt.ZERO_HASH).seal()
    catalog.apply(fmt.FrameKind.COMMIT, 400, first.encode(), None)
    unlinked = fmt.CommitRecord(2, 0, 500, 0, fmt.ZERO_HASH, fmt.ZERO_HASH).seal()
    with pytest.raises(CorruptedError, match="link"):
        catalog.apply(fmt.FrameKind.COMMIT, 500, unlinked.encode(), None)


def test_catalog_reference_checks() -> None:
    catalog = Catalog()
    with pytest.raises(CorruptedError, match="unknown device"):
        catalog.apply(fmt.FrameKind.TOMBSTONE, 8, fmt.encode_tombstone(0, [(1, 2)]), None)
    with pytest.raises(CorruptedError, match="unknown device"):
        catalog.apply(
            fmt.FrameKind.ANNOTATION,
            9,
            fmt.encode_annotation_ops([fmt.AnnotationOp(1, False, 0, 1, 2, "x")]),
            None,
        )
    with pytest.raises(CorruptedError, match="absent annotation"):
        catalog.apply(
            fmt.FrameKind.ANNOTATION,
            10,
            fmt.encode_annotation_ops([fmt.AnnotationOp(1, remove=True)]),
            None,
        )
    from pytrosna import DeviceSchema

    schema = DeviceSchema.build("d", "s", {"a": "bool"})
    with pytest.raises(CorruptedError, match="out of sequence"):
        catalog.apply(fmt.FrameKind.DEVICE, 11, fmt.encode_device(1, schema), None)
    catalog.apply(fmt.FrameKind.DEVICE, 12, fmt.encode_device(0, schema), None)
    with pytest.raises(CorruptedError, match="duplicate device"):
        catalog.apply(fmt.FrameKind.DEVICE, 13, fmt.encode_device(1, schema), None)


def test_index_with_uncommitted_frames_is_rejected() -> None:
    entries = [fmt.IndexEntry(fmt.FrameKind.META, 8, fmt.encode_meta({}))]
    with pytest.raises(CorruptedError, match="uncommitted"):
        Catalog.from_index(entries)


def footer_for(index_offset: int) -> bytes:
    return fmt.encode_frame(fmt.FrameKind.FOOTER, fmt.encode_footer(index_offset))


def test_load_index_errors(tmp_path: Path, sample_file: Path) -> None:
    data = sample_file.read_bytes()
    body = data[:-24]
    p = tmp_path / "x.trosna"

    def load(content: bytes) -> object:
        p.write_bytes(content)
        with p.open("rb") as f:
            return load_index(f, len(content))

    assert load(body) is None  # no footer
    assert load(body + fmt.encode_frame(fmt.FrameKind.META, b"\x00" * 8)) is None  # not a footer
    bad_crc = bytearray(data)
    bad_crc[-1] ^= 1
    assert load(bytes(bad_crc)) is None
    with pytest.raises(CorruptedError, match="outside"):
        load(body + footer_for(len(body) + 100))
    with pytest.raises(CorruptedError, match="does not point to an index"):
        load(body + footer_for(8))
    assert load(fmt.file_header()) is None


def test_read_at_checks_the_file_size(sample_file: Path) -> None:
    with sample_file.open("rb") as f:
        assert read_at(f, 0, 6) == b"TROSNA"
        with pytest.raises(CorruptedError, match="beyond the end"):
            read_at(f, 10**9, 4)


def test_find_valid_frame_after(sample_file: Path) -> None:
    data = sample_file.read_bytes()
    with sample_file.open("rb") as f:
        found = find_valid_frame_after(f, len(data), 8)
        assert found is not None
        assert data[found : found + 4] == b"TBLK"
        assert find_valid_frame_after(f, len(data), len(data) - 10) is None
