"""Building the catalogue: from the index of a finalized file, or by scanning
the frames of a file whose writer did not finish (SPEC §9)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO

from ._catalog import Catalog
from ._format import (
    FILE_HEADER_LEN,
    FOOTER_FRAME_LEN,
    FRAME_HEADER_LEN,
    FRAME_OVERHEAD,
    SYNC,
    FrameKind,
    check_frame_bytes,
    decode_footer,
    decode_index,
    frame_kind,
    is_derived,
    parse_frame_header,
)
from .errors import CorruptedError, UnsupportedError


def file_size(f: BinaryIO) -> int:
    return os.fstat(f.fileno()).st_size


def read_at(f: BinaryIO, offset: int, length: int) -> bytes:
    """Reads ``length`` bytes at ``offset``, reporting a short file as corruption.

    The length usually comes from the file itself, so it is checked against
    the file size before anything is read."""
    if offset < 0 or length < 0 or offset + length > file_size(f):
        msg = "data extends beyond the end of the file"
        raise CorruptedError(msg, offset)
    if _PREAD is not None:
        # Positioned reads bypass Python's buffer, which may hold bytes from
        # before the file was truncated and rewritten (after a rollback).
        parts = []
        done = 0
        while done < length:
            chunk = _PREAD(f.fileno(), length - done, offset + done)
            if not chunk:  # pragma: no cover - the file shrank meanwhile
                break
            parts.append(chunk)
            done += len(chunk)
        data = b"".join(parts)
    else:  # pragma: no cover - platforms without pread (Windows)
        # The unbuffered stream, for the same reason as above.
        raw = getattr(f, "raw", f)
        raw.seek(offset)
        parts = []
        done = 0
        while done < length:
            chunk = raw.read(length - done)
            if not chunk:
                break
            parts.append(chunk)
            done += len(chunk)
        data = b"".join(parts)
    if len(data) != length:  # pragma: no cover - the file shrank meanwhile
        msg = "data extends beyond the end of the file"
        raise CorruptedError(msg, offset)
    return data


_PREAD = getattr(os, "pread", None)


# Why a scan stopped.
END_OF_FILE = "end of file"
TRUNCATED = "truncated"


@dataclass
class ScanOutcome:
    catalog: Catalog
    """Committed frames only."""
    committed_end: int
    """End of the last valid ``Commit`` frame (or of the file header)."""
    stopped_at: int
    stop: str
    """``END_OF_FILE``, ``TRUNCATED`` or the reason a frame was invalid."""
    discarded_frames: int
    ends_with_footer: bool

    @property
    def invalid(self) -> bool:
        return self.stop not in (END_OF_FILE, TRUNCATED)


@dataclass(frozen=True)
class FrameEvent:
    offset: int
    kind: FrameKind | int
    data: bytes
    """The complete frame bytes (sync … crc)."""


def scan(
    f: BinaryIO,
    length: int,
    *,
    full: bool,
    visit: Callable[[FrameEvent], None] | None = None,
) -> ScanOutcome:
    """Scans the frames of a file from the header onwards.

    With ``full``, every frame is read completely and its checksum verified,
    and ``visit`` is called for each valid frame; otherwise only the parts
    needed to build the catalogue are read."""
    catalog = Catalog()
    pos = FILE_HEADER_LEN
    committed_end = FILE_HEADER_LEN
    last_kind: FrameKind | int | None = None
    while True:
        if pos == length:
            stop = END_OF_FILE
            break
        if length - pos < FRAME_OVERHEAD:
            stop = TRUNCATED
            break
        try:
            header = parse_frame_header(read_at(f, pos, FRAME_HEADER_LEN))
        except CorruptedError as e:
            stop = e.reason
            break
        if pos + header.frame_len > length:
            stop = TRUNCATED
            break
        try:
            known: FrameKind | int | None = frame_kind(header.kind)
        except UnsupportedError:
            known = None
        if known is FrameKind.DATA and not full:
            if header.length < 4:
                stop = "Data frame too short"
                break
            header_len = int.from_bytes(read_at(f, pos + FRAME_HEADER_LEN, 4), "little")
            if header_len + 4 > header.length:
                stop = "Data frame header exceeds the frame"
                break
            body = read_at(f, pos + FRAME_HEADER_LEN + 4, header_len)
            kind: FrameKind | int = FrameKind.DATA
        elif known in (FrameKind.INDEX, FrameKind.FOOTER) and not full:
            kind = known
            body = b""
        else:
            frame = read_at(f, pos, header.frame_len)
            try:
                check_frame_bytes(frame)
            except CorruptedError as e:
                stop = e.reason
                break
            # The frame is intact, so an unknown critical kind is a real incompatibility.
            kind = frame_kind(header.kind)
            if visit is not None:
                visit(FrameEvent(pos, kind, frame))
            payload = frame[FRAME_HEADER_LEN:-4]
            if kind is FrameKind.DATA:
                if len(payload) < 4:
                    msg = "Data frame too short"
                    raise CorruptedError(msg, pos)
                header_len = int.from_bytes(payload[:4], "little")
                if header_len + 4 > len(payload):
                    msg = "Data frame header exceeds the frame"
                    raise CorruptedError(msg, pos)
                body = payload[4 : 4 + header_len]
            else:
                body = payload
        if not is_derived(kind):
            catalog.apply(header.kind, pos, body, header.length)
        pos += header.frame_len
        if kind is FrameKind.COMMIT:
            committed_end = pos
        last_kind = kind
    discarded = catalog.pending_count
    catalog.truncate(committed_end)
    return ScanOutcome(
        catalog,
        committed_end,
        pos,
        stop,
        discarded,
        stop == END_OF_FILE and last_kind is FrameKind.FOOTER,
    )


def load_index(f: BinaryIO, length: int) -> Catalog | None:
    """Loads the catalogue from the footer and index of a finalized file.
    Returns ``None`` if the file does not end with a valid footer."""
    if length < FILE_HEADER_LEN + FOOTER_FRAME_LEN:
        return None
    footer_offset = length - FOOTER_FRAME_LEN
    footer = read_at(f, footer_offset, FOOTER_FRAME_LEN)
    try:
        header = parse_frame_header(footer)
    except CorruptedError:
        return None
    if header.kind != FrameKind.FOOTER or header.length != 8:
        return None
    try:
        check_frame_bytes(footer)
    except CorruptedError:
        return None
    index_offset = decode_footer(footer[FRAME_HEADER_LEN : FRAME_HEADER_LEN + 8])
    if index_offset < FILE_HEADER_LEN or index_offset + FRAME_OVERHEAD > footer_offset:
        msg = "the footer points outside the file"
        raise CorruptedError(msg, footer_offset)
    try:
        index_header = parse_frame_header(read_at(f, index_offset, FRAME_HEADER_LEN))
    except CorruptedError as e:
        raise e.at(index_offset) from None
    if (
        index_header.kind != FrameKind.INDEX
        or index_offset + index_header.frame_len > footer_offset
    ):
        msg = "the footer does not point to an index"
        raise CorruptedError(msg, footer_offset)
    try:
        frame = read_at(f, index_offset, index_header.frame_len)
        check_frame_bytes(frame)
        head, head_hash, entries = decode_index(frame[FRAME_HEADER_LEN:-4])
    except CorruptedError as e:
        raise e.at(index_offset) from None
    if any(e.offset >= index_offset for e in entries):
        msg = "the index lists frames after itself"
        raise CorruptedError(msg, footer_offset)
    catalog = Catalog.from_index(entries)
    if catalog.head() != (head, head_hash):
        msg = "the index head does not match its commits"
        raise CorruptedError(msg, footer_offset)
    return catalog


def find_valid_frame_after(f: BinaryIO, length: int, start: int) -> int | None:
    """Looks for a valid frame starting after ``start``. Used to tell an
    interrupted write (garbage only at the end) from corruption in the middle."""
    window = 1 << 16
    begin = start + 1
    while begin + FRAME_OVERHEAD <= length:
        n = min(window, length - begin)
        buf = read_at(f, begin, n)
        i = buf.find(SYNC)
        while i != -1:
            pos = begin + i
            try:
                header = parse_frame_header(read_at(f, pos, FRAME_HEADER_LEN))
            except CorruptedError:
                header = None
            if header is not None and pos + header.frame_len <= length:
                try:
                    check_frame_bytes(read_at(f, pos, header.frame_len))
                except CorruptedError:
                    pass
                else:
                    return pos
            i = buf.find(SYNC, i + 1)
        # Overlap windows so that a marker split across them is not missed.
        begin += max(n - 3, 1)
    return None
