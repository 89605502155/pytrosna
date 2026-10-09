"""File-level tools: integrity verification, recovery and compaction."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import _read
from ._format import FILE_HEADER_LEN, FRAME_HEADER_LEN, ZERO_HASH, CommitRecord, FrameKind
from ._format import check_file_header as _check_header
from ._scan import (
    END_OF_FILE,
    TRUNCATED,
    file_size,
    find_valid_frame_after,
    load_index,
    read_at,
    scan,
)
from .errors import CorruptedError, NotTrosnaError, UnsupportedError
from .reader import Reader
from .writer import RecoveryReport, WriteOptions, Writer

if TYPE_CHECKING:
    from ._scan import FrameEvent

__all__ = ["CompactReport", "RecoverReport", "VerifyReport", "compact", "recover", "verify"]


@dataclass(frozen=True)
class VerifyReport:
    """Result of :func:`verify`."""

    frames: int = 0
    """Number of valid frames."""
    commits: int = 0
    blocks: int = 0
    """Number of ``Data`` blocks whose segments were all decoded successfully."""
    finalized: bool = False
    """Whether the file ends with a valid footer and index."""
    head: str | None = None
    """Hash of the last commit (hexadecimal)."""
    origin: str | None = None
    """``prev_hash`` of the first commit if it is not zero: the head of the
    file this one was compacted from."""
    problems: tuple[str, ...] = ()
    """Problems found; empty if the file is intact."""
    warnings: tuple[str, ...] = field(default=())
    """Observations that are not errors, such as an unfinished transaction."""

    @property
    def ok(self) -> bool:
        """True if no problems were found."""
        return not self.problems

    def __bool__(self) -> bool:
        return self.ok


def verify(path: str | os.PathLike[str]) -> VerifyReport:
    """Verifies a file completely: frame and segment checksums, decoding of all
    segments, the hash chain of commits and the consistency of the index."""
    with open(path, "rb") as f:
        length = file_size(f)
        if length < FILE_HEADER_LEN:
            raise NotTrosnaError
        _check_header(read_at(f, 0, FILE_HEADER_LEN))
        problems: list[str] = []
        warnings: list[str] = []
        chain_problems: list[str] = []
        hasher = hashlib.sha256()
        frames = 0

        def visit(event: FrameEvent) -> None:
            nonlocal hasher, frames
            frames += 1
            if event.kind in (FrameKind.INDEX, FrameKind.FOOTER):
                return
            if event.kind is FrameKind.COMMIT:
                record = CommitRecord.decode(event.data[FRAME_HEADER_LEN:-4])
                if hasher.digest() != record.content_hash:
                    chain_problems.append(
                        f"commit {record.number} at offset {event.offset}: content hash "
                        "mismatch (the sealed frames were modified)"
                    )
                hasher = hashlib.sha256()
            else:
                hasher.update(event.data)

        try:
            outcome = scan(f, length, full=True, visit=visit)
        except (CorruptedError, UnsupportedError) as e:
            return VerifyReport(frames=frames, problems=(str(e),))
        problems.extend(chain_problems)
        if outcome.stop == TRUNCATED:
            warnings.append(
                f"the last frame (at offset {outcome.stopped_at}) is incomplete: "
                "the writer was interrupted"
            )
        elif outcome.stop != END_OF_FILE:
            message = f"invalid frame at offset {outcome.stopped_at}: {outcome.stop}"
            if find_valid_frame_after(f, length, outcome.stopped_at) is not None:
                problems.append(f"{message}; valid frames follow it")
            else:
                warnings.append(f"{message} (at the end of the file)")
        if outcome.discarded_frames:
            warnings.append(
                f"{outcome.discarded_frames} frames of an unfinished transaction at the end "
                "of the file are ignored"
            )
        catalog = outcome.catalog
        blocks = 0
        for entry in catalog.data:
            try:
                _read.decode_block_time(f, entry, verify=True)
                for c in range(len(entry.header.segments) - 1):
                    _read.decode_block_column(f, catalog, entry, c, verify=True)
            except CorruptedError as e:
                problems.append(str(e))
            else:
                blocks += 1
        finalized = False
        try:
            index = load_index(f, length)
        except (CorruptedError, UnsupportedError) as e:
            problems.append(f"damaged index: {e}")
        else:
            if index is None:
                warnings.append("the file is not finalized; run pytrosna.recover()")
            else:
                finalized = outcome.ends_with_footer
                if index.index_entries() != catalog.index_entries():
                    problems.append("the index does not match the frames of the file")
        head = catalog.commits[-1].record.hash.hex() if catalog.commits else None
        first = catalog.commits[0].record.prev_hash if catalog.commits else ZERO_HASH
        return VerifyReport(
            frames=frames,
            commits=len(catalog.commits),
            blocks=blocks,
            finalized=finalized,
            head=head,
            origin=None if first == ZERO_HASH else first.hex(),
            problems=tuple(problems),
            warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class RecoverReport:
    """Result of :func:`recover`."""

    was_finalized: bool
    """The file was already finalized and was not changed."""
    recovery: RecoveryReport | None = None
    """What was removed, if anything."""

    @property
    def removed_bytes(self) -> int:
        return 0 if self.recovery is None else self.recovery.truncated_bytes

    @property
    def removed_frames(self) -> int:
        return 0 if self.recovery is None else self.recovery.discarded_frames


def recover(path: str | os.PathLike[str], *, force: bool = False) -> RecoverReport:
    """Finalizes a file whose writer did not close it: removes the unfinished
    transaction and writes a new index. With ``force``, data after a damaged
    frame in the middle of the file is discarded as well."""
    writer = Writer.open(path, WriteOptions(repair_corruption=force))
    report = RecoverReport(writer.recovery is None, writer.recovery)
    writer.close()
    return report


@dataclass(frozen=True)
class CompactReport:
    """Result of :func:`compact`."""

    version: int
    """Version of the source file that was written."""
    origin: str
    """Hash of that version (hexadecimal), recorded as the origin of the new file."""
    rows: int
    annotations: int
    bytes_before: int
    bytes_after: int


def compact(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    as_of: int | None = None,
    options: WriteOptions | None = None,
    overwrite: bool = False,
) -> CompactReport:
    """Writes the state of ``source`` at version ``as_of`` (the latest by
    default) into a new file without history and without overlapping blocks.
    The first commit of the new file links to the source version's hash."""
    if options is None:
        options = WriteOptions(overwrite=overwrite)
    with Reader(source) as reader:
        head = reader.head
        version = (head.number if head else 0) if as_of is None else as_of
        origin = ZERO_HASH if version == 0 else bytes.fromhex(reader.commit(version).hash)
        metadata = reader.metadata_as_of(version)
        metadata["trosna.compacted_from"] = origin.hex()
        metadata["trosna.compacted_from_commit"] = str(version)
        writer = Writer.create(target, options)
        try:
            writer.set_chain_origin(origin)
            for key, value in sorted(metadata.items(), key=lambda kv: kv[0].encode()):
                writer.set_metadata(key, value)
            rows = 0
            for schema in reader.devices:
                if not reader.device_exists_at(schema.name, version):
                    continue
                writer.create_device(schema)
                for batch in reader.query(schema.name).as_of(version).batches():
                    rows += len(batch)
                    writer.write(schema.name, batch)
            annotations = reader.annotations(as_of=version)
            for a in annotations:
                writer.restore_annotation(
                    a.id, a.device, start=a.start, end=a.end, label=a.label, note=a.note
                )
            writer.commit(message=f"compacted from {origin.hex()[:12]} at version {version}")
        except BaseException:
            writer.rollback()
            writer.close()
            raise
        writer.close()
        return CompactReport(
            version,
            origin.hex(),
            rows,
            len(annotations),
            reader.size,
            os.path.getsize(target),
        )
