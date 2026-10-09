"""pytrosna — read, write and edit Trosna time-series files (``.trosna``) in pure Python.

Trosna is a columnar, log-structured file format for time series with
time-series encodings (delta-of-delta, Gorilla XOR, bit packing, RLE,
dictionaries), atomic hash-chained commits, point-level editing, interval
annotations and time travel. pytrosna is a pure-Python implementation of
format version 1, byte-compatible with the reference Rust implementation.

Read a device into pandas, Polars or PyArrow::

    import pytrosna

    df = pytrosna.read_pandas("metrics.trosna")

Write any table with a time column::

    pytrosna.write("metrics.trosna", df, device="vm01")

Edit, annotate and travel in time::

    f = pytrosna.open("metrics.trosna")
    with f.edit(message="fix a spike") as tx:
        tx.update("vm01", "2026-10-08 10:00:03", cpu=0.95)
        tx.annotate("vm01", "2026-10-08 10:00:00", "2026-10-08 10:05:00", "backup")
    old = f.read_pandas(as_of=1)

The low-level :class:`Writer` and :class:`Reader` work with raw integer time
stamps and NumPy-backed :class:`Batch` objects and need no other libraries.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from ._format import Statistics
from .api import (
    File,
    Transaction,
    compact,
    create,
    iter_batches,
    open,  # noqa: A004 - mirrors the built-in open, like gzip.open
    read,
    read_arrow,
    read_batches,
    read_pandas,
    read_polars,
    recover,
    verify,
    write,
)
from .codecs import Codec
from .column import Batch, Column
from .encodings import Encoding, EncodingPolicy
from .errors import (
    CorruptedError,
    DeviceExistsError,
    InvalidArgumentError,
    LimitExceededError,
    LockedError,
    NotFinalizedError,
    NotTrosnaError,
    PointNotFoundError,
    SchemaError,
    TrosnaError,
    TypeMismatchError,
    UnknownAnnotationError,
    UnknownColumnError,
    UnknownCommitError,
    UnknownDeviceError,
    UnsupportedError,
    UnsupportedVersionError,
)
from .history import (
    Annotation,
    AnnotationChange,
    ChangeKind,
    CommitChanges,
    CommitInfo,
    Diff,
    PointChange,
)
from .reader import BlockInfo, Query, Reader, SegmentInfo
from .timeconv import format_time, to_datetime, to_raw
from .tools import CompactReport, RecoverReport, VerifyReport
from .types import ColumnSchema, DataType, DeviceSchema, TimeUnit
from .writer import RecoveryReport, WriteOptions, Writer

try:
    __version__ = _metadata.version("pytrosna")
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

EXTENSION = "trosna"
"""The recommended file name extension, without the dot."""

FORMAT_VERSION = (1, 0)
"""The version of the Trosna format written by this library."""

__all__ = [
    "EXTENSION",
    "FORMAT_VERSION",
    "Annotation",
    "AnnotationChange",
    "Batch",
    "BlockInfo",
    "ChangeKind",
    "Codec",
    "Column",
    "ColumnSchema",
    "CommitChanges",
    "CommitInfo",
    "CompactReport",
    "CorruptedError",
    "DataType",
    "DeviceExistsError",
    "DeviceSchema",
    "Diff",
    "Encoding",
    "EncodingPolicy",
    "File",
    "InvalidArgumentError",
    "LimitExceededError",
    "LockedError",
    "NotFinalizedError",
    "NotTrosnaError",
    "PointChange",
    "PointNotFoundError",
    "Query",
    "Reader",
    "RecoverReport",
    "RecoveryReport",
    "SchemaError",
    "SegmentInfo",
    "Statistics",
    "TimeUnit",
    "Transaction",
    "TrosnaError",
    "TypeMismatchError",
    "UnknownAnnotationError",
    "UnknownColumnError",
    "UnknownCommitError",
    "UnknownDeviceError",
    "UnsupportedError",
    "UnsupportedVersionError",
    "VerifyReport",
    "WriteOptions",
    "Writer",
    "__version__",
    "compact",
    "create",
    "format_time",
    "iter_batches",
    "open",
    "read",
    "read_arrow",
    "read_batches",
    "read_pandas",
    "read_polars",
    "recover",
    "to_datetime",
    "to_raw",
    "verify",
    "write",
]
