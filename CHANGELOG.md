# Changelog

All notable changes of pytrosna are documented here. The project follows
[Semantic Versioning](https://semver.org).

## 0.1.1 — 2026-10-09

Fixes for Windows.

* The file lock no longer fails on Windows systems that refuse lock offsets
  beyond 2 GiB (it falls back to an offset below 2 GiB), and it is released
  explicitly before the file is closed, so a file can be reopened at once.
* The `pytrosna` command writes UTF-8 even when its output is redirected on
  Windows, where the code page (e.g. cp1252) cannot encode Cyrillic device
  names, labels or values.
* Tests: Windows locking is tested with a stand-in `msvcrt` module, the CLI
  with a legacy code page; property tests no longer have a time limit.
* The GitHub Actions workflows were removed; the package is published with
  `uv publish`.

## 0.1.0 — 2026-10-09

First release: a pure-Python implementation of the Trosna file format,
version 1.0.

* Reading and writing of `.trosna` files, byte-compatible with the reference
  Rust implementation (the golden file is reproduced byte for byte).
* All encodings of the format (plain, delta-bitpack, delta-of-delta, RLE,
  Gorilla XOR, dictionary) with adaptive per-segment selection, and the
  codecs none, LZ4 and Zstandard.
* Editing: inserts, updates, deletions of points and ranges, interval
  annotations, file and device metadata; atomic, SHA-256 hash-chained
  commits; rollback; time travel (`as_of`) and differences between versions.
* Crash safety: recovery of unfinished files, refusal to truncate files
  damaged in the middle, `verify`, `recover` and `compact`.
* High-level API (`pytrosna.open`, `read_*`, `write`, `create`, transactions)
  with adapters for pandas, Polars and PyArrow; a NumPy-only low-level API
  (`Writer`, `Reader`, `Batch`).
* The `pytrosna` command-line tool (CSV import and export, inspection,
  editing, history, maintenance).
* Documentation in English and Russian; about 400 tests including
  property-based, model-based, crash-simulation, corruption and
  interoperability tests.
