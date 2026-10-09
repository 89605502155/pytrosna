# Changelog

All notable changes of pytrosna are documented here. The project follows
[Semantic Versioning](https://semver.org).

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
