"""Shared fixtures of the test suite."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import pytrosna
from pytrosna import DeviceSchema, Writer

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "golden_v1.trosna"
GOLDEN_HEAD = "2764d75ada9d460758cb9d1cdea72a0039bb93f3dc7b0e5e6ef9fe681b55cdd4"

T0 = 1_700_000_000_000  # 2023-11-14T22:13:20Z in milliseconds

SCHEMA = DeviceSchema.build(
    "vm01",
    "ms",
    {
        "cpu": "float64",
        "ram_mb": "int64",
        "swap": "int32",
        "ok": "bool",
        "state": "string",
        "temp": "float32",
    },
    timezone="UTC",
)


def sample_columns(n: int, seed: int = 0) -> dict[str, object]:
    """Columns of SCHEMA with a few nulls, as a mapping for Writer.write."""
    rng = np.random.default_rng(seed)
    swap = [None if i % 3 == 0 else int(v) for i, v in enumerate(rng.integers(0, 10, n))]
    return {
        "time": T0 + np.arange(n, dtype=np.int64) * 1000,
        "cpu": np.round(0.25 + np.cumsum(rng.normal(0, 0.01, n)), 4),
        "ram_mb": 2048 + np.arange(n) // 4,
        "swap": swap,
        "ok": rng.random(n) < 0.8,
        "state": ["работает" if v < 0.7 else "перегрузка" for v in rng.random(n)],
        "temp": (40 + rng.random(n) * 10).astype(np.float32),
    }


@pytest.fixture
def path(tmp_path: Path) -> Path:
    """A path for a new Trosna file."""
    return tmp_path / "test.trosna"


@pytest.fixture
def sample_file(path: Path) -> Path:
    """A file with device vm01: 20 rows (commit 1), an update, a deletion and
    an annotation (commit 2) — the same history as the golden file."""
    with Writer.create(path, sync=False, rows_per_block=8) as w:
        w.set_metadata("title", "sample")
        w.create_device(SCHEMA)
        w.write("vm01", sample_columns(20))
        w.commit(message="initial import", author="tests", time_ns=1_000)
        w.update("vm01", T0 + 3000, cpu=0.99)
        w.delete_range("vm01", T0 + 10_000, T0 + 12_000)
        w.annotate("vm01", T0 + 15_000, T0 + 19_000, "overload", "перегрузка памяти")
        w.commit(message="corrections", time_ns=2_000)
    return path


def _find_rust_cli() -> str | None:
    candidate = os.environ.get("TROSNA_CLI") or shutil.which("trosna")
    if candidate and os.access(candidate, os.X_OK):
        return candidate
    return None


@pytest.fixture(scope="session")
def rust_cli() -> str:
    """The reference ``trosna`` command (set TROSNA_CLI); skips the test if missing."""
    cli = _find_rust_cli()
    if cli is None:
        pytest.skip("the reference Rust CLI is not available (set TROSNA_CLI)")
    return cli


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace: dict[str, object]) -> None:
    doctest_namespace["pytrosna"] = pytrosna
    doctest_namespace["np"] = np
