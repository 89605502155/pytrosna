"""The golden file written by the reference Rust implementation pins the byte format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytrosna
from pytrosna import DeviceSchema, Reader, Writer

from .conftest import GOLDEN, GOLDEN_HEAD


def build(path: Path) -> None:
    """The scenario of crates/trosna/tests/golden.rs of the reference implementation."""
    with Writer.create(path, sync=False, rows_per_block=8, overwrite=True) as w:
        w.set_metadata("title", "Trosna golden file")
        w.create_device(
            DeviceSchema(
                "vm01",
                "ms",
                (
                    ("cpu", "float64"),
                    ("ram_mb", "int64"),
                    ("swap", "int32"),
                    ("ok", "bool"),
                    ("state", "string"),
                    ("temp", "float32"),
                ),
                timezone="UTC",
            )
        )
        n = 20
        w.write(
            "vm01",
            {
                "time": [1_700_000_000_000 + i * 1000 for i in range(n)],
                "cpu": np.array([0.25 + i / 100.0 for i in range(n)]),
                "ram_mb": np.array([2048 + i // 4 for i in range(n)]),
                "swap": [0 if i % 3 else None for i in range(n)],
                "ok": np.array([i % 7 != 6 for i in range(n)]),
                "state": ["работает" if i < 15 else "перегрузка" for i in range(n)],
                "temp": np.array([40.0 + i * 0.5 for i in range(n)], dtype=np.float32),
            },
        )
        w.commit(message="initial import", author="golden", time_ns=1_700_000_100_000_000_000)
        w.update("vm01", 1_700_000_003_000, cpu=0.99)
        w.delete_range("vm01", 1_700_000_010_000, 1_700_000_012_000)
        w.annotate("vm01", 1_700_000_015_000, 1_700_000_019_000, "overload", "перегрузка памяти")
        w.commit(message="corrections", time_ns=1_700_000_200_000_000_000)


def test_writer_reproduces_the_golden_file_byte_for_byte(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.trosna"
    build(fresh)
    assert fresh.read_bytes() == GOLDEN.read_bytes()


def test_golden_file_contents() -> None:
    with Reader(GOLDEN) as r:
        assert r.finalized
        assert r.head is not None
        assert r.head.hash == GOLDEN_HEAD
        assert r.metadata == {"title": "Trosna golden file"}
        now = r.read("vm01")
        before = r.read("vm01", as_of=1)
    assert len(before) == 20
    assert len(now) == 17
    assert now.row(3)["cpu"] == 0.99
    assert now["swap"].to_list()[:4] == [None, 0, 0, None]
    assert now["state"].to_list()[-1] == "перегрузка"
    assert pytrosna.verify(GOLDEN).ok
