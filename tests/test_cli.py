"""The pytrosna command-line tool."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import pytrosna
from pytrosna.cli import _guess_unit, _infer_csv_type, main
from pytrosna.types import DataType, TimeUnit

CSV = """time,temperature,humidity,ok,state
2026-10-08T10:00:00+03:00,21.5,40,true,ok
2026-10-08T10:00:10+03:00,21.6,41,true,
2026-10-08T10:00:20+03:00,21.7,,false,warn
"""


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def room(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    csv_path = tmp_path / "room.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    target = tmp_path / "room.trosna"
    code, out, _ = run(capsys, "convert", csv_path, target, "--device", "room1", "-m", "import")
    assert code == 0
    assert "wrote 3 rows" in out
    return target


def test_import_detects_types(room: Path) -> None:
    f = pytrosna.open(room)
    d = f.device()
    assert d.timezone == "+03:00"
    assert [c.data_type for c in d.columns] == [
        DataType.FLOAT64,
        DataType.INT64,
        DataType.BOOL,
        DataType.STRING,
    ]
    df = f.read_pandas()
    assert df["state"].tolist()[0] == "ok"
    assert f.head is not None
    assert f.head.message == "import"


def test_info_and_cat(room: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "info", room)
    assert code == 0
    assert "device 'room1': 3 points" in out
    assert "time zone +03:00" in out
    assert "delta-of-delta" in out or "delta-bitpack" in out
    _, out, _ = run(capsys, "cat", room)
    assert out.splitlines()[1] == "2026-10-08T10:00:00.000+03:00,21.5,40,true,ok"
    _, out, _ = run(
        capsys, "cat", room, "--format", "table", "--columns", "temperature", "--limit", "2"
    )
    assert len(out.splitlines()) == 4
    _, out, _ = run(
        capsys, "cat", room, "--format", "json", "--raw-time", "--from", "2026-10-08 10:00:10"
    )
    records = [json.loads(line) for line in out.splitlines()]
    assert [r["humidity"] for r in records] == [41, None]
    _, out, _ = run(capsys, "cat", room, "--to", "2026-10-08 10:00:05", "--limit", "5")
    assert len(out.splitlines()) == 2


def test_editing_commands(room: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        run(
            capsys,
            "update",
            room,
            "--time",
            "2026-10-08 10:00:20",
            "temperature=21.65",
            "-m",
            "fix",
        )[0]
        == 0
    )
    assert (
        run(
            capsys,
            "insert",
            room,
            "--time",
            "2026-10-08 10:01:00",
            "temperature=22",
            "ok=yes",
            "state=null",
        )[0]
        == 0
    )
    _, out, _ = run(
        capsys,
        "annotate",
        room,
        "--start",
        "2026-10-08 10:00:05",
        "--end",
        "2026-10-08 10:00:15",
        "--label",
        "door open",
        "--note",
        "draft",
    )
    assert "annotation 1" in out
    assert run(capsys, "annotate", room, "--id", "1", "--label", "door", "--author", "me")[0] == 0
    _, out, _ = run(capsys, "annotations", room)
    assert "door — draft" in out
    assert run(capsys, "delete", room, "--time", "2026-10-08 10:00:10")[0] == 0
    assert run(capsys, "delete", room, "--from", "2026-10-08 10:00:50")[0] == 0
    _, out, _ = run(capsys, "log", room)
    assert out.count("commit ") == 7
    assert "Author:  me" in out
    _, out, _ = run(capsys, "diff", room, "--from", "1")
    assert "~ 2026-10-08T10:00:20.000+03:00  temperature: 21.7 → 21.65" in out
    assert "- 2026-10-08T10:00:10.000+03:00" in out
    assert "added annotation 1: door" in out
    _, out, _ = run(capsys, "diff", room, "--from", "2", "--to", "2")
    assert "no changes" in out
    assert run(capsys, "unannotate", room, "1")[0] == 0
    _, out, _ = run(capsys, "annotations", room)
    assert out.strip() == "no annotations"
    assert pytrosna.open(room).count() == 2


def test_verify_recover_compact(
    room: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(capsys, "verify", room)
    assert code == 0
    assert out.strip().endswith("OK")
    _, out, _ = run(capsys, "recover", room)
    assert "nothing to do" in out
    room.write_bytes(room.read_bytes()[:-24])
    _, out, _ = run(capsys, "recover", room)
    assert "recovered" in out
    small = tmp_path / "small.trosna"
    _, out, _ = run(capsys, "compact", room, small)
    assert "compacted version 1" in out
    _, out, _ = run(capsys, "verify", small)
    assert "origin:" in out
    data = bytearray(room.read_bytes())
    data[30] ^= 0xFF
    room.write_bytes(bytes(data))
    code, out, _ = run(capsys, "verify", room)
    assert code == 1
    assert "DAMAGED" in out


def test_export_and_append(room: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    back = tmp_path / "back.csv"
    assert run(capsys, "convert", room, back)[0] == 0
    assert back.read_text().splitlines()[0] == "time,temperature,humidity,ok,state"
    code, _, err = run(capsys, "convert", room, back)
    assert code == 2
    assert "exists" in err
    assert run(capsys, "convert", room, back, "--force", "--raw-time", "--delimiter", ";")[0] == 0
    assert back.read_text().splitlines()[1].split(";")[0].isdigit()
    more = tmp_path / "more.csv"
    more.write_text(
        "time,temperature,humidity,ok,state\n2026-10-08T10:00:30+03:00,22.0,42,true,ok\n"
    )
    code, _, err = run(capsys, "convert", more, room)
    assert code == 2
    assert "--append" in err
    assert run(capsys, "convert", more, room, "--append", "--device", "room1")[0] == 0
    assert pytrosna.open(room).count() == 4


def test_integer_epochs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "epochs.csv"
    src.write_text("ts;value\n1700000000000;1.5\n1700000001000;2.5\n")
    target = tmp_path / "epochs.trosna"
    assert (
        run(capsys, "convert", src, target, "--delimiter", ";", "--time", "ts", "--codec", "lz4")[0]
        == 0
    )
    f = pytrosna.open(target)
    assert f.device().name == "epochs"
    assert f.device().time_unit is TimeUnit.MILLISECOND
    assert f.device().time_name == "ts"


def test_csv_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "x.trosna"
    cases = {
        "empty.csv": ("", "is empty"),
        "dup.csv": ("a,a\n1,2\n", "duplicate"),
        "ragged.csv": ("time,v\n1,2,3\n", "fields"),
        "gap.csv": ("time,v\n,2\n", "empty cells"),
        "header.csv": ("time,v\n", "no rows"),
    }
    for name, (content, message) in cases.items():
        src = tmp_path / name
        src.write_text(content)
        code, _, err = run(capsys, "convert", src, target)
        assert code == 2, name
        assert message in err, (name, err)
    src = tmp_path / "ok.csv"
    src.write_text("time,v\n1,2\n")
    _, _, err = run(capsys, "convert", src, target, "--time", "when")
    assert "no column" in err


def test_command_errors(room: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(capsys, "update", room, "--time", "2026-10-08 11:00", "temperature=1")[0] == 2
    _, _, err = run(capsys, "insert", room, "--time", "2026-10-08 11:00", "nope=1")
    assert "unknown column" in err
    _, _, err = run(capsys, "insert", room, "--time", "2026-10-08 11:00", "temperature")
    assert "column=value" in err
    _, _, err = run(capsys, "insert", room, "--time", "2026-10-08 11:00", "humidity=many")
    assert "cannot read" in err
    _, _, err = run(capsys, "insert", room, "--time", "2026-10-08 11:00", "ok=maybe")
    assert "cannot read" in err
    _, _, err = run(capsys, "delete", room)
    assert "--time" in err
    _, _, err = run(capsys, "annotate", room, "--label", "x")
    assert "--start" in err
    _, _, err = run(capsys, "annotate", room, "--id", "9", "--label", "x")
    assert "unknown annotation" in err
    _, _, err = run(capsys, "cat", room, "--device", "nope")
    assert "unknown device" in err
    code, _, err = run(capsys, "cat", room.with_suffix(".missing"))
    assert code == 2
    code, _, err = run(capsys, "unannotate", room, "5")
    assert code == 2
    code, _, _ = run(capsys, "update", room, "--time", "2026-10-08 10:00", "temperature=21.5")
    assert code == 0


def test_several_devices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "two.trosna"
    pytrosna.create(
        path,
        pytrosna.DeviceSchema.build("a", "s", {"x": "int64"}),
        pytrosna.DeviceSchema.build("b", "s", {"x": "int64"}),
    )
    _, _, err = run(capsys, "cat", path)
    assert "several devices" in err
    assert run(capsys, "cat", path, "--device", "b")[0] == 0
    _, out, _ = run(capsys, "log", path)
    assert "new devices: a, b" in out
    empty = tmp_path / "empty.trosna"
    pytrosna.create(empty)
    _, out, _ = run(capsys, "info", empty)
    assert "no commits" in out  # a file without devices has nothing to commit
    _, _, err = run(capsys, "cat", empty)
    assert "no devices" in err
    blank = tmp_path / "blank.trosna"
    pytrosna.Writer.create(blank).close()
    _, out, _ = run(capsys, "log", blank)
    assert out.strip() == "no commits"
    _, out, _ = run(capsys, "info", blank)
    assert "no commits" in out


def test_helpers() -> None:
    assert _guess_unit([1_700_000_000]) is TimeUnit.SECOND
    assert _guess_unit([1_700_000_000_000]) is TimeUnit.MILLISECOND
    assert _guess_unit([1_700_000_000_000_000]) is TimeUnit.MICROSECOND
    assert _guess_unit([1_700_000_000_000_000_000]) is TimeUnit.NANOSECOND
    assert _infer_csv_type(["1", "", "2"]) is DataType.INT64
    assert _infer_csv_type(["1", "2.5"]) is DataType.FLOAT64
    assert _infer_csv_type(["TRUE", "false"]) is DataType.BOOL
    assert _infer_csv_type(["true", "1"]) is DataType.STRING
    assert _infer_csv_type(["x"]) is DataType.STRING
    assert _infer_csv_type([""]) is DataType.STRING


def test_version_and_module_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "pytrosna 0.1.0" in capsys.readouterr().out
    result = subprocess.run(
        [sys.executable, "-m", "pytrosna", "--help"], capture_output=True, text=True, check=True
    )
    assert "convert" in result.stdout
