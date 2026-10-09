"""The examples run without errors."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_python_example(script: Path, tmp_path: Path) -> None:
    for _ in range(2):  # an example can be run again
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=tmp_path,
            # the examples print tables with box-drawing characters
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert result.returncode == 0, result.stderr


def test_shell_example() -> None:
    sh = shutil.which("sh")
    command = Path(sys.executable).with_name("pytrosna")
    if sh is None or not command.exists():
        pytest.skip("needs sh and the installed pytrosna command")
    env = {**os.environ, "PATH": f"{command.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    result = subprocess.run(
        [sh, str(EXAMPLES / "cli_demo.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
