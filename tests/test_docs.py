"""The Python examples of the documentation run without errors.

All ``python`` code blocks of a document are executed in order in one
namespace, in a temporary working directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "README.ru.md",
    ROOT / "docs" / "guide.md",
    ROOT / "docs" / "guide.ru.md",
]
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _name(path: Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_code_blocks_run(document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not document.exists():
        pytest.skip(f"{document.name} is missing")
    blocks = BLOCK.findall(document.read_text(encoding="utf-8"))
    assert blocks, f"{document} has no Python examples"
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for number, code in enumerate(blocks, start=1):
        try:
            exec(compile(code, f"{document.name}[{number}]", "exec"), namespace)  # noqa: S102
        except Exception as e:  # pragma: no cover - reported as a failure
            pytest.fail(f"{document.name}, code block {number} failed: {e!r}\n{code}")
    assert os.listdir(tmp_path)
