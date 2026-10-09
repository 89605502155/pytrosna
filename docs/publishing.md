# Publishing pytrosna to PyPI

[Русская версия](publishing.ru.md)

1. Check `pyproject.toml` (`version`, `authors`, `[project.urls]`) and the
   absolute links in `README.md`.
2. Run the checks: `uv run ruff check . && uv run ruff format --check . &&
   uv run mypy && uv run pytest`.
3. Build and check: `rm -rf dist && uv build && uvx twine check dist/*`.
4. Optionally try TestPyPI:
   `uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-...`.
5. Publish: `uv publish --token pypi-...`.
6. For a new version bump `version` in `pyproject.toml` (e.g.
   `uv version --bump patch`; `pytrosna.__version__` follows it) and extend
   `CHANGELOG.md`.
