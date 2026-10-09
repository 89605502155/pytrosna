# Публикация pytrosna на PyPI

Инструкция для автора пакета. Всё делается через `uv`. Имя `pytrosna`
на PyPI на 9 октября 2026 г. свободно.

## 1. Перед публикацией

1. Проверьте метаданные в `pyproject.toml`: `version`, `authors`, ссылки в
   `[project.urls]`. Сейчас они указывают на
   `https://github.com/89605502155/pytrosna`; если репозиторий будет называться
   иначе, исправьте ссылки там же и в `README.md` / `README.ru.md` (PyPI
   показывает README, поэтому ссылки в нём абсолютные).
2. Прогоните проверки:

   ```sh
   uv sync
   uv run ruff check . && uv run ruff format --check .
   uv run mypy
   uv run pytest
   # с эталонной Rust-утилитой — ещё и тесты совместимости:
   TROSNA_CLI=/путь/к/trosna uv run pytest -m interop
   ```

3. Соберите пакет и проверьте его:

   ```sh
   rm -rf dist
   uv build                  # dist/pytrosna-0.1.0.tar.gz и dist/pytrosna-0.1.0-py3-none-any.whl
   uvx twine check dist/*
   ```

## 2. Пробная публикация на TestPyPI (рекомендуется)

1. Зарегистрируйтесь на <https://test.pypi.org> и создайте API-токен
   (Account settings → API tokens).
2. Опубликуйте:

   ```sh
   uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-ВАШ_ТОКЕН
   ```

3. Проверьте установку в чистом окружении:

   ```sh
   uv venv /tmp/check && uv pip install --python /tmp/check/bin/python \
       --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "pytrosna[all]"
   /tmp/check/bin/python -c "import pytrosna; print(pytrosna.__version__)"
   ```

## 3. Публикация на PyPI

### Вариант А: вручную с токеном

1. Зарегистрируйтесь на <https://pypi.org>, включите двухфакторную
   аутентификацию и создайте API-токен (для первой публикации — с областью
   «Entire account», после неё лучше заменить на токен только для проекта
   `pytrosna`).
2. Опубликуйте:

   ```sh
   uv publish --token pypi-ВАШ_ТОКЕН
   ```

   Токен можно передать и через переменную окружения `UV_PUBLISH_TOKEN`.

### Вариант Б: автоматически из GitHub (trusted publishing, без токенов)

1. Создайте на GitHub репозиторий и загрузите в него проект.
2. На PyPI: Account → Publishing → «Add a new pending publisher»:
   имя проекта `pytrosna`, владелец и имя репозитория, workflow
   `publish.yml`, environment `pypi`.
3. На GitHub: Settings → Environments → создайте окружение `pypi`.
4. Поставьте тег версии и отправьте его:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

   Workflow `.github/workflows/publish.yml` прогонит тесты, соберёт пакет и
   опубликует его.

## 4. Новые версии

1. Увеличьте `version` в `pyproject.toml` (например, командой
   `uv version --bump patch`; `pytrosna.__version__` берётся оттуда
   автоматически) и допишите `CHANGELOG.md`.
2. Повторите шаги 1 и 3. Опубликованную версию на PyPI нельзя заменить —
   только выпустить новую.
