# pytrosna

**Чтение, запись и редактирование файлов временных рядов Trosna (`.trosna`) на чистом Python.**

[English version](https://github.com/89605502155/pytrosna/blob/main/README.md) ·
[Руководство](https://github.com/89605502155/pytrosna/blob/main/docs/guide.ru.md) ·
[Справочник API](https://github.com/89605502155/pytrosna/blob/main/docs/api.ru.md) ·
[Примеры](https://github.com/89605502155/pytrosna/tree/main/examples) ·
[Спецификация формата](https://github.com/89605502155/trosna-file/blob/main/docs/SPEC.md)

Trosna — формат файлов для временных рядов. Файл `.trosna` создаётся и
читается так же просто, как CSV, но хранит данные так, как это делают СУБД
временных рядов: по столбцам, с кодированиями, рассчитанными на метки
времени и измерения, и с проверяемой историей всех изменений.

`pytrosna` — реализация версии 1.0 формата на чистом Python, полностью
совместимая с эталонной реализацией на Rust
([trosna-file](https://github.com/89605502155/trosna-file)): файлы,
записанные любой из них, читаются, редактируются и проверяются другой.
Писатель использует те же алгоритмы, вплоть до выбора кодирований, и
воспроизводит эталонный (golden) файл Rust-реализации байт в байт. (Если
сегмент сжат Zstandard, библиотеки могут получить разные, но равноценные
сжатые байты.)

* **Компактно.** Каждый сегмент столбца кодируется самым коротким из
  кодирований delta-of-delta, битовой упаковки в стиле TS_2DIFF, Gorilla
  XOR, RLE и словарного, а затем сжимается Zstandard или LZ4. Ряд показаний
  датчика занимает **1,5 байта на точку** (CSV — 16,9, Parquet + zstd — 2,6).
* **Редактирование с историей.** Точки можно добавлять, изменять и удалять в
  любом месте оси времени, а интервалы — размечать. Каждое изменение —
  атомарный коммит; любая прежняя версия остаётся доступной (`as_of=`).
  Коммиты образуют цепочку хешей SHA-256, поэтому подмена истории
  обнаруживается.
* **Устойчивость к сбоям.** Файл — журнал фреймов с контрольными суммами,
  в который только дописывают. После сбоя он читается в состоянии последнего
  коммита; следующий писатель удаляет незавершённый хвост.
* **Несколько устройств в одном файле**, у каждого свои столбцы, единица
  времени (s, ms, us, ns) и часовой пояс.
* **Работа со стеком данных Python.** Чтение в pandas, Polars и PyArrow и
  запись из них — или пакеты (`Batch`) на NumPy без этих библиотек.
* **Чистый Python.** Компилятор не нужен: зависимости — NumPy, `lz4`,
  `crc32c` и, до Python 3.14, `zstandard`.

## Установка

```sh
pip install pytrosna                 # ядро на NumPy
pip install "pytrosna[pandas]"       # + pandas
pip install "pytrosna[polars]"       # + Polars
pip install "pytrosna[arrow]"        # + PyArrow
pip install "pytrosna[all]"          # всё сразу

uv add "pytrosna[all]"               # через uv
```

Нужен Python 3.11 или новее.

## Быстрый старт

```python
import pandas as pd
import pytrosna

df = pd.DataFrame(
    {
        "time": pd.date_range("2026-10-08 10:00", periods=5, freq="10s", tz="Europe/Moscow"),
        "temperature": [21.5, 21.6, 21.7, 21.6, 21.8],
        "humidity": [40, 41, 41, 42, 42],
    }
)

pytrosna.write("room.trosna", df, device="room1")  # создать файл (один коммит)

pytrosna.read_pandas("room.trosna")  # всё — в pandas
# интервал и столбец — в Polars:
pytrosna.read_polars("room.trosna", start="2026-10-08 10:00:10", columns=["temperature"])
pytrosna.read_arrow("room.trosna")  # pyarrow.Table
pytrosna.read("room.trosna")  # pytrosna.Batch на NumPy
```

Время можно задавать как `datetime`, `pandas.Timestamp`, `numpy.datetime64`,
текст ISO 8601 или «сырое» целое число. Текст без смещения UTC означает
местное время часового пояса устройства.

## Правки, аннотации и путешествие во времени

```python
f = pytrosna.open("room.trosna")

with f.edit(message="проверка датчика", author="я") as tx:  # один атомарный коммит
    tx.update("room1", "2026-10-08 10:00:20", temperature=21.65)
    tx.delete("room1", "2026-10-08 10:00:10")
    tx.insert("room1", "2026-10-08 10:01:00", temperature=22.0)
    tx.annotate("room1", "2026-10-08 10:00:15", "2026-10-08 10:00:25", "дверь открыта")
# если внутри блока возникло исключение, ничего не записывается

f.read_pandas()  # текущие данные
f.read_pandas(as_of=1)  # данные в момент первого импорта
f.diff(1).points  # что изменилось после коммита 1
f.annotations()  # размеченные интервалы
f.commits()  # история — цепочка хешей
f.verify().ok  # контрольные суммы, цепочка хешей, декодирование данных
f.compact("small.trosna")  # копия без истории, связанная с ней по хешу
```

Дозапись так же проста:

```python
more_rows = df.assign(time=df["time"] + pd.Timedelta("1h"))
pytrosna.write("room.trosna", more_rows, device="room1", mode="a")
```

## Низкоуровневый API

`Writer` и `Reader` работают с целыми метками времени и массивами NumPy и
не требуют других библиотек:

```python
import numpy as np
from pytrosna import DeviceSchema, Reader, Writer

with Writer.create("vms.trosna") as w:  # при закрытии — коммит
    w.create_device(DeviceSchema.build("vm01", "ms", {"cpu": "float64", "ram_mb": "int64"}))
    w.write(
        "vm01",
        {
            "time": np.arange(0, 10_000, 1000),
            "cpu": np.random.rand(10),
            "ram_mb": np.full(10, 2048),
        },
    )
    w.commit(message="первые измерения")
    w.update("vm01", 3000, cpu=0.99)
    w.delete_range("vm01", 7000, 9000)

with Reader("vms.trosna") as r:
    batch = r.query("vm01").columns(["cpu"]).time_range(0, 5000).collect()
    batch.time, batch["cpu"].values  # массивы NumPy
    r.query("vm01").as_of(1).count()  # 10 точек в версии 1
```

## Командная строка

Пакет устанавливает команду `pytrosna`:

```sh
pytrosna convert room.csv room.trosna --device room1    # CSV → Trosna (типы определяются сами)
pytrosna info room.trosna                               # устройства, история, объём по столбцам
pytrosna cat room.trosna --from "2026-10-08 10:00" --format table
pytrosna update room.trosna --time "2026-10-08 10:00:20" temperature=21.65 -m "проверка датчика"
pytrosna annotate room.trosna --start "2026-10-08 10:00:15" --end "2026-10-08 10:00:25" --label "дверь открыта"
pytrosna log room.trosna
pytrosna diff room.trosna --from 1
pytrosna verify room.trosna
pytrosna convert room.trosna back.csv                   # Trosna → CSV
```

`pytrosna --help` выводит все команды: `info`, `cat`, `convert`, `insert`,
`update`, `delete`, `annotate`, `unannotate`, `annotations`, `log`, `diff`,
`verify`, `recover`, `compact`.

## Объём и скорость

Миллион показаний медленно меняющегося датчика (значения округлены до 0,01,
одно в секунду), сжатие Zstandard:

| Формат                      | Байт на точку |
|-----------------------------|---------------|
| **Trosna (pytrosna)**       | **1,49**      |
| Arrow IPC (Feather), zstd   | 2,18          |
| Parquet, zstd               | 2,56          |
| CSV, gzip                   | 3,14          |
| CSV                         | 16,90         |

pytrosna записывает около 0,8 и читает около 0,6 млн точек в секунду на
ноутбуке: кодировщики и декодеры векторизованы на NumPy везде, где это
позволяет формат. Если скорость важнее отсутствия компилируемых
зависимостей, реализация на Rust быстрее; файлы у них одинаковые.

## Совместимость и тестирование

* Версия формата 1.0 по
  [SPEC.md](https://github.com/89605502155/trosna-file/blob/main/docs/SPEC.md).
* Писатель воспроизводит golden-файл Rust-реализации **байт в байт**; набор
  тестов (около 400 тестов, покрытие строк 98 %) включает тесты
  совместимости, которые запускают эталонную утилиту `trosna` на файлах
  pytrosna и наоборот. Примеры кода из этого README и руководств тоже
  выполняются тестами.
* Тесты на основе свойств сравнивают каждый кодировщик с эталонной
  реализацией, написанной прямо по спецификации; модельный тест проверяет
  семантику правок на каждой версии; файлы обрезаются на каждом байте и
  портятся случайными битами, чтобы проверить восстановление и обработку
  ошибок.

## Разработка

```sh
uv sync                       # окружение со всеми зависимостями и инструментами
uv run pytest                 # тесты (TROSNA_CLI=/путь/к/trosna включает тесты совместимости)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv build                      # sdist и wheel в dist/
```

Как опубликовать пакет на PyPI — в
[docs/publishing.ru.md](https://github.com/89605502155/pytrosna/blob/main/docs/publishing.ru.md).

## Цитирование

Если вы используете Trosna в исследованиях, пожалуйста, ссылайтесь на обзор,
который послужил основой формата:

> Ферубко А. О., Казаков О. Д. Обзор структур данных и форматов файлов для
> хранения и передачи временных рядов с учётом их поддержки в экосистемах
> Python и Rust. 2026.

## Лицензия

На ваш выбор: [Apache License, Version 2.0](LICENSE-APACHE) или
[MIT](LICENSE-MIT).
