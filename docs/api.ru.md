# Справочник API pytrosna

[English version](api.md) · [Руководство](guide.ru.md) · [README](../README.ru.md)

Всё перечисленное импортируется из пакета верхнего уровня
(`import pytrosna`). *Сырое* время — целое число в единице времени
устройства; *любое время* — сырое целое, `datetime`/`date`,
`pandas.Timestamp`, `numpy.datetime64` или текст ISO 8601 (см.
[Время](#время)).

* [Функции модуля](#функции-модуля)
* [File и Transaction](#file-и-transaction)
* [Writer и WriteOptions](#writer-и-writeoptions)
* [Reader и Query](#reader-и-query)
* [Batch и Column](#batch-и-column)
* [Схемы и типы](#схемы-и-типы)
* [Объекты истории](#объекты-истории)
* [Сведения о хранении](#сведения-о-хранении)
* [Отчёты обслуживания](#отчёты-обслуживания)
* [Время](#время)
* [Адаптеры](#адаптеры)
* [Ошибки](#ошибки)
* [Константы](#константы)

## Функции модуля

| Функция | Описание |
|---|---|
| `open(path) -> File` | Открывает файл для чтения, правок и работы с историей. |
| `read(path, device=None, *, columns=None, start=None, end=None, as_of=None) -> Batch` | Читает устройство в `Batch` на NumPy. `start`/`end` включительно (любое время); `as_of` — номер коммита (0 — пустой файл). `device` можно опустить, если устройство одно. |
| `read_pandas(path, device=None, *, dtype_backend="numpy", **query)` | Читает в `pandas.DataFrame`; время — первый столбец. |
| `read_polars(path, device=None, **query)` | Читает в `polars.DataFrame`. |
| `read_arrow(path, device=None, **query)` | Читает в `pyarrow.Table`. |
| `read_batches(path, device=None, **query)` | Потоковое чтение как `pyarrow.RecordBatchReader`. |
| `iter_batches(path, device=None, **query) -> Iterator[Batch]` | Потоковое чтение объектов `Batch`, примерно по одному на хранимый блок. |
| `write(path, data, device, *, time_column=None, unit=None, mode="w", message=None, author=None, **options) -> CommitInfo \| None` | Записывает таблицу в устройство одним коммитом. `data`: DataFrame (или LazyFrame) pandas/Polars, Table/RecordBatch/RecordBatchReader PyArrow, объект с `__arrow_c_stream__`, `Batch` или словарь `{столбец: значения}`. `unit` — единица целых меток времени. `mode`: `"w"` заменить, `"x"` создать новый, `"a"` дописать (создаёт файл или устройство, если их нет). `options` — поля `WriteOptions`. Возвращает `None`, если ничего не записано. |
| `create(path, *devices, metadata=None, overwrite=False, message=None, author=None) -> File` | Создаёт файл с устройствами (`DeviceSchema`) и метаданными. |
| `verify(path) -> VerifyReport` | Полная проверка целостности. |
| `recover(path, *, force=False) -> RecoverReport` | Завершает файл, запись в который прервалась; `force` также отбрасывает данные после повреждения в середине файла. |
| `compact(source, target, *, as_of=None, overwrite=False) -> CompactReport` | Копирует версию (по умолчанию последнюю) в новый файл без истории. |
| `to_raw(value, unit, tz=None, rounding="floor") -> int` | Переводит любое время в сырую метку. |
| `to_datetime(raw, unit, tz=None) -> datetime` | Переводит сырую метку в `datetime`. |
| `format_time(raw, unit, tz=None) -> str` | Текст ISO 8601 с полной точностью единицы. |

## File и Transaction

### `class File(path)`

Файл Trosna; при открытии читается только индекс. Атрибуты (обновляются
методом `reload()` и после каждой записи через объект):

| Атрибут | Тип | Значение |
|---|---|---|
| `path` | `str` | имя файла |
| `finalized` | `bool` | `False`, если файл не был закрыт правильно (он всё равно читается) |
| `format_version` | `tuple[int, int]` | из заголовка файла |
| `size` | `int` | байт |
| `metadata` | `dict[str, str]` | метаданные последней версии |
| `devices` | `dict[str, DeviceSchema]` | устройства по именам |

| Метод | Описание |
|---|---|
| `device(name=None) -> DeviceSchema` | Устройство по имени или единственное. |
| `read(device=None, *, columns, start, end, as_of) -> Batch` | См. `pytrosna.read`. |
| `read_pandas(device=None, *, dtype_backend="numpy", **query)` / `read_polars(...)` / `read_arrow(...)` / `read_batches(...)` / `iter_batches(...)` | Как одноимённые функции модуля. |
| `count(device=None, *, start=None, end=None, as_of=None) -> int` | Число точек (читаются только метки времени). |
| `time_range(device=None) -> tuple[int, int] \| None` | Первая и последняя сырые метки. |
| `to_raw(value, device=None, rounding="floor") -> int`, `to_datetime(raw, device=None)` | Перевод времени в единице и поясе устройства. |
| `write(data, device, **options) -> CommitInfo \| None` | Дозапись (`mode="a"`). |
| `edit(message=None, author=None) -> Transaction` | Начинает транзакцию. |
| `writer(options=None, **kwargs) -> Writer` | Открывает низкоуровневый писатель. |
| `commits() -> list[CommitInfo]`, `head`, `commit(number)` | История. |
| `commit_at(when) -> int` | Версия на заданный момент (любое время; наивное — в UTC). |
| `annotations(device=None, *, as_of=None) -> list[Annotation]` | Аннотации по возрастанию идентификаторов. |
| `diff(from_commit, to_commit=None, *, device=None) -> Diff` | Изменения между двумя версиями. |
| `blocks(device=None) -> list[BlockInfo]` | Хранимые блоки с кодированиями и статистикой. |
| `verify() -> VerifyReport`, `compact(target, *, as_of=None, overwrite=False) -> File`, `reload()` | Обслуживание. |

### `class Transaction`

Возвращается `File.edit()`. Изменения применяются одним коммитом, когда блок
`with` завершается без исключения (или при вызове `apply()`); иначе не
применяются.

| Метод | Описание |
|---|---|
| `insert(device, time, values=None, **kw)` | Добавляет точку или заменяет точку с тем же временем; не указанные столбцы пусты. `time` должно точно выражаться в единице устройства. |
| `update(device, time, values=None, **kw)` | Меняет часть значений существующей точки (иначе `PointNotFoundError`). |
| `delete(device, time)` | Удаляет точку `time`, если она есть. |
| `delete_range(device, start=None, end=None)` | Удаляет `start <= t <= end` (открытые концы при `None`). |
| `write(device, data, *, time_column=None, unit=None)` | Записывает целую таблицу в существующее устройство. |
| `annotate(device, start, end, label, note=None)` | Добавляет аннотацию. |
| `update_annotation(annotation_id, *, start=None, end=None, label=None, note=None)` | Меняет аннотацию; `None` сохраняет значение, `note=""` удаляет примечание. |
| `remove_annotation(annotation_id)` | Удаляет аннотацию. |
| `set_metadata(key, value)`, `remove_metadata(key)` | Метаданные файла. |
| `apply() -> CommitInfo \| None` | Применяет немедленно. |

Атрибуты после коммита: `commit` (`CommitInfo` или `None`) и
`annotation_ids` (идентификаторы новых аннотаций).

## Writer и WriteOptions

### `class WriteOptions`

Неизменяемый dataclass; все поля необязательны.

| Поле | По умолчанию | Значение |
|---|---|---|
| `codec` | `Codec.ZSTD` | `Codec` или `"zstd"`, `"lz4"`, `"none"` |
| `encoding` | `EncodingPolicy.ADAPTIVE` | `"adaptive"`, `"classic"`, `"plain"` |
| `rows_per_block` | `65536` | 1 … 2²⁴ |
| `max_block_bytes` | `64 МиБ` | порог записи буфера устройства |
| `sync` | `True` | `fsync` при коммите |
| `overwrite` | `False` | `Writer.create` заменяет существующий файл |
| `repair_corruption` | `False` | `Writer.open` обрезает файл после повреждения в середине |
| `zstd_level` | `3` | уровень Zstandard |

### `class Writer`

Записывает и правит файл; в каждый момент у файла один писатель
(исключительная блокировка). Используйте как контекстный менеджер: при
обычном выходе изменения коммитятся и файл завершается, при исключении
незакоммиченные изменения отбрасываются. Всё время — сырые целые числа.

| Метод / атрибут | Описание |
|---|---|
| `Writer.create(path, options=None, **kwargs)` | Создаёт файл (ошибка, если он есть, кроме `overwrite=True`). |
| `Writer.open(path, options=None, **kwargs)` | Открывает для дозаписи; восстанавливает незавершённый файл (см. `recovery`). |
| `recovery: RecoveryReport \| None` | Что было удалено при открытии. |
| `devices`, `device(name)`, `has_device(name)`, `head`, `metadata`, `closed`, `has_pending_changes` | Состояние с учётом незакоммиченных изменений. |
| `create_device(schema)` | Создаёт устройство (`DeviceExistsError`, если оно есть). |
| `ensure_device(schema) -> DeviceSchema` | Создаёт его или проверяет, что существующее устроено так же. |
| `write(device, data) -> int` | Записывает `Batch` или словарь (ключ времени — имя столбца времени устройства или `"time"`); отсутствующие столбцы пусты; строки в любом порядке, повторы заменяют. |
| `write_row(device, time, values=None, **kw)` | Записывает одну строку. |
| `get(device, time) -> dict \| None` | Текущая строка с учётом незакоммиченных изменений. |
| `update(device, time, values=None, **kw)` | Меняет значения существующей точки. |
| `delete(device, time)`, `delete_range(device, start=None, end=None)` | Удаление. |
| `annotate(device, start, end, label, note=None) -> int` | Возвращает новый идентификатор. |
| `update_annotation(id, start=None, end=None, label=None, note=None)`, `remove_annotation(id)`, `annotations()` | Правка аннотаций. |
| `set_metadata(key, value)`, `update_metadata(mapping)`, `remove_metadata(key)` | Метаданные файла. |
| `commit(message=None, author=None, *, time_ns=None) -> CommitInfo \| None` | Запечатывает изменения (`None`, если их нет). |
| `rollback()` | Отменяет всё после последнего коммита. |
| `close()` | Коммитит, пишет индекс и футер, закрывает файл. |
| `set_chain_origin(hash32)`, `restore_annotation(...)` | Используются уплотнением. |

## Reader и Query

### `class Reader(path, *, strict=False, verify_checksums=True)`

Снимок файла только для чтения (последующие коммиты не видны). `strict`
отказывается открывать незавершённый файл; `verify_checksums=False` не
проверяет CRC читаемых сегментов. Контекстный менеджер; `close()`
освобождает файл.

| Член | Описание |
|---|---|
| `finalized`, `format_version`, `size`, `path`, `metadata`, `devices`, `device_names` | Каталог. |
| `device(name)`, `metadata_as_of(commit)`, `device_exists_at(name, commit)` | Поиск. |
| `commits()`, `commit(number)`, `head`, `commit_at(time_ns)` | История. |
| `annotations(device=None, *, as_of=None)`, `diff(device, start, end=None)` | Аннотации и различия. |
| `blocks(device) -> list[BlockInfo]` | Сведения о хранении. |
| `query(device) -> Query` | Начинает запрос. |
| `read(device, *, columns=None, start=None, end=None, as_of=None) -> Batch` | Сокращение с сырым временем. |

### `class Query`

Неизменяем; каждый метод возвращает новый запрос.

| Метод | Описание |
|---|---|
| `columns(names)` | Выбирает столбцы значений (время включено всегда). |
| `time_range(start=None, end=None)` | Сырые границы включительно. |
| `as_of(commit)` | Версия. |
| `collect() -> Batch` | Весь результат. |
| `batches() -> Iterator[Batch]` | Потоково, в порядке времени. |
| `count() -> int` | Число точек. |
| `device` | Схема. |

## Batch и Column

### `class Batch(time, columns=None, schema=None)`

Строки одного устройства: `time` (массив `numpy.int64` сырых меток),
`columns` (`dict[str, Column]`) и `schema` (`DeviceSchema` или `None`).

| Член | Описание |
|---|---|
| `len(batch)`, `batch[name]`, `names` | Размер и столбцы. |
| `row(i) -> dict`, `rows() -> Iterator[(time, dict)]`, `to_dict()` | Значения Python. |
| `slice(start, stop)`, `take(indices)`, `select(names)` | Подмножества. |
| `Batch.concat(batches)`, `Batch.empty(types, schema=None)` | Построение. |
| `datetimes()` | Метки как `numpy.datetime64` (UTC). |
| `to_pandas(*, dtype_backend="numpy")`, `to_polars()`, `to_arrow()` | Преобразования. |

### `class Column(data_type, values, validity=None)`

`values` — массив NumPy с dtype типа (`object` для строк); `validity` —
булева маска (`True` — значение есть) или `None` (пустых нет).

| Член | Описание |
|---|---|
| `Column.from_values(data_type, values, *, nan_is_null=False)` | Из значений Python; `None` — пусто; только преобразования без потерь. |
| `Column.nulls(data_type, length)`, `Column.concat(columns)` | Построение. |
| `data_type`, `null_count`, `len(column)` | Свойства. |
| `value(i)` / `column[i]`, `to_list()`, `to_numpy()`, `dense()`, `valid_mask()`, `is_valid(i)` | Доступ. |
| `slice(start, stop)`, `take(indices)` | Подмножества. |

## Схемы и типы

* `DataType` — `BOOL`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`, `STRING`;
  `label`, `numpy_dtype`, `plain_width`, `DataType.parse("float64")`.
* `TimeUnit` — `SECOND`, `MILLISECOND`, `MICROSECOND`, `NANOSECOND`; `label`
  (`"s"` …), `per_second`, `nanos`, `from_nanos()`, `to_nanos()`,
  `TimeUnit.parse("ms")`.
* `ColumnSchema(name, data_type, metadata={})`.
* `DeviceSchema(name, time_unit="ms", columns=(), timezone=None,
  time_name="time", metadata={})` — `DeviceSchema.build(name, unit,
  {столбец: тип}, **kwargs)`, `with_column(name, type, metadata=None)`,
  `column_names`, `types`, `column_index(name)`, `validate()`,
  `same_layout(other)`.
* `Codec` — `NONE`, `LZ4`, `ZSTD`; `Codec.parse("zstd")`.
* `Encoding` — `PLAIN`, `DELTA_BIT_PACK`, `DELTA_OF_DELTA`, `RLE`, `XOR`,
  `DICTIONARY`; `Encoding.candidates(data_type)`, `Encoding.classic(data_type)`.
* `EncodingPolicy` — `ADAPTIVE`, `CLASSIC`, `PLAIN`.

## Объекты истории

* `CommitInfo` — `number`, `time_ns`, `time` (`datetime` с поясом UTC),
  `author`, `message`, `hash` и `prev_hash` (64 шестнадцатеричные цифры),
  `short_hash`, `offset`, `changes`.
* `CommitChanges` — `devices_created`, `blocks_written`, `rows_written`,
  `ranges_deleted`, `annotation_ops`, `metadata_changed`.
* `Annotation` — `id`, `device`, `start`, `end` (сырые), `label`, `note`.
* `Diff` — `device`, `columns`, `from_commit`, `to_commit`, `points`
  (`PointChange`: `time`, `kind`, `before`, `after`), `annotations`
  (`AnnotationChange`: `id`, `kind`, `before`, `after`); `bool(diff)` истинно,
  если есть изменения.
* `ChangeKind` — `ADDED`, `REMOVED`, `CHANGED`.

## Сведения о хранении

* `BlockInfo` — `offset`, `commit`, `rows`, `t_min`, `t_max`, `segments`.
* `SegmentInfo` — `column`, `data_type` (`None` для времени), `encoding`,
  `codec`, `stored_bytes`, `encoded_bytes`, `null_count`, `statistics`.
* `Statistics` — `min`, `max`, `is_float`.

## Отчёты обслуживания

* `VerifyReport` — `ok`, `frames`, `commits`, `blocks`, `finalized`, `head`,
  `origin`, `problems`, `warnings`; `bool(report)` равно `report.ok`.
* `RecoverReport` — `was_finalized`, `recovery`, `removed_bytes`,
  `removed_frames`.
* `RecoveryReport` — `truncated_bytes`, `discarded_frames`, `reason`.
* `CompactReport` — `version`, `origin`, `rows`, `annotations`,
  `bytes_before`, `bytes_after`.

## Время

`to_raw(value, unit, tz=None, rounding="floor")` принимает:

| Значение | Толкование |
|---|---|
| `int` | уже сырое (возвращается без изменений) |
| `str` (ISO 8601) | со смещением или `Z` — момент времени; без него — местное время пояса `tz` |
| `datetime` | с поясом — момент времени; без пояса — местное время пояса `tz` |
| `date` | полночь, местное время пояса `tz` |
| `pandas.Timestamp`, `numpy.datetime64` | как `datetime`, с наносекундами |

`rounding` определяет, что делать со временем точнее единицы: `"floor"`,
`"ceil"` или `"exact"` (ошибка). Функции чтения округляют `start` вверх, а
`end` вниз; точки, записываемые через `Transaction`, должны быть точными.

## Адаптеры

`pytrosna.adapters` содержит преобразования, на которых построен
высокоуровневый API: `batch_to_arrow`, `batch_to_pandas`, `batch_to_polars`
(чтение) и `normalize(data, time_column=None) -> Iterator[SourceTable]`,
`infer_device(name, source, unit=None) -> DeviceSchema`,
`to_batch(source, schema, unit=None) -> Batch` (запись). С их помощью можно
строить собственные конвейеры, например с низкоуровневым `Writer`:

```text
for source in pytrosna.adapters.normalize(polars_frame):
    schema = writer.ensure_device(pytrosna.adapters.infer_device("vm01", source))
    writer.write("vm01", pytrosna.adapters.to_batch(source, schema))
```

## Ошибки

Базовый класс — `TrosnaError`.

| Исключение | Также | Когда возникает |
|---|---|---|
| `CorruptedError` | | некорректные или повреждённые данные; `reason`, `offset` |
| `NotTrosnaError` | `CorruptedError` | файл не является файлом Trosna |
| `UnsupportedVersionError` | | другая старшая версия формата |
| `UnsupportedError` | | неизвестная возможность или данные, которые Trosna не может хранить |
| `NotFinalizedError` | | `Reader(strict=True)` для незавершённого файла |
| `LockedError` | | файл занят другим писателем |
| `UnknownDeviceError`, `UnknownColumnError`, `UnknownAnnotationError`, `UnknownCommitError`, `PointNotFoundError` | `KeyError` | объект не найден |
| `DeviceExistsError` | | устройство уже существует |
| `TypeMismatchError` | `TypeError` | значение неподходящего типа |
| `SchemaError`, `InvalidArgumentError`, `LimitExceededError` | `ValueError` | неверная схема, аргумент или размер |

## Константы

* `EXTENSION = "trosna"` — расширение имени файла.
* `FORMAT_VERSION = (1, 0)` — записываемая версия формата.
* `__version__` — версия пакета.
