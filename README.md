# hsi-toolkit

Настольное приложение (PySide6 + matplotlib) для визуализации и анализа
гиперспектральных кубов (ENVI HDR/RAW, BSQ/BIL/BIP) с векторной разметкой
полигонов и полилиний.

![Скриншот приложения](docs/screenshot.png)

## Возможности

- **Загрузка набора данных**: куб + белый/тёмный референсы (опционально) +
  JSON-разметка. Автопоиск референсов `WHITEREF_*` / `DARKREF_*` рядом с кубом
  или в `capture/`. Куб остаётся на диске (memmap); референсы сводятся к
  профилям «среднее по строкам» `dark_prof[b, x]` / `white_prof[b, x]`
  (компенсация неравномерности засветки вдоль скана, как
  `bw_correction_broadcast_axis` из исследовательских ноутбуков).
- **Предпросмотр**: псевдо-RGB (ближайшие каналы к 650/550/450 нм) или
  спектральное окно (центр/ширина слайдерами), растяжение перцентили/min-max,
  цветовые карты. При открытии строится уменьшенный corrected-куб (×4) —
  перерисовка мгновенная; координаты разметки — всегда полноразмерные пиксели.
- **Разметка**: полигоны (`PolygonSelector`) и полилинии (свой инструмент:
  клик — вершина, перетаскивание — правка, Backspace — удалить последнюю,
  Enter/двойной клик — завершить, Esc — отмена). Имя, комментарий, цвет
  (tab10), видимость; мультивыбор в списке.
- **Спектры**: полигоны → медиана + доверительный интервал (IQR 25–75 по
  умолчанию, 2.5–97.5 или mean±3σ) одним графиком, свой цвет на объект;
  полилинии → 3D-поверхность (позиция × λ) или водопад (карта/трассы),
  подграфик на линию. Спектры считаются по полному разрешению.
- **Сохранение/загрузка разметки**: JSON `hsi-toolkit.annotations/1`, по
  умолчанию `<куб>.annotations.json`.
- **Экспорт NPZ**: участки изображения + маски — полигон → bbox-вырезка куба
  `(B, H, W)` + булева маска `(H, W)`; полилиния → лента `(B, S, width)` +
  относительные позиции `(S,)`; плюс `wavelengths` и JSON `objects`
  (id/kind/name/comment/color/points/bbox). Выбор объектов и corrected/raw в
  диалоге.

## Запуск

```powershell
uv run hsi-toolkit
```

Формат данных — ENVI HDR с длинами волн в заголовке (нм); каналы сортируются
по возрастанию длины волны при загрузке. Примеры раскладок данных:

```
hypercube_2026-10-23_18-02-43/          # плоская
    hypercube_*.hdr|.raw
    WHITEREF_*.hdr|.raw
    DARKREF_*.hdr|.raw
съёмка/capture/                         # в подпапке capture
    <куб>.hdr|.raw, WHITEREF_*.hdr, DARKREF_*.hdr
```

## Разработка

```powershell
uv sync                 # зависимости (numpy, matplotlib, PySide6)
uv run pytest           # тесты ядра + offscreen smoke-тест GUI
```

Структура:

```
src/hsi_toolkit/
  io/envi_hdr.py        # ENVI-загрузчик (memmap, сортировка по λ)
  core/
    dataset.py          # HSIWorkspace: ленивая коррекция, LRU каналов, lowres-куб
    preview.py          # pseudo_rgb(), window_preview()
    annotations.py      # Annotation/AnnotationSet, JSON, растеризация
    spectra.py          # статистики полигонов, профиль полилинии (порт ноутбука)
    exporter.py         # export_npz()
  app/
    main_window.py      # окна/тулбар/доки, фоновые QThread-загрузки
    data_dialog.py      # выбор куба/референсов/разметки
    preview_panel.py    # режимы, слайдеры λ, растяжение
    annotation_panel.py # список объектов, имена, комментарии
    spectrum_panel.py   # графики спектров (3D/водопад/CI)
    export_dialog.py    # экспорт NPZ
    canvas/             # канвас предпросмотра, PolylineTool
tests/                  # pytest: ядро + offscreen GUI smoke
```

Примечание: коррекция `corrected[b,y,x] = clip((raw - dark_prof[b,x]) /
(white_prof[b,x] - dark_prof[b,x] + eps), 0, 1)` соответствует
`bw_correction_broadcast_axis(sample, dark, white, axis=1)`; без референсов
приложение работает с сырыми данными (с предупреждением). Импорт старых
grayscale PNG-масок не поддерживается — разметка создаётся в приложении.
