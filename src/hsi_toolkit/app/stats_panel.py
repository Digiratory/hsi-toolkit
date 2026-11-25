"""Statistics panel: pairwise polygon (ROI) comparisons per band.

Follows the ``SpectrumPanel`` worker conventions (generation guard, cancel
token, ``_running_workers`` + ``workers_idle``) so the main window's deferred
workspace-close logic also sees this panel's memmap readers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from hsi_toolkit.app.workers import FunctionWorker, shutdown_workers
from hsi_toolkit.core.annotations import Annotation, rasterize
from hsi_toolkit.core.spectra import mask_band_values
from hsi_toolkit.core.stats import (
    CORRECTION_BH,
    CORRECTION_HOLM,
    CORRECTION_NONE,
    TEST_MANN_WHITNEY,
    TEST_WELCH,
    RoiStatsResult,
    bh_adjust,
    export_csv,
    export_xlsx,
    holm_adjust,
    pairwise_roi_stats,
    table_rows,
)

TEST_LABELS = ("Mann–Whitney U", "t-тест Уэлча")
CORRECTION_LABELS = ("Холма", "FDR Бенджамини–Хохберга", "Без поправки")
ALPHA_LABELS = ("0.05", "0.01", "0.001")
ALPHA_VALUES = (0.05, 0.01, 0.001)
METRIC_LABELS = ("p-корр", "ΔR", "ΔR, %", "d (Коэна)", "AUC")

TEST_BY_INDEX = {0: TEST_MANN_WHITNEY, 1: TEST_WELCH}
CORRECTION_BY_INDEX = {0: CORRECTION_HOLM, 1: CORRECTION_BH, 2: CORRECTION_NONE}
METRICS = ("p_adj", "delta", "delta_pct", "d", "auc")

SIGNIFICANT_BG = QColor("#ffcccc")
TEST_NAMES = {TEST_MANN_WHITNEY: "Mann–Whitney U", TEST_WELCH: "t-тест Уэлча"}
CORR_NAMES = {
    CORRECTION_HOLM: "Холм",
    CORRECTION_BH: "BH (FDR)",
    CORRECTION_NONE: "без поправки",
}

NO_DATA_HINT = "Нет набора данных"
EMPTY_HINT = "Нужно ≥2 видимых полигона"


def _fmt(value: float, spec: str = ".4g") -> str:
    """NaN-safe metric text for table cells and tooltips."""
    value = float(value)
    return f"{value:{spec}}" if np.isfinite(value) else "—"


_STATS_HELP_HTML = """
<h3>Сравнение зон ROI по каналам</h3>
<p>Таблица: строки — каналы (длины волн), столбцы — пары зон («A vs B» в порядке списка
видимых полигонов). Значения берутся из тех же данных, что и в панели «Спектры»
(corrected при наличии референсов, иначе raw).</p>
<h4>Эталон</h4>
<p>Это зона, относительно которой считается <b>ΔR,% = ΔR / среднее(эталон) × 100</b>.
Это единственная метрика, зависящая от эталона: ΔR, p-значения, d и AUC считаются для пары
напрямую и при смене эталона не меняются. Чтобы увидеть эффект смены эталона,
переключите «Ячека» в «ΔR, %».</p>
<h4>Метрики</h4>
<ul>
<li><b>ΔR</b> = среднее(A) − среднее(B).</li>
<li><b>ΔR,%</b> — ΔR относительно среднего эталонной зоны, в процентах
(среднее эталона = 0 → не определено).</li>
<li><b>Медианы, ΔR медиан</b> — робастный вариант ΔR по медианам зон (в экспорте).</li>
<li><b>Cohen's d</b> — ΔR, нормированная на объединённое СКО (ddof=1); ориентировочно:
0.2 — малый, 0.5 — средний, 0.8 — большой эффект. Нулевая дисперсия → не определено.</li>
<li><b>Mann–Whitney U</b> (по умолчанию) — непараметрический ранговый тест, двусторонний p;
не требует нормальности.</li>
<li><b>t-тест Уэлча</b> — параметрический тест для нормально распределённых данных
с разными дисперсиями.</li>
<li><b>AUC</b> = U₁/(n₁·n₂) = P(A &gt; B) — вероятность, что случайный пиксель зоны A
больше случайного пикселя зоны B. 0.5 — распределения неразличимы; 1 или 0 — полное
разделение (направление зависит от порядка зон в паре). Считается всегда, независимо
от выбранного теста.</li>
</ul>
<h4>Поправка на множественные сравнения</h4>
<p>Проверяется сразу вся семья гипотез (каналы × пары), поэтому «сырые» p-значения
завышают значимость. Поправки: <b>Холм</b> (по умолчанию, консервативная),
<b>FDR Бенджамини–Хохберга</b> (контролирует ожидаемую долю ложных отклонений, мягче),
<b>без поправки</b>. p = NaN (мало данных, вырожденные выборки) проходит через поправку
без изменений.</p>
<h4>α и подсветка</h4>
<p>Ячейка подсвечивается, если скорректированное p &lt; α. Выбор «Ячейка» влияет только
на отображение; тултип ячейки показывает все метрики пары в этом канале.</p>
<p><i>Замечание: пиксели внутри полигона считаются независимыми выборками;
пространственная автокорреляция завышает значимость.</i></p>
"""


class StatsPanel(QWidget):
    """«Статистика»: таблица «каналы × пары зон» с фоновой калькуляцией."""

    workers_idle = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ws = None
        self._rois: List[Annotation] = []
        self._generation = 0
        self._worker: Optional[FunctionWorker] = None
        self._running_workers: list = []
        self._cancel_current: Optional[dict] = None
        self._result: Optional[RoiStatsResult] = None
        self._result_key: tuple = ()
        # Single-entry cache of extracted per-ROI pixel data, keyed by
        # (ws id, ROI ids+geometry): re-extraction is the expensive memmap part.
        self._extraction_cache: dict = {}
        self._export_busy = False

        self.ref_combo = QComboBox()
        self.test_combo = QComboBox()
        self.test_combo.addItems(list(TEST_LABELS))
        self.correction_combo = QComboBox()
        self.correction_combo.addItems(list(CORRECTION_LABELS))
        self.alpha_combo = QComboBox()
        self.alpha_combo.addItems(list(ALPHA_LABELS))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(METRIC_LABELS))

        self.csv_button = QPushButton("Экспорт CSV…")
        self.xlsx_button = QPushButton("Экспорт XLSX…")
        self.csv_button.setEnabled(False)
        self.xlsx_button.setEnabled(False)
        self.csv_button.setToolTip("Экспорт таблицы «каналы × пары» в CSV (utf-8-sig)")
        self.xlsx_button.setToolTip("Экспорт таблицы «каналы × пары» в Excel (XLSX)")
        self.help_button = QPushButton()
        self.help_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        )
        self.help_button.setToolTip("Описание статистик")

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Эталон:"))
        bar.addWidget(self.ref_combo)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Тест:"))
        bar.addWidget(self.test_combo)
        bar.addWidget(QLabel("Поправка:"))
        bar.addWidget(self.correction_combo)
        bar.addWidget(QLabel("α:"))
        bar.addWidget(self.alpha_combo)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Ячейка:"))
        bar.addWidget(self.metric_combo)
        bar.addStretch(1)
        bar.addWidget(self.csv_button)
        bar.addWidget(self.xlsx_button)
        bar.addWidget(self.help_button)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setVisible(False)

        self.status_label = QLabel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addLayout(bar)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.status_label)

        # Reference/test/correction/alpha change the cached values -> recompute;
        # the cell metric is a pure redraw.
        self.ref_combo.currentIndexChanged.connect(lambda _: self._recompute())
        self.test_combo.currentIndexChanged.connect(lambda _: self._recompute())
        self.correction_combo.currentIndexChanged.connect(lambda _: self._recompute())
        self.alpha_combo.currentIndexChanged.connect(lambda _: self._recompute())
        self.metric_combo.currentIndexChanged.connect(lambda _: self._redraw())
        self.csv_button.clicked.connect(self._export_csv)
        self.xlsx_button.clicked.connect(self._export_xlsx)
        self.help_button.clicked.connect(self._show_help)

        self.show_message(NO_DATA_HINT)

    # -- public ---------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return bool(self._running_workers)

    def set_workspace(self, ws) -> None:
        self._generation += 1
        self._ws = ws
        self._rois = []
        self._result = None
        self._result_key = ()
        self._extraction_cache = {}
        self._sync_ref_combo()
        self.show_message(NO_DATA_HINT if ws is None else EMPTY_HINT)
        self._redraw()

    def update_rois(self, annotations: List[Annotation]) -> None:
        """Recompute stats for the given (visible) polygon annotations."""
        self._rois = [a for a in annotations if a.kind == "polygon"]
        self._sync_ref_combo()
        self._recompute()

    def shutdown(self, wait: bool = True) -> None:
        """Request cancellation; optionally block until workers finish."""
        if self._cancel_current is not None:
            self._cancel_current["cancelled"] = True
        if not wait:
            return
        shutdown_workers(self._running_workers)
        self._export_busy = False

    # -- settings helpers -------------------------------------------------------

    def _settings(self) -> Tuple[str, str, float]:
        test = TEST_BY_INDEX.get(self.test_combo.currentIndex(), TEST_MANN_WHITNEY)
        correction = CORRECTION_BY_INDEX.get(self.correction_combo.currentIndex(), CORRECTION_HOLM)
        alpha_index = self.alpha_combo.currentIndex()
        alpha = ALPHA_VALUES[alpha_index] if 0 <= alpha_index < len(ALPHA_VALUES) else 0.05
        return test, correction, alpha

    def _cache_key(self) -> tuple:
        test, correction, alpha = self._settings()
        return (
            id(self._ws),
            tuple((a.id, tuple(map(tuple, a.points))) for a in self._rois),
            self.ref_combo.currentData(),
            test,
            correction,
            alpha,
        )

    def _sync_ref_combo(self) -> None:
        current = self.ref_combo.currentData()
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        for ann in self._rois:
            self.ref_combo.addItem(ann.name, ann.id)
        index = self.ref_combo.findData(current) if current is not None else -1
        if index >= 0:
            self.ref_combo.setCurrentIndex(index)
        self.ref_combo.blockSignals(False)

    # -- computation ------------------------------------------------------------

    def _recompute(self) -> None:
        self._generation += 1
        generation = self._generation
        if self._ws is None:
            self._invalidate(NO_DATA_HINT)
            return
        if len(self._rois) < 2:
            self._invalidate(EMPTY_HINT)
            return
        key = self._cache_key()
        if key == self._result_key and self._result is not None:
            self._redraw()  # e.g. a rename only: fresh labels, same numbers
            return
        if self._result is not None and key[:4] == self._result_key[:4]:
            # Only correction/alpha changed: re-adjust the stored p-values,
            # no re-extraction and no re-testing needed.
            self._adjust_settings(key)
            return

        if self._cancel_current is not None:
            self._cancel_current["cancelled"] = True
        cancel = {"cancelled": False}
        self._cancel_current = cancel
        self.show_message("Расчёт статистики…")
        ws = self._ws
        rois = list(self._rois)
        test, correction, alpha = self._settings()
        requested_ref_id = self.ref_combo.currentData()
        ref_position = next((i for i, a in enumerate(rois) if a.id == requested_ref_id), 0)
        extraction_key = (key[0], key[1])
        bundle = self._extraction_cache.get(extraction_key)

        def compute(progress):
            if bundle is None:
                samples = []
                used = []
                skipped = []
                for i, ann in enumerate(rois):
                    if cancel["cancelled"]:
                        return None
                    progress.emit(int(60 * i / max(len(rois), 1)), f"Извлечение: {ann.name}")
                    try:
                        mask = rasterize(ann, (ws.lines, ws.samples))
                        samples.append(mask_band_values(ws, mask))
                        used.append(i)
                    except ValueError as exc:
                        skipped.append(f"{ann.name} ({exc})")
                if cancel["cancelled"]:
                    return None
                if len(samples) < 2:
                    detail = "; ".join(skipped) if skipped else "нет корректных ROI"
                    raise ValueError(f"Сравнение невозможно: меньше 2 ROI ({detail})")
                names = [rois[i].name for i in used]
                ids = [rois[i].id for i in used]
                new_bundle = (samples, names, ids, used, skipped)
            else:
                samples, names, ids, used, skipped = bundle
                new_bundle = None
            ref_index = used.index(ref_position) if ref_position in used else 0
            progress.emit(70, "Сравнение ROI…")
            result = pairwise_roi_stats(
                samples,
                names,
                ref_index,
                test=test,
                alpha=alpha,
                correction=correction,
                wavelengths=ws.wavelengths,
                ids=ids,
                cancel=cancel,
            )
            progress.emit(100, "Готово")
            return (
                key,
                result,
                skipped,
                ids,
                requested_ref_id,
                None if new_bundle is None else (extraction_key, new_bundle),
            )

        self._worker = FunctionWorker(compute)
        self._running_workers.append(self._worker)
        self._worker.finished.connect(lambda worker=self._worker: self._forget_worker(worker))
        self._worker.finished_ok.connect(
            lambda payload, gen=generation: self._on_computed(gen, payload)
        )
        self._worker.failed.connect(lambda message, gen=generation: self._on_failed(gen, message))
        self._worker.start()

    def _forget_worker(self, worker: FunctionWorker) -> None:
        if worker in self._running_workers:
            self._running_workers.remove(worker)
        if not self._running_workers:
            self.workers_idle.emit()

    def _invalidate(self, message: str) -> None:
        self._result = None
        self._result_key = ()
        self.show_message(message)
        self._redraw()

    def _on_computed(self, generation: int, payload) -> None:
        if generation != self._generation or payload is None:
            return
        key, result, skipped, used_ids, requested_ref_id, cache_entry = payload
        if result is None:  # cancelled inside the stats sweep
            return
        if cache_entry is not None:
            extraction_key, bundle = cache_entry
            self._extraction_cache = {extraction_key: bundle}
        # When the requested reference was skipped as empty, the result
        # reflects the fallback reference: resync the combo and the stored
        # key so later renames still hit the redraw path.
        note = ""
        if requested_ref_id is not None and used_ids and requested_ref_id not in used_ids:
            effective_id = used_ids[result.ref_index]
            index = self.ref_combo.findData(effective_id)
            if index >= 0:
                self.ref_combo.blockSignals(True)
                self.ref_combo.setCurrentIndex(index)
                self.ref_combo.blockSignals(False)
            key = key[:2] + (effective_id,) + key[3:]
            skipped_ref = self._find_roi(requested_ref_id)
            skipped_ref_name = skipped_ref.name if skipped_ref else requested_ref_id
            note = (
                f"; эталон «{skipped_ref_name}» пропущен — "
                f"используется «{result.names[result.ref_index]}»"
            )
        self._result = result
        self._result_key = key
        if skipped:
            self.show_message("Пропущены ROI с пустыми масками: " + "; ".join(skipped) + note)
        elif note:
            self.show_message(note.lstrip("; "))
        else:
            self.show_message("Готово. " + self._ref_note(result))
        self._redraw()

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._result = None
        self._result_key = ()
        self.show_message(f"Ошибка расчёта статистики: {message}")
        self._redraw()

    def _adjust_settings(self, key: tuple) -> None:
        """Re-apply correction/alpha to the cached p-values without recompute."""
        result = self._result
        correction, alpha = key[4], key[5]
        if correction == CORRECTION_NONE:
            p_adjusted = result.p_raw.copy()
        elif correction == CORRECTION_HOLM:
            p_adjusted = holm_adjust(result.p_raw)
        else:
            p_adjusted = bh_adjust(result.p_raw)
        self._result = replace(
            result,
            alpha=alpha,
            correction=correction,
            p_adjusted=p_adjusted,
            significant=p_adjusted < alpha,
        )
        self._result_key = key
        self.show_message("Готово. " + self._ref_note(self._result))
        self._redraw()

    # -- drawing ----------------------------------------------------------------

    def show_message(self, text: str) -> None:
        self.status_label.setText(text)

    def _find_roi(self, roi_id: str) -> Optional[Annotation]:
        return next((a for a in self._rois if a.id == roi_id), None)

    def _ref_note(self, result: RoiStatsResult) -> str:
        """Current reference name for the status line."""
        name = result.names[result.ref_index]
        if result.ref_index < len(result.used_ids):
            ann = self._find_roi(result.used_ids[result.ref_index])
            if ann is not None:
                name = ann.name
        return f"Эталон: «{name}»"

    def _pair_label(self, result: RoiStatsResult, pair_index: int) -> str:
        # Renames recompute nothing: resolve labels by id so they stay fresh
        # even when some ROIs were skipped as empty-mask.
        i, j = result.pair_indices[pair_index]
        ids = result.used_ids
        if max(i, j) < len(ids):
            first = self._find_roi(ids[i])
            second = self._find_roi(ids[j])
            if first is not None and second is not None:
                return f"{first.name} vs {second.name}"
        return result.pairs[pair_index]

    def _redraw(self) -> None:
        result = self._result
        if result is None or not result.pairs:
            self.table.setVisible(False)
            self._set_export_enabled(False)
            return
        metric_index = self.metric_combo.currentIndex()
        metric = METRICS[metric_index] if 0 <= metric_index < len(METRICS) else "p_adj"
        values = {
            "p_adj": result.p_adjusted,
            "delta": result.delta,
            "delta_pct": result.delta_pct,
            "d": result.cohens_d,
            "auc": result.auc,
        }[metric]

        n_bands = int(result.wavelengths.size)
        n_pairs = len(result.pairs)
        self.table.setVisible(True)
        self.table.setUpdatesEnabled(False)
        self.table.clear()
        self.table.setRowCount(n_bands)
        self.table.setColumnCount(n_pairs)
        self.table.setHorizontalHeaderLabels([self._pair_label(result, p) for p in range(n_pairs)])
        self.table.setVerticalHeaderLabels([f"{wl:.1f} нм" for wl in result.wavelengths])
        for p in range(n_pairs):
            for b in range(n_bands):
                item = QTableWidgetItem(_fmt(values[p, b]))
                item.setToolTip(self._tooltip(result, p, b))
                if result.significant[p, b]:
                    item.setBackground(QBrush(SIGNIFICANT_BG))
                self.table.setItem(b, p, item)
        self.table.resizeColumnsToContents()
        self.table.setUpdatesEnabled(True)
        self._set_export_enabled(True)

    def _tooltip(self, result: RoiStatsResult, p: int, b: int) -> str:
        test_name = TEST_NAMES[result.test]
        corr_name = CORR_NAMES[result.correction]
        verdict = (
            "значимо" if result.significant[p, b] else "не значимо"
        ) + f" при α = {result.alpha:g}"
        lines = [
            f"{self._pair_label(result, p)} — {result.wavelengths[b]:.1f} нм",
            f"Средние: {_fmt(result.mean_a[p, b])} / {_fmt(result.mean_b[p, b])}",
            f"ΔR = {_fmt(result.delta[p, b])} ({_fmt(result.delta_pct[p, b])} % от эталона)",
            f"Медианы: {_fmt(result.median_a[p, b])} / {_fmt(result.median_b[p, b])}"
            f", ΔR медиан = {_fmt(result.delta_median[p, b])}",
            f"Cohen's d = {_fmt(result.cohens_d[p, b])}, AUC = {_fmt(result.auc[p, b], '.2f')}",
            f"{test_name}: статистика = {_fmt(result.statistic[p, b])}, "
            f"p = {_fmt(result.p_raw[p, b])}",
            f"p ({corr_name}) = {_fmt(result.p_adjusted[p, b])} — {verdict}",
            "Пиксели считаются независимыми выборками; пространственная "
            "автокорреляция завышает значимость.",
        ]
        return "\n".join(lines)

    # -- export -----------------------------------------------------------------

    def _set_export_enabled(self, enabled: bool) -> None:
        state = enabled and self._result is not None and not self._export_busy
        self.csv_button.setEnabled(state)
        self.xlsx_button.setEnabled(state)

    def _export_csv(self) -> None:
        self._run_export("csv")

    def _export_xlsx(self) -> None:
        self._run_export("xlsx")

    def _run_export(self, kind: str) -> None:
        if self._result is None or self._ws is None or self._export_busy:
            return
        suffix = ".stats.csv" if kind == "csv" else ".stats.xlsx"
        default = str(Path(self._ws.hdr_path).with_suffix(suffix))
        file_filter = (
            "CSV (*.csv);;All files (*)" if kind == "csv" else "Excel (*.xlsx);;All files (*)"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт CSV" if kind == "csv" else "Экспорт XLSX", default, file_filter
        )
        if not path:
            return
        result = self._result
        self._export_busy = True
        self._set_export_enabled(False)
        self.show_message("Экспорт…")

        def compute(progress):
            rows = table_rows(result)
            progress.emit(50, "Запись…")
            if kind == "csv":
                export_csv(rows, path)
            else:
                export_xlsx(rows, path)
            progress.emit(100, "Готово")
            return path

        worker = FunctionWorker(compute)
        self._running_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget_worker(w))
        worker.finished_ok.connect(lambda written: self._on_export_done(written))
        worker.failed.connect(lambda message: self._on_export_failed(message))
        worker.start()

    def _on_export_done(self, path: str) -> None:
        self._export_busy = False
        self._set_export_enabled(True)
        self.show_message(f"Экспортировано: {path}")

    def _on_export_failed(self, message: str) -> None:
        self._export_busy = False
        self._set_export_enabled(True)
        self.show_message(f"Ошибка экспорта: {message}")
        QMessageBox.critical(self, "Ошибка экспорта", message)

    # -- help --------------------------------------------------------------------

    def _build_help_dialog(self) -> QDialog:
        """Build (but do not show) the statistics description dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("О статистиках")
        browser = QTextBrowser()
        browser.setHtml(_STATS_HELP_HTML)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(browser)
        layout.addWidget(buttons)
        dialog.resize(640, 560)
        return dialog

    def _show_help(self) -> None:
        self._build_help_dialog().exec()
