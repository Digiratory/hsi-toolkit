"""Generic QThread worker running a callable off the GUI thread."""

from __future__ import annotations

from typing import Any, Callable, List

from PySide6.QtCore import QThread, Signal


class FunctionWorker(QThread):
    """Runs ``fn(progress_emit)`` in a background thread.

    ``fn`` receives the ``progressed`` signal (call ``progressed.emit(percent,
    text)``) and returns a result delivered via ``finished_ok``. Exceptions are
    reported as text via ``failed``. The owner must keep a reference and wait
    for the thread before it is destroyed.
    """

    finished_ok = Signal(object)
    failed = Signal(str)
    progressed = Signal(int, str)

    def __init__(self, fn: Callable[..., Any], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn(self.progressed)
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_ok.emit(result)


class WorkerPool:
    """Tracks running workers so teardown can join them before closing data.

    ``QThread`` objects must not be garbage-collected while running, and
    memmaps must not be closed underneath a thread still reading them — every
    long-lived worker in the app registers here.
    """

    def __init__(self) -> None:
        self._workers: List[FunctionWorker] = []

    def register(self, worker: FunctionWorker) -> FunctionWorker:
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._discard(w))
        return worker

    def _discard(self, worker: FunctionWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def wait_all(self, timeout_ms: int = 30000) -> None:
        for worker in list(self._workers):
            if worker.isRunning():
                worker.wait(timeout_ms)
