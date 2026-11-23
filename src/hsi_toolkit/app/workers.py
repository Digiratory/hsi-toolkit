"""Generic QThread worker running a callable off the GUI thread."""

from __future__ import annotations

from typing import Any, Callable, List

from PySide6.QtCore import QObject, QThread, Signal


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


class WorkerPool(QObject):
    """Tracks running workers so teardown can join them before closing data.

    ``QThread`` objects must not be garbage-collected while running, and
    memmaps must not be closed underneath a thread still reading them — every
    long-lived worker in the app registers here. ``worker_finished`` fires on
    every busy→idle transition so owners can defer workspace closes until the
    last reader is done.
    """

    worker_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: List[FunctionWorker] = []

    def register(self, worker: FunctionWorker) -> FunctionWorker:
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._discard(w))
        return worker

    def _discard(self, worker: FunctionWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        self.worker_finished.emit()

    @property
    def is_busy(self) -> bool:
        return any(worker.isRunning() for worker in self._workers)

    def wait_all(self, timeout_ms: int = 30000) -> None:
        for worker in list(self._workers):
            if worker.isRunning():
                worker.wait(timeout_ms)


def shutdown_workers(running_workers: List[FunctionWorker], timeout_ms: int = 30000) -> None:
    """Join a panel's workers, then drop only the ones that finished.

    Timed-out workers stay in ``running_workers`` so the owner's ``is_busy``
    stays truthful (their ``finished`` signal removes them later); callers
    that must not close memmaps under a live reader rely on this.
    """
    for worker in list(running_workers):
        if worker.isRunning():
            worker.wait(timeout_ms)
    running_workers[:] = [w for w in running_workers if w.isRunning()]
