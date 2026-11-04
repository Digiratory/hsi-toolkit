"""PySide6 application: entry point and Qt backend selection."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")


def run() -> int:
    from PySide6.QtWidgets import QApplication

    from hsi_toolkit.app.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("HSI Toolkit")
    app.setOrganizationName("hsi-toolkit")
    window = MainWindow()
    window.show()
    return app.exec()
