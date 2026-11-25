"""Persisted application settings (QSettings wrapper).

Keeps the folder of the last successfully selected dataset image so file
dialogs start where the user last picked an image. Only actual selections
update the stored value; cancelled dialogs never touch it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings

_ORG = "hsi-toolkit"
_APP = "hsi-toolkit"
_LAST_DIR_KEY = "data/last_dir"


def last_data_dir() -> Optional[Path]:
    """Folder of the last selected dataset image, or None if never saved."""
    settings = QSettings(_ORG, _APP)
    value = settings.value(_LAST_DIR_KEY, "")
    value = str(value) if value is not None else ""
    return Path(value) if value else None


def set_last_data_dir(path: Path) -> None:
    """Remember the folder of the selected dataset image."""
    settings = QSettings(_ORG, _APP)
    settings.setValue(_LAST_DIR_KEY, str(path))
