"""PyInstaller entry point for the packaged hsi-toolkit GUI executable."""

import sys

from hsi_toolkit.app import run

if __name__ == "__main__":
    sys.exit(run())
