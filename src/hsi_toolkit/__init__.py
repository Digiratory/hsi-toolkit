try:
    from hsi_toolkit._version import __version__
except ImportError:
    __version__ = "0.0.0"


def main() -> int:
    from hsi_toolkit.app import run

    return run()
