"""
Small CLI wrapper used as the console script entry point.

This calls the Typer application defined in `jerry.__main__` so installing the
package exposes a `jerry` command.
"""
from typing import List
import sys

from . import __version__
from .__main__ import app


def main(argv: List[str] | None = None) -> None:
    """Console entry point for the `jerry` command.

    This function is referenced by the `pyproject.toml` project.scripts entry.
    It simply forwards args to the Typer app defined in `jerry.__main__`.
    """
    # Typer expects to be called as if from `if __name__ == '__main__'`.
    # app() will read from sys.argv by default, so only replace argv when one is provided.
    if argv is not None:
        sys.argv[1:] = argv
    app()


if __name__ == "__main__":
    main()
