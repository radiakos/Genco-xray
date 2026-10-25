"""Shared command-line boilerplate for scripts/*.py."""

from __future__ import annotations

import argparse
import logging

from .config import SYNTHETIC


def parser(doc: str | None, *, grid: bool = False) -> argparse.ArgumentParser:
    """ArgumentParser showing the script docstring verbatim; ``grid=True`` adds the analysis scripts' ``--grid`` and
    its ``--synthetic`` alias (resolve with :func:`grid_name`)."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if grid:
        ap.add_argument("--grid", required=True)
        ap.add_argument("--synthetic", action="store_true", help="alias for --grid synthetic")
    return ap


def grid_name(args: argparse.Namespace) -> str:
    return SYNTHETIC if args.synthetic else args.grid


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
