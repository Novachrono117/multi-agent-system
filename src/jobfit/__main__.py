"""Allows ``python -m jobfit``, which is the form the README uses."""

from __future__ import annotations

from jobfit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
