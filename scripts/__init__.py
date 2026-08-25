"""Admin CLI scripts for operating a Ganymede coordinator.

Each module here is both a `python3 -m scripts.<name>` entrypoint and an
importable set of functions tests call directly (docs/03-roadmap.md M1). None
of these talk to a running coordinator process -- they open the same SQLite
file and object store the FastAPI app does, which is safe because SQLite's WAL
mode lets a short-lived script and the long-running server share the database
without stepping on each other (ganymede/coordinator/db.py).
"""

from __future__ import annotations
