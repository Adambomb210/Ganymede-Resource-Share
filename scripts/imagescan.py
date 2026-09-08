"""Run the image-scan sweep (docs/11 §1.3).

The scan is out-of-band by design: ``/finalize`` marks an image ``pending`` and
returns, and this sweep does the reading. That split is what keeps a 10 GiB
archive off the request cycle, and it is why a submitter can enqueue a job
against a `pending` image (docs/11 §1.4) -- the scan and the queue advance in
parallel, and only the *lease* waits.

Cadence: alongside the ledger sweep, once a minute. Each run scans at most
``--max-images`` archives so one enormous upload cannot starve the others, and
an image whose bytes cannot be read is left ``pending`` for the next run rather
than being failed -- "we could not look" is not "we looked and it was bad".
"""

from __future__ import annotations

import argparse
import os
import sys

from ganymede.coordinator import images
from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect
from ganymede.coordinator.store import Store


def run(db_path: str, settings: Settings, store, max_images: int) -> dict:
    conn = connect(db_path)
    try:
        scanned = images.drain_pending(
            conn, store, images.ScanLimits.from_settings(settings),
            max_images=max_images,
        )
        # Retention (docs/11 §1.2) rides the same sweep: it is the same cadence
        # and the same two objects, and a separate cron for one DELETE is a
        # thing to forget to install.
        collected = images.gc(conn, store, keep_days=settings.image_keep_days)
    finally:
        conn.close()
    return {
        "scanned": scanned,
        "collected": collected,
        "clean": sum(1 for _, status in scanned if status == "clean"),
        "flagged": sum(1 for _, status in scanned if status == "flagged"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ganymede-imagescan",
        description="Scan finalized-but-unscanned images (docs/11). For cron.",
    )
    p.add_argument("--db", default=None,
                   help="path to the coordinator database; default GANYMEDE_DB")
    p.add_argument("--max-images", type=int, default=10,
                   help="most images to scan in one sweep")
    args = p.parse_args(argv)

    settings = Settings.from_env()
    db_path = args.db or os.environ.get("GANYMEDE_DB")
    if not db_path:
        print("no database: pass --db or set GANYMEDE_DB", file=sys.stderr)
        return 2

    report = run(db_path, settings, Store(settings.storage), args.max_images)
    print(f"scanned {len(report['scanned'])} image(s): "
          f"{report['clean']} clean, {report['flagged']} flagged; "
          f"collected {len(report['collected'])}")
    for image_id, status in report["scanned"]:
        print(f"  {image_id} {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
