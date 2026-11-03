"""Collect CPU and memory usage from the local computer into a CSV."""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=120,
                        help="how long to sample (default: 120 seconds)")
    parser.add_argument("--output", default="data/system_activity.csv",
                        help="CSV output path")
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error("--seconds must be at least 2")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Collecting CPU and memory readings for {args.seconds} seconds...")
    for i in range(args.seconds):
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        rows.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "cpu_percent": cpu, "memory_percent": memory})
        if (i + 1) % 10 == 0 or i + 1 == args.seconds:
            print(f"Recorded {i + 1}/{args.seconds} readings")

    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} readings to {output.resolve()}")
    print("These readings are unlabeled observations, not proof of safe or malicious activity.")


if __name__ == "__main__":
    main()
