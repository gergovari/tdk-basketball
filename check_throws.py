#!/usr/bin/env python3
"""Check which runs didn't produce exactly 5 throws."""

import argparse
import os
import re
from pathlib import Path


def check_data(data_dir, expected=5, show_ok=False):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Error: '{data_dir}' does not exist.")
        return

    # Collect all run videos: <data_dir>/<experiment>/<participant>/runs/*.mp4
    run_videos = []
    for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
        run_videos.extend(data_dir.glob(f"*/*/runs/{ext}"))
    run_videos.sort()

    if not run_videos:
        print(f"No run videos found in {data_dir}/*/*/runs/")
        return

    problems = []
    ok_count = 0

    for run_path in run_videos:
        participant_dir = run_path.parent.parent
        throw_dir = participant_dir / "throw"
        base_id = run_path.stem  # e.g. "1-angle1"

        # Build a relative label for display
        experiment = participant_dir.parent.name
        participant = participant_dir.name
        label = f"{experiment}/{participant}/{run_path.name}"

        if not throw_dir.exists():
            problems.append((label, "NO OUTPUT", "throw/ directory missing"))
            continue

        throw_files = list(throw_dir.glob(f"{base_id}-*.mp4"))

        # Check for status-tagged files (NO, PARTIAL, MORE)
        status_file = None
        for tag in ["NO", "PARTIAL", "MORE"]:
            candidate = throw_dir / f"{base_id}-{tag}.mp4"
            if candidate.exists():
                status_file = tag
                break

        # Count numbered throw files (e.g. 1-angle1-1.mp4 ... 1-angle1-5.mp4)
        numbered = [f for f in throw_files if re.match(rf"^{re.escape(base_id)}-\d+\.mp4$", f.name)]
        count = len(numbered)

        if status_file:
            problems.append((label, status_file, f"Tagged as {status_file}"))
        elif count != expected:
            problems.append((label, f"{count}/{expected}", f"Found {count} throws instead of {expected}"))
        else:
            ok_count += 1
            if show_ok:
                print(f"  ✓  {label} — {count} throws")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Throw Check Report")
    print(f"{'='*60}")
    print(f"  Total runs scanned:  {len(run_videos)}")
    print(f"  OK ({expected} throws):       {ok_count}")
    print(f"  Problems:            {len(problems)}")
    print(f"{'='*60}\n")

    if problems:
        # Group by experiment/participant
        from collections import defaultdict
        grouped = defaultdict(list)
        for label, status, detail in problems:
            parts = label.split("/")
            group = f"{parts[0]}/{parts[1]}"
            grouped[group].append((parts[2], status, detail))

        for group in sorted(grouped.keys()):
            print(f"  {group}/")
            for filename, status, detail in grouped[group]:
                print(f"    ✗  {filename:<30s}  [{status}]")
            print()
    else:
        print("  All runs produced exactly 5 throws! 🎉\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check which runs didn't produce the expected number of throws")
    parser.add_argument("data_folder", nargs="?", default="data/", help="Root data folder (default: data/)")
    parser.add_argument("--expected", type=int, default=5, help="Expected number of throws per run (default: 5)")
    parser.add_argument("--show-ok", action="store_true", help="Also list runs that are OK")
    args = parser.parse_args()

    check_data(args.data_folder, expected=args.expected, show_ok=args.show_ok)
