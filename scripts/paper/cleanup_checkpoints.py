#!/usr/bin/env python3
"""Safe checkpoint cleanup helper (dry-run by default).

Default behavior targets obvious low-value files:
- smoke-test checkpoints (`*smoke*`)
- explicit glob patterns passed by user

Use `--delete` to actually remove files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def bytes_to_gib(n: int) -> float:
    return float(n) / (1024**3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dry-run cleanup planner for checkpoints")
    ap.add_argument("--checkpoints-dir", type=Path, default=Path("outputs/checkpoints"))
    ap.add_argument("--include-glob", nargs="*", default=["*smoke*.pt"], help="Glob patterns under checkpoints dir to include")
    ap.add_argument("--exclude-glob", nargs="*", default=[], help="Glob patterns to exclude")
    ap.add_argument("--min-size-mb", type=float, default=0.0, help="Only include files >= this size (MB)")
    ap.add_argument("--delete", action="store_true", help="Actually delete matched files (default: dry-run)")
    args = ap.parse_args()

    root = args.checkpoints_dir
    if not root.exists():
        raise SystemExit(f"Checkpoint dir not found: {root}")

    matches: List[Path] = []
    for pat in args.include_glob:
        matches.extend(root.glob(pat))
    uniq = []
    seen = set()
    for p in sorted(matches):
        if not p.is_file():
            continue
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)

    def excluded(p: Path) -> bool:
        return any(p.match(glob) for glob in args.exclude_glob)

    min_bytes = int(args.min_size_mb * 1024 * 1024)
    keep = [p for p in uniq if (p.stat().st_size >= min_bytes and not excluded(p))]
    total = sum(p.stat().st_size for p in keep)

    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"[{mode}] {len(keep)} checkpoint(s), total {bytes_to_gib(total):.2f} GiB")
    for p in keep:
        sz_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {sz_mb:8.1f} MB  {p}")

    if args.delete:
        deleted = 0
        for p in keep:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"[WARN] Failed to delete {p}: {e}")
        print(f"[DELETE] Removed {deleted}/{len(keep)} files; reclaimed ~{bytes_to_gib(total):.2f} GiB (before failures)")


if __name__ == "__main__":
    main()

