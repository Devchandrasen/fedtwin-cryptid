"""Validate an IEEE manuscript abstract word count without shell-sensitive regex quoting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def abstract_word_count(path: str | Path) -> int:
    text = Path(path).read_text(encoding="utf-8")
    start = r"\begin{abstract}"
    end = r"\end{abstract}"
    if start not in text or end not in text:
        raise ValueError(f"abstract delimiters are missing from {path}")
    abstract = text.split(start, 1)[1].split(end, 1)[0]
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript")
    parser.add_argument("--minimum", type=int, default=150)
    parser.add_argument("--maximum", type=int, default=250)
    args = parser.parse_args()
    count = abstract_word_count(args.manuscript)
    if not args.minimum <= count <= args.maximum:
        parser.error(f"abstract has {count} words; expected {args.minimum}--{args.maximum}")
    print(f"abstract_words={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
