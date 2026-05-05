#!/usr/bin/env python3
"""Prompt for one golden-eval record and append it to data/eval/golden.jsonl."""

import json
import re
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "eval" / "golden.jsonl"


def prompt(label, *, required=True):
    while True:
        val = input(f"{label}: ").strip()
        if val or not required:
            return val
        print("  (required)")


def prompt_list(label):
    raw = input(f"{label} (comma-separated, blank for none): ").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else []


def prompt_int_list(label):
    raw = input(f"{label} (comma-separated ints, blank for none): ").strip()
    if not raw:
        return []
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def slugify(name):
    if not name:
        return "general"
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "general"


def next_id(game):
    slug = slugify(game)
    if not GOLDEN.exists():
        return f"{slug}-001"
    n = 0
    with GOLDEN.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id", "").startswith(f"{slug}-"):
                n += 1
    return f"{slug}-{n + 1:03d}"


def main():
    game = prompt("game (blank for cross-game/general)", required=False)
    suggested = next_id(game)
    id_ = input(f"id [{suggested}]: ").strip() or suggested
    question = prompt("question")
    expected_answer_contains = prompt_list("expected_answer_contains")
    expected_pages = prompt_int_list("expected_pages")
    tags = prompt_list("tags")

    record = {
        "id": id_,
        "game": game if game else None,
        "question": question,
        "expected_answer_contains": expected_answer_contains,
        "expected_pages": expected_pages,
        "tags": tags,
    }

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nappended to {GOLDEN}")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
