#!/usr/bin/env python3
"""
Update manifest.toml by matching PDFs in data/pdfs/ against games in export.csv.
Uses the exact safe_filename() logic from download_rulebooks.py to compute
expected PDF names deterministically — no fuzzy matching needed.

Usage:
    python3 scripts/update_manifest.py
"""

import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
CSV_PATH = ROOT / "data" / "export.csv"
MANIFEST_OUT = PDF_DIR / "manifest.toml"


# Exact copy of safe_filename() from download_rulebooks.py
def safe_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\.{2,}", "-", name)
    DASH_CHARS = set("-–—/:")
    DROP_CHARS = set("''‘’“”!?,;()[]&+#@*^~.")
    result = []
    for c in name:
        if c.isalnum():
            result.append(c)
        elif c.isspace() or c in DASH_CHARS:
            result.append("-")
        elif c in DROP_CHARS:
            pass
    return re.sub(r"-{2,}", "-", "".join(result)).strip("-")


def load_games(csv_path: Path) -> list[dict]:
    games = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"].strip().strip('"')
            if name:
                games.append({"name": name, "game_id": row["Game ID"].strip()})
    return games


def build_pdf_index(pdf_dir: Path) -> dict[str, Path]:
    """Return {filename: absolute_path} for all PDFs directly in pdf_dir."""
    return {
        p.name: p
        for p in pdf_dir.iterdir()
        if p.suffix == ".pdf" and ":Zone" not in p.name
    }


def find_pdf(game_name: str, pdf_index: dict[str, Path]) -> Path | None:
    stem = safe_filename(game_name)
    # Primary pattern used by download_rulebooks.py
    for candidate in [f"{stem}-rulebook.pdf", f"{stem}.pdf"]:
        if candidate in pdf_index:
            return pdf_index[candidate]
    return None


def write_manifest(matched: list[dict], out_path: Path) -> None:
    lines = []
    for m in matched:
        pdf_path: Path = m["path"]
        # Use a path relative to the repo root, with .txt extension (the extracted text)
        rel = pdf_path.relative_to(ROOT)
        txt_file = "./" + str(rel).removesuffix(".pdf") + ".txt"
        lines.append("[[document]]")
        lines.append(f'file = "{txt_file}"')
        lines.append(f'game = "{m["name"]}"')
        lines.append('doc_type = "rules"')
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    games = load_games(CSV_PATH)
    pdf_index = build_pdf_index(PDF_DIR)

    print(f"Found {len(pdf_index)} PDFs, {len(games)} games in CSV\n")

    matched = []
    not_downloaded = []

    for g in games:
        path = find_pdf(g["name"], pdf_index)
        if path is None:
            not_downloaded.append(g["name"])
        else:
            matched.append({"name": g["name"], "path": path})

    matched.sort(key=lambda m: m["name"].lower())
    write_manifest(matched, MANIFEST_OUT)

    print(f"Matched {len(matched)} PDFs → manifest.toml")

    if not_downloaded:
        print(f"\n--- No PDF found ({len(not_downloaded)}) ---")
        for name in not_downloaded:
            stem = safe_filename(name)
            print(f"  {name!r:45s}  (expected: {stem}-rulebook.pdf)")

    # Check for orphan PDFs (in data/pdfs/ but matched to no game)
    matched_files = {m["path"].name for m in matched}
    orphans = [name for name in pdf_index if name not in matched_files]
    if orphans:
        print(f"\n--- PDFs with no matching game in CSV ({len(orphans)}) ---")
        for name in sorted(orphans):
            print(f"  {name}")


if __name__ == "__main__":
    main()
