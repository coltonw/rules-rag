#!/usr/bin/env bash
#
# Extract text from a PDF rulebook with per-page markers.
#
# Usage: pdf2txt.sh <input.pdf> [output.txt]
#
# If output is omitted, writes to <input>.txt next to the source.
# Output format: each page is preceded by a line "===== PAGE N =====".
#
# Pipeline:
#   1. Normalize via Ghostscript (-dUseCropBox), with mutool as fallback.
#   2. Per-page extraction with pdftotext -layout.
#   3. OCR fallback (ocrmypdf) if extraction yielded < 50 words.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <input.pdf> [output.txt]" >&2
  exit 2
fi

input="$1"
out="${2:-${input%.pdf}.txt}"

if [ ! -f "$input" ]; then
  echo "error: input not found: $input" >&2
  exit 1
fi

for cmd in gs pdftotext pdfinfo; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Stage 1: normalize. -dUseCropBox forces the CropBox to be the page boundary,
# which fixes the most common layout-extraction failure mode. mutool is a
# different engine and occasionally salvages files gs chokes on.
if ! gs -sDEVICE=pdfwrite -dUseCropBox -dQUIET -dBATCH -dNOPAUSE \
        -sOutputFile="$tmp/norm.pdf" "$input" 2>/dev/null; then
  if command -v mutool >/dev/null 2>&1; then
    mutool convert -o "$tmp/norm.pdf" "$input"
  else
    echo "error: gs failed and mutool not installed" >&2
    exit 1
  fi
fi

# Stage 2: per-page extraction. Each page is independent so page numbers are
# trustworthy and there's no cross-page bleed.
extract() {
  local src="$1" dst="$2"
  local n
  n=$(pdfinfo "$src" | awk '/^Pages:/ {print $2}')
  : > "$dst"
  for p in $(seq 1 "$n"); do
    printf '\n===== PAGE %d =====\n' "$p" >> "$dst"
    pdftotext -layout -f "$p" -l "$p" "$src" - >> "$dst" 2>/dev/null || true
  done
}

extract "$tmp/norm.pdf" "$out"

# Stage 3: OCR fallback. If the file is a scanned image PDF, stage 2 produces
# almost nothing. ocrmypdf adds a text layer; --skip-text is fast and only
# touches pages with no text, --force-ocr is the heavier fallback.
if [ "$(wc -w < "$out")" -lt 50 ]; then
  if ! command -v ocrmypdf >/dev/null 2>&1; then
    echo "warning: extraction was nearly empty and ocrmypdf is not installed" >&2
    exit 0
  fi
  if ! ocrmypdf --skip-text --quiet "$input" "$tmp/ocr.pdf" 2>/dev/null; then
    ocrmypdf --force-ocr --quiet "$input" "$tmp/ocr.pdf"
  fi
  extract "$tmp/ocr.pdf" "$out"
fi
