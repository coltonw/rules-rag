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
#      The CropBox step is LOAD-BEARING — some rulebooks (e.g. Pandemic)
#      are 2-up scans where each PDF page is the same scan with a CropBox
#      selecting only the left or right half. Skipping this duplicates
#      every page in the output.
#   2. Column-aware text extraction via scripts/pdf_columns.py, which
#      runs `pdftotext -bbox-layout` and reorders blocks into proper
#      reading order using a recursive XY-cut pass with wide-block
#      lifting (handles multi-column regions interleaved with full-width
#      headings/paragraphs).
#   3. OCR fallback (ocrmypdf) if extraction yielded < 50 words.
#   4. Unicode NFKC normalization to fold ligatures (ﬁ→fi, ﬂ→fl) and
#      compatibility forms common in PDF text streams.

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

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pdf_columns="$script_dir/pdf_columns.py"
if [ ! -f "$pdf_columns" ]; then
  echo "error: helper not found: $pdf_columns" >&2
  exit 1
fi

for cmd in gs pdftotext pdfinfo perl python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# ---------- Stage 1: normalize via gs (CropBox) -------------------------------
# -dUseCropBox forces gs to crop each page to its CropBox, producing a PDF
# whose page boundary is the visible content. This collapses 2-up "scan +
# half-page CropBox" PDFs into single pages.
if ! gs -sDEVICE=pdfwrite -dUseCropBox -dQUIET -dBATCH -dNOPAUSE \
        -sOutputFile="$tmp/norm.pdf" "$input" 2>/dev/null; then
  if command -v mutool >/dev/null 2>&1; then
    mutool convert -o "$tmp/norm.pdf" "$input"
  else
    echo "error: gs failed and mutool not installed" >&2
    exit 1
  fi
fi

# ---------- Stage 2: column-aware extraction ---------------------------------
# pdf_columns.py invokes `pdftotext -bbox-layout` and runs a layout-analysis
# pass to emit blocks in human reading order. Output already contains the
# "===== PAGE N =====" markers.
extract() {
  local src="$1" dst="$2"
  python3 "$pdf_columns" "$src" "$dst"
  # Squeeze runs of blank lines so paragraph-aware chunkers downstream can
  # rely on \n\n as a real paragraph break.
  cat -s "$dst" > "$dst.tmp" && mv "$dst.tmp" "$dst"
}

extract "$tmp/norm.pdf" "$out"

# ---------- Stage 3: OCR fallback --------------------------------------------
# If the file is a scanned image PDF, stage 2 produces almost nothing.
# ocrmypdf adds a text layer; --skip-text is fast and only touches pages
# with no text, --force-ocr is the heavier fallback.
if [ "$(wc -w < "$out")" -lt 50 ]; then
  if ! command -v ocrmypdf >/dev/null 2>&1; then
    echo "warning: extraction was nearly empty and ocrmypdf is not installed" >&2
    exit 0
  fi
  if ! ocrmypdf --skip-text --quiet "$input" "$tmp/ocr.pdf" 2>/dev/null; then
    ocrmypdf --force-ocr --quiet "$input" "$tmp/ocr.pdf"
  fi
  # Re-normalize the OCR output too (CropBox may differ).
  if ! gs -sDEVICE=pdfwrite -dUseCropBox -dQUIET -dBATCH -dNOPAUSE \
          -sOutputFile="$tmp/ocr_norm.pdf" "$tmp/ocr.pdf" 2>/dev/null; then
    cp "$tmp/ocr.pdf" "$tmp/ocr_norm.pdf"
  fi
  extract "$tmp/ocr_norm.pdf" "$out"
fi

# ---------- Stage 4: Unicode NFKC normalization ------------------------------
# PDF text streams routinely contain presentation-form ligatures (U+FB01 ﬁ,
# U+FB02 ﬂ, etc.) and compatibility characters that hurt both display and
# lexical search.
perl -CSDA -MUnicode::Normalize -i -pe '$_ = NFKC($_)' "$out"
