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
#   2. Layout-aware extraction via Marker (marker-pdf), running on GPU
#      when CUDA is available. Marker uses ML-based layout detection,
#      reading-order analysis, and OCR for image PDFs, producing
#      Markdown with structure preserved (headings, paragraphs, tables).
#      Page boundaries are emitted as "{N}===PAGE_BREAK===" tokens which
#      we rewrite to our "===== PAGE M =====" format (Marker is 0-indexed,
#      our format is 1-indexed).
#   3. Unicode NFKC normalization to fold ligatures (ﬁ→fi, ﬂ→fl) and
#      compatibility forms common in PDF text streams. Marker does most
#      of this internally; this is a belt-and-suspenders pass.

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
venv_python="$script_dir/.venv/bin/python"
marker_cli="$script_dir/.venv/bin/marker_single"

if [ ! -x "$marker_cli" ]; then
  echo "error: marker not installed at $marker_cli" >&2
  echo "       create the venv and install: python3 -m venv $script_dir/.venv && $script_dir/.venv/bin/pip install marker-pdf" >&2
  exit 1
fi

for cmd in gs perl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# ---------- Stage 1: normalize via gs (CropBox) ------------------------------
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

# ---------- Stage 2: Marker extraction ---------------------------------------
# Marker writes <basename>/<basename>.md into the output dir. We feed it the
# CropBox-normalized PDF (named norm.pdf) so the output basename is stable.
#
# --force_ocr is LOAD-BEARING. Marker's default text-extraction path uses
# pypdfium, which silently drops glyphs whose ToUnicode CMap is partial —
# e.g. Pandemic's stylized initial-cap "Th" → "e clock is ticking". OCR via
# surya re-reads the rendered glyph and recovers the character. Slower but
# robust across the kinds of font weirdness rulebook designers love.
"$marker_cli" \
  --output_format markdown \
  --paginate_output \
  --page_separator "===PAGE_BREAK===" \
  --output_dir "$tmp/marker" \
  --disable_image_extraction \
  --force_ocr \
  "$tmp/norm.pdf"

md="$tmp/marker/norm/norm.md"
if [ ! -f "$md" ]; then
  echo "error: marker produced no markdown at $md" >&2
  exit 1
fi

# Rewrite Marker's pagination tokens to our format. Marker emits one token
# per page (0-indexed) at the START of each page, so the first page gets
# "{0}===PAGE_BREAK===" → "===== PAGE 1 =====".
"$venv_python" - "$md" "$out" <<'PY'
import re, sys

md_path, out_path = sys.argv[1:3]
with open(md_path, encoding="utf-8") as f:
    text = f.read()

def repl(m):
    return f"===== PAGE {int(m.group(1)) + 1} ====="

text = re.sub(r"\{(\d+)\}===PAGE_BREAK===", repl, text)

# Trim leading whitespace so the file starts with "===== PAGE 1 =====".
text = text.lstrip()

# Squeeze runs of >2 blank lines so paragraph-aware chunkers can rely on
# \n\n as a paragraph break.
text = re.sub(r"\n{3,}", "\n\n", text)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
PY

# ---------- Stage 3: Unicode NFKC normalization ------------------------------
perl -CSDA -MUnicode::Normalize -i -pe '$_ = NFKC($_)' "$out"
