Here's a battle-tested pipeline. Three stages, each handles a different class of weird PDF, and the whole thing is just bash you can shell out to from Rust.

## The pipeline

```
1. Normalize the PDF        → strips the MediaBox/CropBox trap, fixes most layout weirdness
2. Extract text per page    → reliable page numbers via the loop, not the form-feed
3. OCR fallback if empty    → catches scanned PDFs that have no embedded text
```

## Stage 1: Normalize with Ghostscript

Ghostscript has `-dUseCropBox`, which re-renders the PDF with the CropBox treated as the page boundary — exactly the override you wanted from `pdftotext -cropbox`. It also flattens transparency, normalizes fonts, and generally produces a sane file.

```bash
gs -sDEVICE=pdfwrite -dUseCropBox -dQUIET -dBATCH -dNOPAUSE \
   -sOutputFile=/tmp/normalized.pdf input.pdf
```

If `gs` isn't installed: `sudo apt install ghostscript` (it's almost always already there).

**Backup if `gs` somehow doesn't fix it:** `mutool convert -o /tmp/normalized.pdf input.pdf` does similar work via a different engine, so a second pass on the rare uncooperative file works.

## Stage 2: Per-page extraction with markers

Run `pdftotext` once per page so each page is independent — this also prevents weird cross-page bleed and gives you precise page numbers for citations.

```bash
pages=$(pdfinfo /tmp/normalized.pdf | awk '/^Pages:/ {print $2}')
: > /tmp/output.txt
for p in $(seq 1 "$pages"); do
  printf '\n===== PAGE %d =====\n' "$p" >> /tmp/output.txt
  pdftotext -layout -f "$p" -l "$p" /tmp/normalized.pdf - >> /tmp/output.txt
done
```

Use `-layout` for rulebooks (preserves columns/tables); for prose-heavy docs, drop it. If you don't know in advance, `-layout` is safer for vector DB — it keeps related text adjacent.

## Stage 3: OCR fallback for scanned PDFs

If the extraction returned nearly nothing, the PDF is probably scanned images with no embedded text. `ocrmypdf` adds a text layer in place:

```bash
words=$(wc -w < /tmp/output.txt)
if [ "$words" -lt 50 ]; then
  ocrmypdf --skip-text --quiet input.pdf /tmp/ocr.pdf || \
    ocrmypdf --force-ocr --quiet input.pdf /tmp/ocr.pdf
  # then re-run stage 2 on /tmp/ocr.pdf
fi
```

`sudo apt install ocrmypdf` (pulls in tesseract). Use `--skip-text` first (fast, only OCRs pages without text); fall back to `--force-ocr` for files where the embedded text is broken.

## Putting it together

Save this as a script and shell out to it from Rust with one argument (the input PDF):

```bash
#!/usr/bin/env bash
set -euo pipefail

input="$1"
out="${2:-${input%.pdf}.txt}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# 1. Normalize (CropBox-respecting)
gs -sDEVICE=pdfwrite -dUseCropBox -dQUIET -dBATCH -dNOPAUSE \
   -sOutputFile="$tmp/norm.pdf" "$input" 2>/dev/null \
   || mutool convert -o "$tmp/norm.pdf" "$input"

extract() {
  local src="$1" dst="$2"
  local n; n=$(pdfinfo "$src" | awk '/^Pages:/ {print $2}')
  : > "$dst"
  for p in $(seq 1 "$n"); do
    printf '\n===== PAGE %d =====\n' "$p" >> "$dst"
    pdftotext -layout -f "$p" -l "$p" "$src" - >> "$dst" 2>/dev/null || true
  done
}

# 2. First extraction attempt
extract "$tmp/norm.pdf" "$out"

# 3. OCR fallback if nearly empty
if [ "$(wc -w < "$out")" -lt 50 ]; then
  ocrmypdf --skip-text --quiet "$input" "$tmp/ocr.pdf" 2>/dev/null \
    || ocrmypdf --force-ocr --quiet "$input" "$tmp/ocr.pdf"
  extract "$tmp/ocr.pdf" "$out"
fi
```

From Rust:
```rust
use std::process::Command;

let status = Command::new("/path/to/pdf2txt.sh")
    .arg(&pdf_path)
    .arg(&out_path)
    .status()?;
```

## Why this is reliable for rulebooks

- **CropBox issue** → solved by `-dUseCropBox` at stage 1, before pdftotext sees the file.
- **Wrong page-text ordering** → per-page extraction prevents cross-page contamination.
- **Multi-column layouts** → `-layout` preserves them; if your DB chunks per-page that's fine, the columns become adjacent paragraphs.
- **Scanned rulebooks** → stage 3 handles them.
- **Page citations** → the `===== PAGE N =====` markers are easy to parse in Rust to attach `page` metadata to each chunk.

## When this *won't* be enough

Some rulebooks (Cole Wehrle / Leder Games designs come to mind) have illustrated callouts, marginalia, or non-rectangular text flow that no positional extractor handles well. For those, your last resort is `pdfplumber` (Python) with custom column cropping per page — but that's a one-off, not a pipeline. For 90%+ of board game rulebooks, the three stages above will be enough.

## Chunking for vectors

Don't ingest the whole text file as one document. Split on the `===== PAGE N =====` markers and store each page (or each paragraph within a page) as a separate vector with `{game, page}` metadata. That way retrieval cites which page it came from — which is what you originally wanted from the page-numbering question.