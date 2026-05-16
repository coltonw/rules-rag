#!/usr/bin/env bash
# Run pdf2txt.sh for every PDF in data/pdfs/ that doesn't have a .txt yet.
# Skips image-only-rulebooks/ — those need OCR handled separately.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pdf_dir="$script_dir/../data/pdfs"

total=0
skipped=0
processed=0
failed=0

while IFS= read -r -d '' pdf; do
    total=$((total + 1))
    txt="${pdf%.pdf}.txt"

    if [ -f "$txt" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    echo "==> $(basename "$pdf")"
    if "$script_dir/pdf2txt.sh" "$pdf"; then
        processed=$((processed + 1))
    else
        echo "    FAILED: $(basename "$pdf")" >&2
        failed=$((failed + 1))
    fi
done < <(find "$pdf_dir" -maxdepth 1 -name "*.pdf" -print0 | sort -z)

echo ""
echo "Done. $processed extracted, $skipped already had .txt, $failed failed (of $total total)."
