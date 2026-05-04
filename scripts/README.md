# scripts/

Shell helpers that the Rust pipeline shells out to. The thinking behind the
PDF pipeline lives in [pdf-process.md](../pdf-process.md); this README just
covers how to run the scripts and how to call them from Rust.

## `pdf2txt.sh` — PDF → page-marked text

Extracts text from a rulebook PDF with reliable page numbers. Runs three
stages (gs CropBox normalize → Marker layout-aware extraction with forced
OCR → NFKC normalize) and writes a single text file where each page is
preceded by `===== PAGE N =====`.

### Dependencies

System:

- `ghostscript` (`gs`)
- `perl` with `Unicode::Normalize` (ships with most distros)
- `python3` ≥ 3.10
- `python3-venv`, `python3-pip`

Optional:

- `mupdf-tools` (`mutool`) — fallback if Ghostscript fails on a file

On Debian/Ubuntu:

```bash
sudo apt install ghostscript mupdf-tools python3-venv python3-pip
```

Python (Marker, in a local venv):

```bash
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install marker-pdf
```

Marker pulls PyTorch + CUDA libs (~3 GB) and downloads ~1 GB of model
weights to `~/.cache/datalab/` on first run. With CUDA available, Marker
auto-uses the GPU.

### Usage

```bash
./scripts/pdf2txt.sh <input.pdf> [output.txt]
```

If `output.txt` is omitted, the script writes alongside the input
(`foo.pdf` → `foo.txt`).

Examples:

```bash
./scripts/pdf2txt.sh data/pdfs/root.pdf
./scripts/pdf2txt.sh data/pdfs/root.pdf /tmp/root.txt
```

### Output format

```
===== PAGE 1 =====
<text from page 1>

===== PAGE 2 =====
<text from page 2>
...
```

The marker is a stable separator the ingest crate splits on to attach
`{game, page}` metadata to each chunk.

### Calling from Rust

The script is designed for `Command` shell-out. Resolve the path relative to
`CARGO_MANIFEST_DIR` (or accept it from config) so it works from any cwd.

```rust
use std::path::Path;
use std::process::Command;

pub fn pdf_to_text(pdf: &Path, out: &Path) -> anyhow::Result<()> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/pdf2txt.sh");

    let status = Command::new(&script)
        .arg(pdf)
        .arg(out)
        .status()?;

    if !status.success() {
        anyhow::bail!("pdf2txt.sh failed with status {status}");
    }
    Ok(())
}
```

Notes:

- The script uses `set -euo pipefail`, so a non-zero exit means a real
  failure — propagate it.
- It writes nothing to stdout on success; Marker's progress bars and any
  warnings/errors go to stderr. Inherit stderr (the default) so they
  reach the user's terminal, or pipe it into `tracing` if you want
  structured logs.
- Marker is slow compared to the old `pdftotext` pipeline (~5s/page on a
  3080 with `--force_ocr`, much slower on CPU). Treat ingest as a
  one-time-per-rulebook batch job, not interactive.
