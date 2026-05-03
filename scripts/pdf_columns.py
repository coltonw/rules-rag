#!/usr/bin/env python3
"""
Column-aware text extraction from PDFs.

Reads a PDF, runs `pdftotext -bbox-layout` to get text blocks with bounding
boxes, then applies a layout-analysis pass that produces blocks in proper
human reading order — top-to-bottom, left-to-right within each column,
with multi-column regions correctly separated from full-width
headers/footers above and below.

Why this exists: `pdftotext -layout` preserves the visual grid by padding
with spaces, but as soon as you collapse runs of whitespace (which we do
for embedding/BM25), N columns of prose interleave into garbage. This
script reads structure from coordinates instead of from a text grid.

Algorithm: a recursive XY-cut variant with two key extensions over the
classic algorithm:

  1. Wide-block lifting. A block whose width spans >= 60% of the current
     region's width is treated as a section separator (heading, full-width
     paragraph, etc). Wide blocks slice the region into horizontal bands;
     XY-cut runs inside each band on the remaining narrow blocks. This is
     what lets a 2-column GAME END section be detected even when full-width
     paragraphs sit immediately above and below it.

  2. Region-relative thresholds. Minimum gap widths are computed against
     the current region size, not the whole page. So a column gutter that
     looks small relative to the page (because the region is one half of a
     2-up scan) is still proportionally a real gutter inside its half.

Usage:
    pdf_columns.py <input.pdf>            # writes to stdout
    pdf_columns.py <input.pdf> <out.txt>  # writes to file

Each page is preceded by a line "===== PAGE N =====".
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------- Data model -------------------------------------------------------


@dataclass
class Word:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    text: str


@dataclass
class Line:
    y_min: float
    words: List[Word] = field(default_factory=list)

    def render(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.x_min))


@dataclass
class Block:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    lines: List[Line] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def render(self) -> str:
        return "\n".join(
            ln.render() for ln in sorted(self.lines, key=lambda ln: ln.y_min)
        )


@dataclass
class Page:
    width: float
    height: float
    blocks: List[Block] = field(default_factory=list)


# ---------- Parser -----------------------------------------------------------

# pdftotext -bbox-layout emits XHTML with a strict, regular nesting of
# <page> > <flow> > <block> > <line> > <word>. Regex parsing is faster and
# more forgiving than xml.etree here (the document has an XHTML doctype that
# some parsers try to fetch over the network).

_PAGE_RE = re.compile(
    r'<page\s+width="([^"]+)"\s+height="([^"]+)">(.*?)</page>',
    re.DOTALL,
)
_BLOCK_RE = re.compile(
    r'<block\s+xMin="([^"]+)"\s+yMin="([^"]+)"\s+xMax="([^"]+)"\s+yMax="([^"]+)">'
    r"(.*?)</block>",
    re.DOTALL,
)
_LINE_RE = re.compile(
    r'<line\s+xMin="[^"]+"\s+yMin="([^"]+)"\s+xMax="[^"]+"\s+yMax="[^"]+">'
    r"(.*?)</line>",
    re.DOTALL,
)
_WORD_RE = re.compile(
    r'<word\s+xMin="([^"]+)"\s+yMin="([^"]+)"\s+xMax="([^"]+)"\s+yMax="([^"]+)">'
    r"(.*?)</word>",
    re.DOTALL,
)


def parse_bbox_xhtml(text: str) -> List[Page]:
    pages: List[Page] = []
    for pm in _PAGE_RE.finditer(text):
        page = Page(width=float(pm.group(1)), height=float(pm.group(2)))
        for bm in _BLOCK_RE.finditer(pm.group(3)):
            block = Block(
                x_min=float(bm.group(1)),
                y_min=float(bm.group(2)),
                x_max=float(bm.group(3)),
                y_max=float(bm.group(4)),
            )
            for lm in _LINE_RE.finditer(bm.group(5)):
                line = Line(y_min=float(lm.group(1)))
                for wm in _WORD_RE.finditer(lm.group(2)):
                    raw = wm.group(5)
                    if not raw:
                        continue
                    word = Word(
                        x_min=float(wm.group(1)),
                        y_min=float(wm.group(2)),
                        x_max=float(wm.group(3)),
                        y_max=float(wm.group(4)),
                        text=html.unescape(raw),
                    )
                    line.words.append(word)
                if line.words:
                    block.lines.append(line)
            if block.lines:
                page.blocks.append(block)
        pages.append(page)
    return pages


# ---------- Layout analysis --------------------------------------------------


# Tunables. The defaults work for typical rulebooks; if a specific PDF
# misorders, these are the first knobs to turn.
MIN_GAP_X_FRAC = 0.015   # column gutter must be >= 1.5% of region width
MIN_GAP_Y_FRAC = 0.015   # band gap must be >= 1.5% of region height
WIDE_FRAC      = 0.60    # block is "wide" if it covers >= 60% of region
MAX_DEPTH      = 24


def _largest_gap(
    intervals: List[Tuple[float, float]],
    bound_lo: float,
    bound_hi: float,
    min_width: float,
) -> Optional[Tuple[float, float]]:
    """Find the widest interior gap (>= min_width) in a 1D coverage map.

    Margins on either side of the bounds are ignored — we only care about
    gaps that actually separate two non-empty regions. Returns (gap_lo,
    gap_hi) for the widest such gap, or None.
    """
    if len(intervals) < 2:
        return None
    sorted_iv = sorted(intervals)
    merged: List[List[float]] = [list(sorted_iv[0])]
    for lo, hi in sorted_iv[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    best: Optional[Tuple[float, float]] = None
    best_w = min_width
    for i in range(len(merged) - 1):
        gap_lo = merged[i][1]
        gap_hi = merged[i + 1][0]
        w = gap_hi - gap_lo
        if w > best_w:
            best = (gap_lo, gap_hi)
            best_w = w
    return best


def _xy_cut_narrow(
    blocks: List[Block],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    depth: int,
) -> List[Block]:
    """Standard XY-cut on a set of narrow blocks (no wide spanners).

    Picks whichever axis has the largest dominant gap relative to its span,
    cuts there, and recurses. Falls back to (y, x) sort when no usable gap
    exists.
    """
    if not blocks:
        return []
    if len(blocks) == 1 or depth > MAX_DEPTH:
        return sorted(blocks, key=lambda b: (b.y_min, b.x_min))

    region_w = max(x_hi - x_lo, 1.0)
    region_h = max(y_hi - y_lo, 1.0)
    min_gap_x = region_w * MIN_GAP_X_FRAC
    min_gap_y = region_h * MIN_GAP_Y_FRAC

    h_gap = _largest_gap(
        [(b.y_min, b.y_max) for b in blocks], y_lo, y_hi, min_gap_y
    )
    v_gap = _largest_gap(
        [(b.x_min, b.x_max) for b in blocks], x_lo, x_hi, min_gap_x
    )
    h_score = (h_gap[1] - h_gap[0]) / region_h if h_gap else 0.0
    v_score = (v_gap[1] - v_gap[0]) / region_w if v_gap else 0.0

    if h_gap and h_score >= v_score:
        cut_lo, cut_hi = h_gap
        top = [b for b in blocks if b.y_max <= cut_lo]
        bot = [b for b in blocks if b.y_min >= cut_hi]
        return _xy_cut_narrow(
            top, x_lo, x_hi, y_lo, cut_lo, depth + 1
        ) + _xy_cut_narrow(
            bot, x_lo, x_hi, cut_hi, y_hi, depth + 1
        )
    if v_gap:
        cut_lo, cut_hi = v_gap
        left = [b for b in blocks if b.x_max <= cut_lo]
        right = [b for b in blocks if b.x_min >= cut_hi]
        return _xy_cut_narrow(
            left, x_lo, cut_lo, y_lo, y_hi, depth + 1
        ) + _xy_cut_narrow(
            right, cut_hi, x_hi, y_lo, y_hi, depth + 1
        )
    return sorted(blocks, key=lambda b: (b.y_min, b.x_min))


def order_blocks(
    blocks: List[Block],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    depth: int = 0,
) -> List[Block]:
    """Top-level reading-order analysis.

    Splits the region into bands using wide spanner blocks (full-width
    headings or paragraphs), then runs standard XY-cut on the narrow blocks
    inside each band. The band-and-XY-cut combination handles the common
    case where multi-column regions are interspersed with full-width prose.
    """
    if not blocks:
        return []
    if len(blocks) == 1 or depth > MAX_DEPTH:
        return sorted(blocks, key=lambda b: (b.y_min, b.x_min))

    region_w = max(x_hi - x_lo, 1.0)
    wide_cutoff = region_w * WIDE_FRAC
    wide = [b for b in blocks if b.width >= wide_cutoff]
    narrow = [b for b in blocks if b.width < wide_cutoff]

    if not wide or not narrow:
        # No mixed wide/narrow content: regular XY-cut handles this fine
        # (it will, e.g., split a 2-up page on the central gutter).
        return _xy_cut_narrow(blocks, x_lo, x_hi, y_lo, y_hi, depth)

    # Wide blocks slice the region into horizontal bands. Sort them by y to
    # walk top-to-bottom; coalesce overlapping wide blocks so we don't
    # produce empty bands between back-to-back spanners.
    wide_sorted = sorted(wide, key=lambda b: b.y_min)
    result: List[Block] = []
    cursor_y = y_lo

    i = 0
    while i < len(wide_sorted):
        # Coalesce a run of overlapping wide blocks into one separator group.
        run_start = i
        run_y_max = wide_sorted[i].y_max
        i += 1
        while i < len(wide_sorted) and wide_sorted[i].y_min <= run_y_max:
            run_y_max = max(run_y_max, wide_sorted[i].y_max)
            i += 1
        run = wide_sorted[run_start:i]
        run_y_min = min(b.y_min for b in run)

        # Process the band of narrow blocks between cursor_y and the start of
        # this wide-block run.
        band = [
            b for b in narrow
            if b.y_min >= cursor_y and b.y_max <= run_y_min
        ]
        if band:
            result.extend(
                order_blocks(band, x_lo, x_hi, cursor_y, run_y_min, depth + 1)
            )
        # Emit the wide blocks in y order. (They may also have a small x
        # ordering among themselves if two are at the same y.)
        result.extend(sorted(run, key=lambda b: (b.y_min, b.x_min)))
        cursor_y = max(cursor_y, run_y_max)

    # Final band below all wide blocks.
    tail = [b for b in narrow if b.y_min >= cursor_y]
    if tail:
        result.extend(
            order_blocks(tail, x_lo, x_hi, cursor_y, y_hi, depth + 1)
        )

    # Safety net: any narrow block that didn't fit cleanly into a band
    # (shouldn't happen geometrically, but if a narrow block straddles a
    # wide block in y, it'd be skipped). Re-insert at the end sorted by y.
    seen = {id(b) for b in result}
    leftovers = [b for b in narrow if id(b) not in seen]
    if leftovers:
        result.extend(sorted(leftovers, key=lambda b: (b.y_min, b.x_min)))
    return result


# ---------- Rendering --------------------------------------------------------


def render_page(page: Page, page_num: int) -> str:
    ordered = order_blocks(
        page.blocks, x_lo=0.0, x_hi=page.width, y_lo=0.0, y_hi=page.height
    )
    parts = [f"===== PAGE {page_num} ====="]
    for block in ordered:
        body = block.render().rstrip()
        if body:
            parts.append(body)
    return "\n\n".join(parts) + "\n"


# ---------- Driver -----------------------------------------------------------


def extract(pdf_path: str) -> str:
    proc = subprocess.run(
        ["pdftotext", "-bbox-layout", pdf_path, "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    pages = parse_bbox_xhtml(proc.stdout)
    return "".join(render_page(p, i) for i, p in enumerate(pages, 1))


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("usage: pdf_columns.py <input.pdf> [output.txt]", file=sys.stderr)
        return 2
    pdf = sys.argv[1]
    out = extract(pdf)
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
