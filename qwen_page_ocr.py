#!/usr/bin/env python3
"""Direct-Qwen pre-conversion: scanned PDF -> Markdown + images + page JSON.

The pre-conversion tool, for documents
where real OCR *and* real image extraction both matter (e.g.
SeminoleWarHeritageTrail's pure-scan brochure). Bypasses docling-serve
entirely — its queue-hang bug, qwen-vl's pictures-never-detected gap, and
granite_docling's unreliable boxes (see SeminoleWarHeritageTrail/README.md)
— and instead makes one chat-completions call per page to the same
Qwen3.6-27B-MTP endpoint every other LLM call in this project uses, asking
for the page's markdown transcription and photo bounding boxes together.

Coordinate calibration (measured, 2026-07-03): the model returns bounding
boxes normalized to 0-1000 of the sent image's dimensions, regardless of
input resolution — verified against ground-truth crops at 640/1206/2513 px
input widths, identical numbers each time. The "~1.6x scale mystery" from
earlier testing was just height/1000. So: pixel = coord * dim / 1000. As a
fallback, a box with any coordinate > 1000 is treated as raw pixels of the
sent image.

Output contract:
    <out_md>            assembled markdown, pages in order, with relative
                        ![](<stem>_images/pN_imgM.jpg) refs inline — the
                        same relative-link convention openkb's .md path
                        already picks up via copy_relative_images().
    <out_md>.pages.json [{"page": N, "content": str, "images":
                        [{"path": str}]}] — the shape
                        _normalize_page_content in openkb/indexer.py
                        accepts.
    <out_md stem>_images/   cropped photos, JPEG.

This produces the artifacts, nothing else — it
does not run `openkb add`.

Usage:
    qwen_page_ocr.py raw/doc.pdf raw/doc.md              # whole document
    qwen_page_ocr.py raw/doc.pdf raw/doc.md --pages 16   # single page
    qwen_page_ocr.py raw/doc.pdf raw/doc.md --pages 5-12 # range

Reads OPENAI_API_BASE and LLM_API_KEY from .env in the current directory
or ~/.config/openkb/.env; defaults to sriaitoo:8080.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageStat

MODEL = os.environ.get("QWEN_OCR_MODEL", "Qwen3.6-27B-MTP")
RENDER_DPI = 300          # for pages that aren't a single full-page scan
JPEG_QUALITY = 90         # both what's sent to the model and what's saved
MAX_TOKENS = 4096         # transcript + JSON; a dense brochure page fits
MAX_TOKENS_THINK = 16384   # --think: reasoning shares the budget, so grow it
MAX_ATTEMPTS = 3          # same 3-retry pattern as indexer.index_long_document
RETRY_WAIT_S = 10
MIN_CROP_PX = 40          # reject slivers
MIN_CROP_STDDEV = 10.0    # reject near-uniform crops (granite's black-box mode)

PHOTO_MARKER_RE = re.compile(r"^[ \t>*-]*<<<PHOTO\s*(\d+)>>>[ \t]*$", re.MULTILINE)
JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)
# The model sometimes echoes the prompt's own section labels ("TASK 1 —
# TRANSCRIPTION", "TASK 2 — PHOTOS"), especially on near-empty pages;
# they are answer scaffolding, not page text.
SCAFFOLD_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?TASK\s*[12]\b[^\n]*$",
    re.MULTILINE | re.IGNORECASE)

PROMPT = """\
You are transcribing one scanned page of a printed document.

TASK 1 — TRANSCRIPTION. Transcribe the page's full text as clean Markdown,
in natural reading order (follow columns if the layout is multi-column).
Use heading levels that reflect the page's visual hierarchy. Transcribe
every text block, including hard-to-read ones such as photographed signs
or plaques — attempt them rather than skipping them. Do not describe the
page; transcribe it. At the point in the reading order where an image
listed in TASK 2 appears in the layout, insert a line containing exactly:
<<<PHOTO 1>>>
(numbering images 1, 2, 3... in reading order). If an image itself
contains readable text (a sign, historical marker, or plaque), transcribe
that text as a Markdown blockquote immediately after the image's marker
line, then continue with the page's own text (captions, next section).

TASK 2 — PHOTOS. After the transcription, output exactly one fenced JSON
block listing @@SCOPE@@:
```json
[{"id": 1, "label": "<short description>", "bbox": [x1, y1, x2, y2]}]
```
where bbox is the image's top-left and bottom-right corners, enclosing the
image itself tightly — the whole image, none of the text around it — and
each id matches a <<<PHOTO n>>> marker. If there are none, output an
empty JSON array and no markers.
"""

# Appended by --tables (pair with --think): information-first table
# handling. The wiki's consumers are LLMs reading markdown — explicit
# statements of meaning beat a mimicked grid, and grid mimicry is what
# breaks on multi-level headers / row groups / spanning cells anyway.
TABLE_ADDENDUM = """
TABLES. If the page contains a table, your goal is to convey the
INFORMATION the table holds — not to reproduce its visual layout.
First reason through what the table actually says: what is being
looked up, by which criteria, and what each cell means — resolving
multi-level headers, row-group labels, spanning cells, and footnote
markers into their plain meaning. Then write that information in the
clearest Markdown form:
- a simple table only if the data is truly flat (one header row,
  same columns on every row);
- otherwise restructure: split it into one small table per group, or
  state each record as a bulleted entry that names every attribute
  explicitly, e.g.
  "- Type II, 55 ft / 5 stories — max floor area 15,000 sq. ft.
  fronting 1 street; 18,000 for 2 streets; 20,000 for 3+ streets;
  increase 100% with complete sprinkler protection."
Preserve every number, unit, and footnote marker exactly as printed.
Make implicit relationships explicit instead of mirroring the grid.
Give the table's title as a heading and its footnotes as plain text.
"""

# TASK 2 scope, swapped by --figures. The photos-only default was tuned on
# a brochure whose non-photos were decorative banners; books whose line
# drawings and engravings ARE content want the wider scope.
PHOTO_SCOPE = (
    "every real photograph on the page (actual photos of places, people, "
    "objects, or signs). Do NOT include decorative graphics, borders, "
    "title banners, drawings, or maps")
FIGURE_SCOPE = (
    "every real photograph AND every illustration on the page: line "
    "drawings, engravings, technical diagrams, and pictorial vignettes or "
    "chapter head-pieces. Exclude only plain rules, plain borders, page "
    "numbers, and enlarged initial capital letters")


def load_client() -> OpenAI:
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path.home() / ".config" / "openkb" / ".env", override=False)
    base_url = os.environ.get("OPENAI_API_BASE", "http://sriaitoo:8080/v1")
    api_key = os.environ.get("LLM_API_KEY", "no-key")
    print(f"LLM endpoint: {base_url}", file=sys.stderr)
    return OpenAI(api_key=api_key, base_url=base_url)


def get_page_image(doc: pymupdf.Document, page: pymupdf.Page) -> Image.Image:
    """Best-quality page image: the embedded scan itself when the page is a
    single full-page image (no re-encode loss), else a 300 DPI render."""
    images = page.get_images()
    if len(images) == 1 and not page.get_text("text").strip():
        xref, w, h = images[0][0], images[0][2], images[0][3]
        page_aspect = page.rect.width / page.rect.height
        if h and abs(w / h - page_aspect) / page_aspect < 0.05:
            data = doc.extract_image(xref)["image"]
            return Image.open(io.BytesIO(data)).convert("RGB")
    pix = page.get_pixmap(dpi=RENDER_DPI)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def call_model(client: OpenAI, img: Image.Image, page_num: int,
               prompt: str, think: bool = False) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode()
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=MAX_TOKENS_THINK if think else MAX_TOKENS,
                temperature=0.2,  # damp run-to-run bbox variance
                # Thinking is off by default — reasoning burns the whole
                # token budget and content comes back empty. --think turns
                # it on (with a bigger budget) for pages that need
                # structural reasoning, e.g. complex tables.
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            text = resp.choices[0].message.content or ""
            # Some server configs inline the reasoning; never let it
            # reach the transcription.
            text = _THINK_RE.sub("", text)
            if text.strip():
                return text
            raise RuntimeError("empty completion content")
        except Exception as exc:  # noqa: BLE001 — retry any request failure
            last_exc = exc
            print(f"  page {page_num} attempt {attempt}/{MAX_ATTEMPTS} "
                  f"failed: {exc}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_S * attempt)
    raise RuntimeError(
        f"LLM call failed for page {page_num} after {MAX_ATTEMPTS} attempts"
    ) from last_exc


def parse_response(text: str) -> tuple[str, list[dict]]:
    """Split the completion into (markdown transcript, photo box list).

    The photo list comes from the *last* ```json fence; anything the model
    wrapped around it is dropped. Unparseable JSON -> no photos, full text
    kept as transcript (markers stripped later by the caller)."""
    boxes: list[dict] = []
    transcript = text
    fences = list(JSON_FENCE_RE.finditer(text))
    if fences:
        last = fences[-1]
        try:
            parsed = json.loads(last.group(1))
            if isinstance(parsed, list):
                boxes = [b for b in parsed if isinstance(b, dict)]
                transcript = text[:last.start()] + text[last.end():]
        except json.JSONDecodeError:
            pass
    transcript = SCAFFOLD_RE.sub("", transcript)
    return transcript.strip(), boxes


def box_to_pixels(box: list, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    W, H = size
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) <= 1000:  # calibrated: 0-1000 normalized
        x1, x2 = x1 * W / 1000, x2 * W / 1000
        y1, y2 = y1 * H / 1000, y2 * H / 1000
    px = (int(max(0, x1)), int(max(0, y1)),
          int(min(W, x2)), int(min(H, y2)))
    if px[2] - px[0] < MIN_CROP_PX or px[3] - px[1] < MIN_CROP_PX:
        return None
    return px


def crop_ok(crop: Image.Image) -> bool:
    """Reject near-uniform crops — granite_docling's solid-black failure mode."""
    stat = ImageStat.Stat(crop.convert("L"))
    return stat.stddev[0] >= MIN_CROP_STDDEV


def process_page(client: OpenAI, doc: pymupdf.Document, page_num: int,
                 images_dir: Path, images_dir_name: str, prompt: str,
                 think: bool = False) -> dict:
    """One page: image -> one LLM call -> transcript + verified photo crops.

    Returns {"page": N, "content": str, "images": [{"path": str}]} with
    image paths relative to the output .md's directory."""
    page = doc[page_num - 1]
    img = get_page_image(doc, page)
    print(f"Page {page_num}: image {img.size[0]}x{img.size[1]}, calling model...",
          file=sys.stderr)
    raw = call_model(client, img, page_num, prompt, think=think)
    transcript, boxes = parse_response(raw)

    # Save each box that survives calibration + sanity checks.
    saved: dict[int, str] = {}   # photo id -> md-relative path
    page_images: list[dict] = []
    img_counter = 0
    for i, b in enumerate(boxes, start=1):
        pid = b.get("id", i)
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = i
        px = box_to_pixels(b.get("bbox") or b.get("bbox_2d") or [], img.size)
        if px is None:
            print(f"  page {page_num} photo {pid}: unusable bbox "
                  f"{b.get('bbox') or b.get('bbox_2d')}; skipped", file=sys.stderr)
            continue
        crop = img.crop(px)
        if not crop_ok(crop):
            print(f"  page {page_num} photo {pid}: crop failed sanity check "
                  f"(near-uniform); skipped", file=sys.stderr)
            continue
        img_counter += 1
        filename = f"p{page_num}_img{img_counter}.jpg"
        images_dir.mkdir(parents=True, exist_ok=True)
        crop.save(images_dir / filename, format="JPEG", quality=JPEG_QUALITY)
        rel = f"{images_dir_name}/{filename}"
        saved[pid] = rel
        page_images.append({"path": rel})
        label = str(b.get("label", "")).strip()
        print(f"  page {page_num} photo {pid}: {crop.size[0]}x{crop.size[1]} "
              f"-> {rel} ({label[:60]})", file=sys.stderr)

    # Replace each <<<PHOTO n>>> marker with its image ref (or drop it if the
    # photo didn't survive); append refs whose marker never appeared.
    def _sub(m: re.Match) -> str:
        rel = saved.pop(int(m.group(1)), None)
        return f"![photo]({rel})" if rel else ""

    content = PHOTO_MARKER_RE.sub(_sub, transcript)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    for rel in saved.values():  # boxes the model never marked in the text
        content += f"\n\n![photo]({rel})"

    return {"page": page_num, "content": content, "images": page_images}


def parse_pages_arg(spec: str | None, page_count: int) -> list[int]:
    if not spec:
        return list(range(1, page_count + 1))
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", spec.strip())
    if not m:
        raise SystemExit(f"--pages must be N or N-M, got: {spec!r}")
    first, last = int(m.group(1)), int(m.group(2) or m.group(1))
    if not (1 <= first <= last <= page_count):
        raise SystemExit(
            f"--pages {spec} out of range for a {page_count}-page document")
    return list(range(first, last + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("out_md", type=Path)
    parser.add_argument("--pages", default=None,
                        help="Page N or range N-M (1-based, inclusive); "
                             "default: whole document")
    parser.add_argument("--pages-json", type=Path, default=None,
                        help="Default: <out_md stem>.pages.json next to out_md")
    parser.add_argument("--figures", action="store_true",
                        help="Also extract figure illustrations (line "
                             "drawings, engravings, diagrams), not just "
                             "photographs")
    parser.add_argument("--think", action="store_true",
                        help="Enable model reasoning (slower, bigger token "
                             "budget) — for pages needing structural "
                             "analysis, e.g. complex tables")
    parser.add_argument("--tables", action="store_true",
                        help="Append table-reconstruction instructions to "
                             "the prompt (pair with --think)")
    parser.add_argument("--prompt-extra", default=None, metavar="TEXT",
                        help="Free-text instructions appended to the prompt")
    args = parser.parse_args()
    prompt = PROMPT.replace(
        "@@SCOPE@@", FIGURE_SCOPE if args.figures else PHOTO_SCOPE)
    if args.tables:
        prompt += TABLE_ADDENDUM
    if args.prompt_extra:
        prompt += "\n" + args.prompt_extra.strip() + "\n"

    out_pages_json = args.pages_json or args.out_md.with_suffix(".pages.json")
    images_dir_name = f"{args.out_md.stem}_images"
    images_dir = args.out_md.parent / images_dir_name

    client = load_client()
    doc = pymupdf.open(str(args.pdf_path))
    page_nums = parse_pages_arg(args.pages, doc.page_count)

    # Strictly one page at a time — one LLM call per page, sequential.
    pages: list[dict] = []
    n_images = 0
    for page_num in page_nums:
        result = process_page(client, doc, page_num, images_dir,
                              images_dir_name, prompt, think=args.think)
        pages.append(result)
        n_images += len(result["images"])
    doc.close()

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "\n\n".join(p["content"] for p in pages if p["content"]) + "\n",
        encoding="utf-8")
    out_pages_json.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(pages)} page(s) -> {args.out_md}\n"
          f"    page array -> {out_pages_json}\n"
          f"    {n_images} photo crop(s) -> {images_dir}/",
          file=sys.stderr)


if __name__ == "__main__":
    main()
