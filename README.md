# okforge-vision-ocr

OCR a scanned PDF **and extract its photographs** with a single
vision-LLM call per page. Works against any OpenAI-compatible endpoint
serving a vision-language model — built and tuned against a
locally-hosted **Qwen3.6-27B-MTP** (via llama.cpp), because that's what
runs well on the author's own hardware (RTX 5090, RTX 6000 Pro
Blackwell). A couple of behaviors described below are genuinely
Qwen-VL-family specific, called out where they apply.

For each page it:

1. Extracts the embedded scan directly when the page *is* one full-page
   image (no re-encode loss), otherwise renders at 300 DPI.
2. Makes **one combined chat call** asking for (a) a clean Markdown
   transcription in reading order — including the text *inside*
   photographed signs and plaques, as blockquotes — and (b) bounding boxes
   for real photographs only (decorative banners/borders/maps excluded).
3. Crops each photo itself, rejects degenerate crops (near-uniform pixel
   variance), saves the rest as JPEGs, and splices `![]()` references into
   the Markdown at each photo's position in the reading order.

## Install

```bash
pip install okforge-vision-ocr
```

This installs two commands, `okforge-vision-ocr` and
`okforge-translate-pages`. Running from a source checkout without
installing works too — see "Configuration" below.

```bash
okforge-vision-ocr scanned.pdf out.md              # whole document
okforge-vision-ocr scanned.pdf out.md --pages 16   # one page
okforge-vision-ocr scanned.pdf out.md --pages 5-12 # range
okforge-vision-ocr scanned.pdf out.md --figures    # also extract drawings
```

By default only real photographs are extracted. `--figures` widens the
scope to every illustration — line drawings, engravings, diagrams, and
pictorial vignettes/head-pieces — excluding only plain rules and borders,
page numbers, and enlarged drop-cap initials. Use it for books whose
figures are drawn rather than photographed; the strict default exists for
documents where the non-photo graphics are decorative page furniture you
don't want.

New to this or to okforge? [**GETTING_STARTED.md**](https://github.com/okforge/okforge/blob/main/GETTING_STARTED.md)
in the main okforge repo is a full beginner walkthrough covering venv
setup, installing both tools, local vs. cloud (OpenRouter) model setup,
and a complete scan-to-wiki example.

Output: `out.md` (images referenced relatively from `out_images/`), plus
`out.pages.json` — `[{"page": N, "content": str, "images": [{"path": str}]}]`
— so downstream tooling can attach real page numbers to every chunk. This
is the same contract [okforge](https://github.com/okforge/okforge) reads
directly for real `(p. N)` citations in compiled summaries — this tool is
okforge's companion pre-conversion step for scans and photo catalogs.

## Difficult tables (`--think --tables`)

Complex tables — multi-level headers, row-group labels spanning several
rows, cells that span columns — get mangled by fast single-pass
transcription. `--think` turns the model's reasoning on for the page
(with a 4x token budget; any inline `<think>` block is stripped from
the output), and `--tables` appends an information-first prompt: convey
what the table MEANS — what is looked up, by which criteria, what each
cell says — as a flat table or explicit bulleted records, rather than
mimicking the printed grid. Numbers, units, and footnote markers are
preserved exactly.

```bash
okforge-vision-ocr scanned.pdf out.md --pages 68 --think --tables
```

Reasoning is deliberately OFF otherwise: without `enable_thinking:
false` per request, **Qwen3-family models** burn the whole token budget on
reasoning and return empty content. This flag is sent on every request
regardless of model — harmless with servers that ignore unrecognized
`extra_body` keys, meaningful specifically for Qwen3 chat templates.
`--prompt-extra "…"` appends one-off instructions for a stubborn page.

## Crop padding (`--crop-pad`)

The model emits tight bounding boxes by training, and prompt hints
barely loosen them — padding is a deterministic post-step, not a prompt
matter. `--crop-pad 5` expands every detected box by 5% of its own
width/height on each side before cropping, clamped to the page edges
(range 0–50; default 0 keeps the historical tight crops). Set it
per-KB instead via `OKFORGE_VISION_CROP_PAD` in the `.env` the tool
already reads (cwd, then `~/.config/openkb/.env`); the flag wins over
the env var.

When a figure sits inside printed text, padding necessarily pulls
fragments of the surrounding words into the crop. That trade is
deliberate and worth it: stray text at a crop's edge costs nothing,
while a clipped figure or half a caption is lost content. Tune the
percentage for "never clips", not for "no text visible".

```bash
okforge-vision-ocr scanned.pdf out.md --pages 2 --crop-pad 5
```

## The coordinate calibration trick (Qwen-VL-family specific)

**Qwen-VL** grounding responses through llama.cpp return bounding boxes
**normalized to 0–1000 of the image dimensions**, regardless of input
resolution — measured identical at 640/1206/2513 px input widths. If your
boxes look "off by ~1.6×", that's just `height/1000`. This tool scales by
`dim/1000`, falls back to raw-pixel interpretation when any coordinate
exceeds 1000, and accepts both `bbox` and `bbox_2d` keys (the model
alternates).

This is a **Qwen-VL-family convention, not a universal one** — other
vision models return boxes as 0–1 normalized floats, raw pixel
coordinates, or point-based formats. If you point this tool at a
non-Qwen model, check your first few crops; the box math may need
adjusting for your model's actual grounding output format.

One more Qwen-specific detail baked in: `chat_template_kwargs:
{"enable_thinking": false}` is mandatory on thinking-capable Qwen3
builds — otherwise reasoning consumes the whole token budget and
`content` comes back empty (see above).

Because every serving stack spells "don't reason" differently, both
tools send **both** dialects on every request: `chat_template_kwargs:
{"enable_thinking": …}` (llama.cpp/vLLM) and `reasoning:
{"enabled": …}` (hosted routers like OpenRouter). Each side ignores the
other's key — verified against llama.cpp and OpenRouter with
`qwen3.6-27b`. Without the router dialect, a thinking-capable Qwen3
model on OpenRouter reasons on every page into the shared 4096
`max_tokens` budget: dense pages can truncate, and every page pays
reasoning cost and latency.

Model-agnostic detail: pages are processed strictly sequentially (one
in-flight call), which single-slot LLM servers of any kind need.

## Translating the output (`okforge-translate-pages`)

For non-English documents, keep OCR faithful and translate afterwards.
`okforge-translate-pages` reads the `.pages.json`, makes one **text-only**
call per page (skipping pages that are only photos), and writes a
translated `.md` + `.pages.json` with every image reference kept
byte-identical — put the output in the same directory as the source `.md`
and both language versions share one `_images/` directory:

```bash
okforge-translate-pages out.pages.json out_en.md --from Catalan --to English
```

First production run: a 116-page Catalan firearms catalog, 158 photos,
zero lost image references.

## Configuration

Environment (or a `.env` in the working directory):

```
OPENAI_API_BASE=http://your-host:8080/v1
LLM_API_KEY=no-key
OKFORGE_VISION_MODEL=Qwen3.6-27B-MTP   # optional override — any vision-capable model your endpoint serves
```

Any OpenAI-compatible hosted service works the same way — e.g.
OpenRouter is `OPENAI_API_BASE=https://openrouter.ai/api/v1` with your
`sk-or-...` key and a vision-capable model slug like
`OKFORGE_VISION_MODEL=qwen/qwen3.6-27b`. Thinking suppression works
there too — both tools send OpenRouter's `reasoning` dialect alongside
the llama.cpp one (see above).

From a source checkout, without installing the package: `pip install -r
requirements.txt` (pymupdf, pillow, openai, python-dotenv), then run
`python okforge_vision_ocr.py ...` / `python translate_pages.py ...`
directly.

## Provenance

Built 2026-07-03 to ingest a 60-page pure-scan brochure (zero text layer)
after several docling-based pipelines failed on it in different ways —
one combined OCR + image-extraction call per page turned out simpler and
more reliable than a general-purpose document-conversion service for
this kind of material.

The canonical working copy lives inside a private tooling repo (under
the filename `qwen_page_ocr.py`, wired into that repo's own pipeline and
kept there under its original name for internal continuity); this repo
is a renamed, cleaned-up standalone publication of the same script for
anyone who wants the tool without the rest of that pipeline. Changes are
ported across deliberately, not auto-synced.

## License

MIT — see [LICENSE](LICENSE).
