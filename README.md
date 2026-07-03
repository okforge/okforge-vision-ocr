# qwen-page-ocr

OCR a scanned PDF **and extract its photographs** with a single
vision-LLM call per page, against any OpenAI-compatible endpoint serving a
Qwen-VL-family model (tested with Qwen3.6-27B-MTP on llama.cpp).

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

```bash
python qwen_page_ocr.py scanned.pdf out.md              # whole document
python qwen_page_ocr.py scanned.pdf out.md --pages 16   # one page
python qwen_page_ocr.py scanned.pdf out.md --pages 5-12 # range
```

Output: `out.md` (images referenced relatively from `out_images/`), plus
`out.pages.json` — `[{"page": N, "content": str, "images": [{"path": str}]}]`
— so downstream tooling can attach real page numbers to every chunk.

## The coordinate calibration trick

Qwen-VL grounding responses through llama.cpp return bounding boxes
**normalized to 0–1000 of the image dimensions**, regardless of input
resolution — measured identical at 640/1206/2513 px input widths. If your
boxes look "off by ~1.6×", that's just `height/1000`. This tool scales by
`dim/1000`, falls back to raw-pixel interpretation when any coordinate
exceeds 1000, and accepts both `bbox` and `bbox_2d` keys (the model
alternates).

Two more hard-won details baked in:

- `chat_template_kwargs: {"enable_thinking": false}` is mandatory on
  thinking-capable Qwen builds — otherwise reasoning consumes the whole
  token budget and `content` comes back empty.
- Pages are processed strictly sequentially (one in-flight call), which
  single-slot llama.cpp servers need.

## Configuration

Environment (or a `.env` in the working directory):

```
OPENAI_API_BASE=http://your-host:8080/v1
LLM_API_KEY=no-key
QWEN_OCR_MODEL=Qwen3.6-27B-MTP   # optional override
```

Install: `pip install -r requirements.txt` (pymupdf, pillow, openai,
python-dotenv).

## Provenance

Built 2026-07-03 to ingest a 60-page pure-scan brochure (zero text layer)
into an [OpenKB](https://github.com/VectifyAI/OpenKB) knowledge base after
several docling-serve pipelines failed on it in different ways. The
canonical working copy lives in that machine's OpenKB tooling repo; this
repo is the standalone publication of the same script.
