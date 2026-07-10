#!/usr/bin/env python
"""Translate an okforge_vision_ocr.py output into another
language, page by page, preserving the .md + .pages.json contract.

Reads <in>.pages.json (the per-page source of truth), makes one text-only
LLM call per page, and writes <out>.md + <out>.pages.json with the same
page numbers and the same image entries. Image references inside the
Markdown are kept byte-identical, so the output .md must live in the SAME
directory as the input .md for the relative paths to keep resolving.

Usage:
    translate_pages.py in.pages.json out.md [--to English] [--from Catalan]

Environment (or .env in cwd / ~/.config/openkb/.env):
    OPENAI_API_BASE      default http://localhost:8080/v1
    LLM_API_KEY          default no-key
    OKFORGE_VISION_MODEL default Qwen3.6-27B-MTP
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MODEL = os.environ.get("OKFORGE_VISION_MODEL", "Qwen3.6-27B-MTP")
MAX_TOKENS = 4096
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 10

IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

PROMPT = """\
Translate the following Markdown page transcription from {src} into {dst}.

Rules:
- Preserve the Markdown structure exactly: headings stay headings at the
  same level, blockquotes stay blockquotes, lists stay lists.
- Keep every image reference line (like ![photo](dir/p5_img1.jpg))
  byte-identical and at the same position in the text.
- Do not translate proper names, place names, catalog numbers (e.g.
  "Nº K42"), dates, or measurements — copy them as-is.
- Translate everything else, including text inside blockquotes.
- Output ONLY the translated Markdown. No commentary, no fences.

Markdown to translate:

{content}
"""


def load_client() -> OpenAI:
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path.home() / ".config" / "openkb" / ".env", override=False)
    base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
    api_key = os.environ.get("LLM_API_KEY", "no-key")
    print(f"LLM endpoint: {base_url}", file=sys.stderr)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_model(client: OpenAI, prompt: str, page_num: int) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.2,
                # Both thinking-off dialects: chat_template_kwargs for
                # llama.cpp/vLLM, reasoning for hosted routers (OpenRouter).
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reasoning": {"enabled": False},
                },
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip()
            raise RuntimeError("empty completion content")
        except Exception as exc:  # noqa: BLE001 — retry any request failure
            last_exc = exc
            print(
                f"  page {page_num} attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}",
                file=sys.stderr,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_S * attempt)
    raise RuntimeError(
        f"LLM call failed for page {page_num} after {MAX_ATTEMPTS} attempts"
    ) from last_exc


def translate_page(client: OpenAI, page: dict, src: str, dst: str) -> dict:
    content = page.get("content", "").strip()
    if not content:
        return dict(page)
    # Pure-image pages (only refs / page numbers, no prose) skip the call.
    stripped = IMAGE_REF_RE.sub("", content)
    if not re.search(r"[A-Za-zÀ-ÿ]{3,}", stripped):
        return dict(page)
    out = call_model(
        client, PROMPT.format(src=src, dst=dst, content=content), page["page"]
    )
    # Guardrail: every image reference must survive translation verbatim.
    want = IMAGE_REF_RE.findall(content)
    got = set(IMAGE_REF_RE.findall(out))
    missing = [r for r in want if r not in got]
    if missing:
        print(
            f"  page {page['page']}: {len(missing)} image ref(s) lost in "
            f"translation; re-appending",
            file=sys.stderr,
        )
        out += "\n\n" + "\n\n".join(missing)
    return {**page, "content": out}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("in_pages_json", type=Path)
    parser.add_argument("out_md", type=Path)
    parser.add_argument("--to", dest="dst", default="English")
    parser.add_argument("--from", dest="src", default="the source language")
    args = parser.parse_args()

    pages = json.loads(args.in_pages_json.read_text(encoding="utf-8"))
    client = load_client()

    out_pages: list[dict] = []
    for page in pages:  # strictly sequential — single-slot server friendly
        print(f"Page {page['page']}: translating...", file=sys.stderr)
        out_pages.append(translate_page(client, page, args.src, args.dst))

    out_pages_json = args.out_md.with_suffix(".pages.json")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "\n\n".join(p["content"] for p in out_pages if p["content"]) + "\n",
        encoding="utf-8",
    )
    out_pages_json.write_text(
        json.dumps(out_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"OK: {len(out_pages)} page(s) -> {args.out_md}\n"
        f"    page array -> {out_pages_json}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
