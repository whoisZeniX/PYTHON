# Voxora

Voxora is a question-answering web app. You can type a question or ask it out loud (English or
Hindi). If the answer is in our knowledge base of 1,000 documents, Voxora answers from those
documents and shows which ones it used. If not, it answers from the language model's general
knowledge and labels the answer as such.

Built with FastAPI, Jinja2 templates, HTMX and Tailwind CSS. Answers come from Groq
(`openai/gpt-oss-20b`), voice is transcribed by Sarvam, and documents are searched with the
multilingual E5 embedding model, FAISS and BM25.

## Running it

```bash
cd ~/PROJECTPYTHONFINAL
python3 main.py
```

It opens http://localhost:8001 in your browser. Stop it with CTRL+C.

## Setup on a new machine

```bash
cd voxora_py
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('Xenova/multilingual-e5-small', 'onnx/model_int8.onnx'), 'onnx/model_int8.onnx')"
.venv/bin/python scripts/build_css.py
cp .env.example .env
```

Then put your keys in `.env`:

```
GROQ_API_KEY=...       # https://console.groq.com/keys
SARVAM_API_KEY=...     # https://dashboard.sarvam.ai (only needed for voice)
SESSION_SECRET=...     # any long random string
```

The `data/` folder (corpus and indexes) and `onnx/tokenizer.json` come from the original Voxora
project. The indexes can be rebuilt with the scripts in `scripts/` (see below).

## How a question is answered

1. Questions that try to jailbreak the model ("ignore previous instructions", "system prompt", ...)
   are refused straight away.
2. The question is embedded and matched against the documents. We search two FAISS indexes, one
   built from the Hindi passages and one from their English translations, and keep the best five
   documents. If no document scores above 0.30 we fall back to BM25 keyword search.
3. Groq gets the question plus the five documents. It either answers from them and cites the
   document ids, or answers from general knowledge.
4. Answers based on documents are checked: the cited ids must be real, and enough of the answer's
   words must appear in the documents. That decides the badge:

| Badge | Meaning |
| --- | --- |
| GROUNDED RESPONSE | Answered from the documents and the check passed |
| UNVERIFIED | Answered from the documents but the check failed |
| GENERAL KNOWLEDGE | The documents didn't cover it, the model answered on its own |
| UNSAFE QUERY REFUSED | Blocked by the safety filter |

Timings for each step are recorded and can be opened with **Show timings** under an answer.

## Project layout

```
main.py              starts the server
voxora/              the backend
  config.py          settings from .env
  services.py        loads the data and clients at startup
  embedder.py        E5 embeddings with ONNX Runtime
  retrieval.py       FAISS + BM25 search
  generation.py      the Groq call
  guardrails.py      safety filter and grounding check
  pipeline.py        run_rag(), the whole question flow
  stt.py             Sarvam speech-to-text
  telemetry.py       latency numbers
  sessions.py        per-browser history and theme
  api.py             JSON routes
  ui.py              page routes
  app.py             FastAPI app setup
templates/           HTML (Jinja2)
static/              CSS and app.js (microphone + history drawer)
scripts/             build the CSS and the search indexes
tests/               pytest tests
```

## API

| Method | Path | |
| --- | --- | --- |
| POST | `/api/query` | `{"query": "...", "language": "English"}` returns the answer as JSON |
| POST | `/api/voice-query` | multipart `audio` file, same response |
| GET | `/api/latency/summary` | p50 / p70 / p100 per stage |
| GET | `/healthz` | health check |

Interactive API docs are at http://localhost:8001/docs.

## Rebuilding the indexes

```bash
.venv/bin/python scripts/build_embeddings.py
.venv/bin/python scripts/build_faiss_index.py
.venv/bin/python scripts/build_embeddings.py --column english_text --output data/processed/dev_embeddings_en.npy
.venv/bin/python scripts/build_faiss_index.py --embeddings data/processed/dev_embeddings_en.npy --output data/indexes/dev_en.faiss
.venv/bin/python scripts/build_bm25_index.py
```

## Tests

```bash
.venv/bin/python -m pytest
```

Groq and Sarvam are faked in the tests, so no keys are needed. `tests/test_e2e_browser.py` needs
Playwright's browser (`.venv/bin/playwright install chromium`) and is skipped without it.
