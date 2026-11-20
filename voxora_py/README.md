# Voxora (pure Python)

Voxora answers a spoken or typed question from a 1,000-document knowledge base (Hindi
passages from `ai4bharat/MSMARCO-XI`), shows the evidence it used, says whether the answer
is grounded, times every stage and keeps a per-session history.

This is a rebuild of the React + FastAPI version. Python does the real work: serving, page
rendering (Jinja2), state, retrieval, generation and grounding. The browser gets
server-rendered HTML, HTMX from a CDN for text queries and partial swaps, and one small
script (`static/app.js`) for the microphone, the voice upload and the history drawer.

## Setup (macOS / Linux)

```bash
cd voxora_py
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp -R ../voxora/data ./data
mkdir -p onnx && cp ../voxora/onnx/tokenizer.json onnx/
.venv/bin/python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('Xenova/multilingual-e5-small', 'onnx/model_int8.onnx'), 'onnx/model_int8.onnx')"
.venv/bin/python scripts/build_css.py
cp .env.example .env   # then fill in the keys
python3 main.py
```

Open http://localhost:8001. On Windows use `.venv\Scripts\python`.

`main.py` refuses to start if the ONNX model, the indexes or `static/app.css` are missing and
prints the command that fixes each one. `python3 main.py --reload` reloads on code and
template changes and runs the Tailwind CLI in watch mode. `PORT=8002 python3 main.py` changes
the port.

`.env` keys: `SARVAM_API_KEY`, `GROQ_API_KEY`, `LLM_PROVIDER=groq`,
`GROQ_MODEL=openai/gpt-oss-20b`, `SESSION_SECRET` (any long random string; if empty, a
random one is generated per process and sessions reset on restart). Every other value in
`voxora/config.py` can also be overridden by an environment variable of the same name, for
example `MIN_DENSE_SCORE=0.9`.

## Layout

```
main.py                 entry point (checks files, runs Uvicorn)
voxora/config.py        pydantic-settings: keys, model, paths, thresholds
voxora/services.py      lifespan loader: corpus, FAISS, BM25, embedder, Groq, Sarvam
voxora/embedder.py      multilingual-e5-small on ONNX Runtime (query:/passage: prefixes)
voxora/retrieval.py     FAISS top 10, BM25 fallback below 0.30, keep 5
voxora/generation.py    Groq call with a strict JSON schema
voxora/guardrails.py    unsafe-query patterns, grounding check
voxora/pipeline.py      run_rag(): the one pipeline shared by text and voice
voxora/stt.py           Sarvam speech-to-text
voxora/telemetry.py     latency records and p50/p70/p100
voxora/sessions.py      per-browser history (newest first, max 50) and current result
voxora/api.py           JSON routes
voxora/ui.py            HTML routes and Jinja helpers
voxora/app.py           FastAPI app factory (sessions, static files, error handlers)
templates/              base.html, pages/, partials/ (ported 1:1 from the React components)
static/input.css        Tailwind source: the React index.css plus @source lines
static/app.js           the one script (under 200 lines)
scripts/                build_css, build_corpus, build_embeddings, build_faiss_index,
                        build_bm25_index, compare_screenshots
tests/                  pytest suite (Groq and Sarvam mocked)
```

## Routes

| Method | Path | Returns |
|---|---|---|
| GET | `/`, `/features`, `/app` | Landing, Features, Voice Query workspace (re-shows the session's current result) |
| POST | `/ui/query` | HTMX text submit: results + telemetry, or the error box |
| POST | `/ui/voice` | multipart `audio` from app.js: same partial |
| GET | `/ui/history/{n}` | re-show history item Q{n} as the current result |
| GET | `/ui/theme/toggle?next=` | flips the `theme` cookie, 303 back to `next` |
| POST | `/api/query` | JSON `{query, language}` → response schema |
| POST | `/api/voice-query` | multipart `audio` → response schema |
| GET | `/api/latency/summary?sample_size=10` | p50/p70/p100 per stage |
| GET | `/healthz` | `{"status":"ok","documents":1000}` |

Upstream failures never surface as a bare 500: a bad Groq or Sarvam key is HTTP 502
("Groq API key is invalid" / "Sarvam API key is invalid"), a timeout is 504, and the HTML
routes show the same message in the error box.

## Tests

```bash
.venv/bin/python -m pytest
```

`tests/test_telemetry.py::test_summary_matches_old_backend` imports the old backend and
compares summaries for identical records; it runs when `../voxora` exists or `VOXORA_REF`
points at the old project, and is skipped otherwise.

## Screenshot comparison

```bash
cd ../voxora/frontend && npx vite --port 5174          # the old React UI
cd ../../voxora_py && .venv/bin/playwright install chromium
.venv/bin/python scripts/compare_screenshots.py --old-url http://localhost:5174
```

Both UIs get identical fixture data (the React app's API calls are answered by Playwright;
the Python app runs in-process with the pipeline patched), so no API keys are needed.
Screenshots go to `screenshots/<view>-<state>-<theme>-<width>-{old,new}.png`, diff masks to
`screenshots/_diff/`, and a table to `screenshots/REPORT.md`.

## Behaviour notes

- Ask anything. Questions the 1,000 documents cover are answered from them with citations
  (GROUNDED RESPONSE / UNVERIFIED); any other question is answered from the model's general
  knowledge and labelled GENERAL KNOWLEDGE. Unsafe questions are still refused.
- Retrieval searches two FAISS indexes, the Hindi passages (`dev.faiss`) and their English
  translations (`dev_en.faiss`), and the LLM context uses the English text.

- The theme is a cookie read by Python, so it survives reloads with no light flash (the React
  app reset to light on every load).
- On the real data the E5 dense score is always about 0.78 to 0.89, even for nonsense or
  empty queries, so the 0.30 BM25 fallback threshold from the original never fires. It is
  kept as specified and configurable (`MIN_DENSE_SCORE`). Tests cover the fallback with a
  stub embedder and with a raised threshold on the real index.
- `/api/latency/summary` uses the last `sample_size` runs for the percentiles. The old
  backend counted the last `sample_size` runs but computed percentiles over all runs; the two
  agree whenever there are at most `sample_size` runs.
- Refusals (unsafe, no context) return every key of the response schema
  (`fallback_used`, `grounding_confidence` included), and are not added to the latency records.
