# Emo_Text backend

The audiobook service: turns PDF / EPUB / TXT / Markdown into chaptered MP3s and an M4B audiobook.
See [`../GO_ONLINE_PLAN.md`](../GO_ONLINE_PLAN.md) for where this is going.

## Quick start (no Docker)

Needs Python 3.10+ and `ffmpeg` on PATH.

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -e "backend[dev]"
emotext inspect samples/lighthouse.md
emotext build samples/lighthouse.md --provider gtts
```

Output lands in `output/<book>/`: `chapters/NN - Title.mp3`, `<book>.m4b`, `manifest.json`.

## Speech engines

| `--provider` | Needs | Emotion | Notes |
|--------------|-------|---------|-------|
| `gtts` | internet | no | Free Google Translate voice; dev default. `--voice` is a language code (`en`, `ar`). |
| `openai` | `OPENAI_API_KEY` | yes | `gpt-4o-mini-tts`; emotion becomes delivery instructions. `--voice` e.g. `sage`, `ballad`. |
| `tone` | nothing | no | Offline beeps sized to the text; used by tests and CI. |

New engines implement `emotext.tts.base.TTSProvider` (including `voices()` for casting) and
register in `emotext/tts/__init__.py`.

## Director and cast

The Director decides who says each quotation and how every line is performed; casting then
gives each character a consistent voice from the engine's catalog (the narrator keeps the
default voice).

| `--director` | Needs | What it does |
|--------------|-------|--------------|
| `heuristic` (default) | nothing | dialogue tags, action beats, pronouns, turn-taking; keyword emotions |
| `claude` | `ANTHROPIC_API_KEY` or `ant auth login` | Claude reads each chapter and returns speakers, emotion + intensity per quote, narration mood, and a character list. Only ids come back, so the text is never rewritten. Results are cached; any quote it skips keeps the heuristic answer. Model/effort: `EMOTEXT_DIRECTOR_MODEL` (default `claude-opus-5-5`), `EMOTEXT_DIRECTOR_EFFORT` (default `medium`). |
| `none` | nothing | single narrator voice, no dialogue splitting |

```bash
emotext build book.epub --director claude --provider openai
emotext build book.epub --cast cast.json     # {"Mara": "nova", "Dr. Hale": "onyx"}
```

Each build writes `script.json` (every line with speaker, emotion, intensity, voice) next to
`manifest.json` (which records the character list and cast).

### Eval

`samples/eval/*.txt` are short scenes with a gold speaker label before every quotation
(`{{Mara}}"Run," she said.`). Score a director with:

```bash
emotext eval --director heuristic -v
```

Current results: heuristic 89.7% (35/39), none 5.1%. These scenes were written alongside the
heuristic, so treat them as a regression check rather than a measure of real-book accuracy;
add passages from real (public-domain) books to make the number meaningful.

## API + worker

```bash
docker compose up --build        # from the repo root; API docs at http://localhost:8000/docs
```

Without Docker, run jobs inline (no Redis):

```bash
EMOTEXT_CELERY_EAGER=true uvicorn emotext.api:app --reload
```

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness |
| `POST /v1/books/inspect` | upload a file, get detected chapters (no audio) |
| `POST /v1/jobs` | upload a file (+ `provider`, `voice`, `max_chapters`), queue rendering |
| `GET /v1/jobs/{id}` | status, progress, and result manifest |
| `GET /v1/jobs/{id}/files/{path}` | download a chapter MP3 or the M4B |
| `DELETE /v1/jobs/{id}` | delete a job and its files |

The API has **no authentication yet**; don't expose it publicly until accounts land.

## Settings

Environment variables (or `.env`): `EMOTEXT_DATA_DIR`, `EMOTEXT_REDIS_URL`, `EMOTEXT_CELERY_EAGER`,
`EMOTEXT_TTS_PROVIDER`, `EMOTEXT_TTS_VOICE`, `EMOTEXT_OPENAI_TTS_MODEL`, `EMOTEXT_MAX_UPLOAD_MB`,
`EMOTEXT_DIRECTOR`, `EMOTEXT_DIRECTOR_MODEL`, `EMOTEXT_DIRECTOR_EFFORT`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`. `POST /v1/jobs` also accepts a `director` form field.

## Tests

```bash
cd backend && pytest -q && ruff check .
```
