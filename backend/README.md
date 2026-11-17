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

New engines implement `emotext.tts.base.TTSProvider` and register in `emotext/tts/__init__.py`.

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
`OPENAI_API_KEY`.

## Tests

```bash
cd backend && pytest -q && ruff check .
```
