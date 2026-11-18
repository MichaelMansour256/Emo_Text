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
docker compose up --build        # from the repo root: web on :3000, API docs on :8000/docs
```

Without Docker, run jobs inline (no Redis):

```bash
EMOTEXT_CELERY_EAGER=true uvicorn emotext.api:app --app-dir backend --reload
```

Every `/v1` route needs a user, except signed file links and the Stripe webhook.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness |
| `GET /ready` | readiness: the database answers |
| `GET /v1/me` | the caller and their credit balance (accounts are created on first call) |
| `GET /v1/options` | engines (with voices, and why one is unavailable), directors, credit packs |
| `POST /v1/projects` | upload a book with `rights_confirmed=true`; it is parsed at once (chapters, word counts) |
| `GET /v1/projects`, `GET /v1/projects/{id}` | list / detail, including progress and signed audio links |
| `POST /v1/projects/{id}/render` | `{provider, voice, director, max_chapters, preview, cast}`: charges credits and queues the render |
| `GET /v1/projects/{id}/events` | server-sent events with status and progress until the render ends |
| `GET /v1/projects/{id}/script` | every rendered line with speaker, emotion and voice |
| `GET /v1/projects/{id}/files/{path}?expires=&sig=` | signed download, so `<audio>` needs no auth header |
| `DELETE /v1/projects/{id}` | delete the project, its upload and its audio |
| `GET /v1/billing/ledger`, `POST /v1/billing/checkout`, `POST /v1/billing/webhook` | credit history, Stripe Checkout, Stripe webhook |

### Accounts and auth

`EMOTEXT_AUTH_MODE=dev` (default) trusts an `X-Dev-User: email` header: for local use only.
For anything reachable from the internet use `EMOTEXT_AUTH_MODE=jwt` with your auth provider's
tokens, verified with either `EMOTEXT_AUTH_JWKS_URL` (Clerk, Supabase: RS256/ES256) or
`EMOTEXT_AUTH_JWT_SECRET` (Supabase legacy HS256), plus optional `EMOTEXT_AUTH_AUDIENCE` /
`EMOTEXT_AUTH_ISSUER`. Set `EMOTEXT_SECRET_KEY` to a long random value: it signs audio links.

Accounts, projects and the credit ledger live in `EMOTEXT_DATABASE_URL` (default SQLite in the
data dir; Compose uses Postgres). The schema is managed by Alembic (`emotext/migrations`):
migrations run at startup and with `emotext db upgrade` (Fly runs that on every release). After
changing a model, add a revision with `cd backend && alembic revision --autogenerate -m "..."`;
a test fails if the models and migrations drift apart.

### Files

`EMOTEXT_STORAGE_BACKEND=local` (default) keeps uploads and audio in the data dir and serves them
through signed API links. `s3` uses any S3-compatible bucket (Cloudflare R2 in production):
renders happen in a scratch folder and are uploaded when done, and listeners stream straight
from the bucket through presigned URLs. Settings: `EMOTEXT_S3_BUCKET`, `EMOTEXT_S3_ENDPOINT_URL`,
`EMOTEXT_S3_REGION` (`auto` for R2), `EMOTEXT_S3_ACCESS_KEY_ID`, `EMOTEXT_S3_SECRET_ACCESS_KEY`.

### Limits and monitoring

Per user: `EMOTEXT_MAX_UPLOADS_PER_DAY` (20), `EMOTEXT_MAX_ACTIVE_RENDERS` (2) and
`EMOTEXT_MAX_BOOK_WORDS` (400,000); uploads also need the rights confirmation. Set
`EMOTEXT_SENTRY_DSN` to report errors from the API and the worker (request bodies, and so
book text, are never sent).

### Credits

One credit renders one word. New accounts get `EMOTEXT_SIGNUP_CREDITS` (20,000). The preview
(the opening ~`EMOTEXT_PREVIEW_WORDS` words of chapter 1) is free, and a failed render is
refunded automatically. Packs are sold through Stripe Checkout when `EMOTEXT_STRIPE_SECRET_KEY`
and `EMOTEXT_STRIPE_WEBHOOK_SECRET` are set; point a Stripe webhook for
`checkout.session.completed` at `/v1/billing/webhook`. Grants are idempotent per event.

## Settings

Environment variables (or `.env`), all prefixed `EMOTEXT_` except the provider keys:

| Area | Variables |
|------|-----------|
| Basics | `DATA_DIR`, `REDIS_URL`, `CELERY_EAGER`, `MAX_UPLOAD_MB`, `ENVIRONMENT`, `PUBLIC_URL`, `CORS_ORIGINS` |
| Voices | `TTS_PROVIDER`, `TTS_VOICE`, `OPENAI_TTS_MODEL`, plus `OPENAI_API_KEY` |
| Director | `DIRECTOR`, `DIRECTOR_MODEL`, `DIRECTOR_EFFORT`, plus `ANTHROPIC_API_KEY` |
| Accounts | `DATABASE_URL`, `AUTH_MODE`, `AUTH_JWKS_URL`, `AUTH_JWT_SECRET`, `AUTH_AUDIENCE`, `AUTH_ISSUER`, `SECRET_KEY` |
| Credits | `SIGNUP_CREDITS`, `PREVIEW_WORDS`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` |
| Files | `STORAGE_BACKEND`, `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `DOWNLOAD_TTL_S` |
| Limits | `MAX_UPLOADS_PER_DAY`, `MAX_ACTIVE_RENDERS`, `MAX_BOOK_WORDS` |
| Errors | `SENTRY_DSN` |

Deploying: see [`../DEPLOY.md`](../DEPLOY.md).

## Tests

```bash
cd backend && pytest -q && ruff check .
```
