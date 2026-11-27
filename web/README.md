# Emo_Text web

Next.js app for the audiobook service: sign in, upload a book, pick voices and a director,
hear a free preview, render, listen, download the M4B, buy credits.

## Run locally

Start the API first (from the repo root, jobs run inline so Redis isn't needed):

```bash
EMOTEXT_CELERY_EAGER=true uvicorn emotext.api:app --app-dir backend --port 8000
```

Then the web app:

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

The browser calls `/api/*`, which Next forwards to `EMOTEXT_API_URL` (default
`http://127.0.0.1:8000`), so there is no CORS setup and signed audio links work in `<audio>`.
With inline jobs the render request waits until the audio is ready; with a Celery worker
(`docker compose up`) the project page shows live progress over server-sent events.

## Sign-in

Chosen at build time:

- **Clerk** when `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (and `CLERK_SECRET_KEY`) are set. The app
  sends Clerk's session token to the API, which runs with `EMOTEXT_AUTH_MODE=jwt` and Clerk's
  JWKS URL. `proxy.ts` (Next 16's name for middleware) keeps the Clerk session fresh.
- **Development sign-in** otherwise: any email, no password, kept in `localStorage`, matching
  the API's default `EMOTEXT_AUTH_MODE=dev`.

Uploading requires ticking "I own this book or have permission…", which the API records.

## Production

Set `NEXT_PUBLIC_API_URL` to the API's public URL so the browser calls it directly (no proxy
limits on uploads, unbuffered progress), and add the web domain to the API's
`EMOTEXT_CORS_ORIGINS`. Full steps: [`../DEPLOY.md`](../DEPLOY.md).

## Pages

| Route | What it does |
|-------|--------------|
| `/` | sign-in, or the library with drag-and-drop upload |
| `/projects/[id]` | chapters, render options and cost, live progress, player, cast, delete |
| `/projects/[id]/edit` | line editor (speaker, emotion, strength, text), pronunciations, listening-check flags, "Update audio" |
| `/billing` | balance, credit packs (Stripe Checkout), history |
| `/terms`, `/privacy` | plain-language beta drafts; review before launch |

`npm run build` type-checks and builds; CI runs it on every push.
