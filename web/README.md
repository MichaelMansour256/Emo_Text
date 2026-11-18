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

Today the app uses the backend's dev sign-in: any email, no password, stored in
`localStorage`. Before going public, add Clerk or Supabase Auth and connect it in one place:

```ts
import { setTokenGetter } from "@/lib/api";
setTokenGetter(() => getToken());   // e.g. Clerk's useAuth().getToken
```

and run the backend with `EMOTEXT_AUTH_MODE=jwt` and the provider's JWKS URL
(see `backend/README.md` → Accounts and auth).

## Pages

| Route | What it does |
|-------|--------------|
| `/` | sign-in, or the library with drag-and-drop upload |
| `/projects/[id]` | chapters, render options and cost, live progress, player, cast, delete |
| `/billing` | balance, credit packs (Stripe Checkout), history |

`npm run build` type-checks and builds; CI runs it on every push.
