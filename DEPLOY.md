# Deploying Emo_Text

The production setup this repo is configured for:

| Piece | Service | Config |
|-------|---------|--------|
| Web app | Vercel (Next.js) | `web/`, environment variables below |
| API + render worker | Fly.io, one image, two process groups | `backend/fly.toml` |
| Database | Postgres: Neon, Supabase or Fly Postgres | `EMOTEXT_DATABASE_URL` |
| Job queue | Redis: Upstash or Fly Redis | `EMOTEXT_REDIS_URL` |
| Files | Cloudflare R2 (any S3-compatible bucket works) | `EMOTEXT_S3_*` |
| Sign-in | Clerk | publishable key (web), JWKS URL (API) |
| Payments | Stripe Checkout + webhook | `EMOTEXT_STRIPE_*` |
| Errors | Sentry | `EMOTEXT_SENTRY_DSN` |

Each step below says what to create and which values to copy. Nothing here needs code changes.

## 1. Accounts to create

1. **Clerk**: create an application, enable email (and Google if you like).
   Copy the *publishable key*, the *secret key*, and the *Frontend API URL*
   (e.g. `https://clever-cat-12.clerk.accounts.dev`).
   Optional: in **Sessions → Customize session token** add
   `{"email": "{{user.primary_email_address}}"}` so the API knows each user's email
   (Stripe receipts use it).
2. **Cloudflare R2**: create a bucket (e.g. `emotext-prod`) and an API token with
   *Object Read & Write* on it. Copy the account id, access key id and secret.
   Under the bucket's CORS policy allow `GET` from your web domain.
3. **Postgres**: create a database and copy its URL. Use the
   `postgresql+psycopg://` scheme, e.g. `postgresql+psycopg://user:pass@host/db?sslmode=require`.
4. **Redis**: create a database and copy its TLS URL. Celery needs the certificate mode in the URL:
   `rediss://default:pass@host:6379/0?ssl_cert_reqs=required`.
5. **Stripe** (when you're ready to take money): copy the secret key; after step 3 add a webhook
   endpoint `https://<api host>/v1/billing/webhook` for `checkout.session.completed` and copy
   its signing secret.
6. **Sentry** (optional): create a Python project and copy the DSN.

## 2. API and worker on Fly.io

```bash
cd backend
fly launch --no-deploy --copy-config     # keeps fly.toml; pick an app name and region
fly secrets set \
  EMOTEXT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  EMOTEXT_DATABASE_URL="postgresql+psycopg://..." \
  EMOTEXT_REDIS_URL="rediss://...?ssl_cert_reqs=required" \
  EMOTEXT_AUTH_JWKS_URL="https://<clerk frontend api>/.well-known/jwks.json" \
  EMOTEXT_AUTH_ISSUER="https://<clerk frontend api>" \
  EMOTEXT_S3_BUCKET="emotext-prod" \
  EMOTEXT_S3_ENDPOINT_URL="https://<account id>.r2.cloudflarestorage.com" \
  EMOTEXT_S3_ACCESS_KEY_ID="..." EMOTEXT_S3_SECRET_ACCESS_KEY="..." \
  EMOTEXT_PUBLIC_URL="https://<web domain>" \
  EMOTEXT_CORS_ORIGINS='["https://<web domain>"]' \
  OPENAI_API_KEY="..." ANTHROPIC_API_KEY="..." \
  EMOTEXT_SENTRY_DSN="..." \
  EMOTEXT_STRIPE_SECRET_KEY="..." EMOTEXT_STRIPE_WEBHOOK_SECRET="..."
fly deploy
```

`fly deploy` runs `emotext db upgrade` (migrations) before starting new machines, then the
API (`/ready` health check) and the worker. Check with `curl https://<app>.fly.dev/ready`.

Automatic deploys: create a token with `fly tokens create deploy` and save it as the
`FLY_API_TOKEN` repository secret. CI then deploys the API on every push to `main` after the
tests pass.

## 3. Web app on Vercel

Import the GitHub repo in Vercel, set **Root Directory** to `web`, and add:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | `https://<app>.fly.dev` (the browser calls the API directly) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | from Clerk |
| `CLERK_SECRET_KEY` | from Clerk |

Calling the API directly (instead of through the `/api` rewrite) avoids proxy limits on large
uploads and keeps the progress stream unbuffered; it is why `EMOTEXT_CORS_ORIGINS` is set
above. Vercel deploys every push to `main` and gives each pull request a preview.

## 4. Before inviting beta users

- [ ] Sign up on the live site, upload a public-domain book, render the preview and a chapter.
- [ ] Buy the smallest pack in Stripe test mode, and check the credits arrive (webhook).
- [ ] Trigger an error and see it in Sentry.
- [ ] Read `/terms` and `/privacy` and adjust them; they are plain-language drafts, not legal advice.
- [ ] Set limits for the beta (`EMOTEXT_SIGNUP_CREDITS`, `EMOTEXT_MAX_UPLOADS_PER_DAY`,
      `EMOTEXT_MAX_ACTIVE_RENDERS`, `EMOTEXT_MAX_BOOK_WORDS`) and a spending cap with the
      OpenAI/Anthropic dashboards.
- [ ] Turn on Postgres backups (point-in-time recovery on Neon/Supabase) and R2 bucket
      versioning or lifecycle rules.

## Running the production setup locally

`docker compose up --build` runs the whole stack with dev sign-in and local files. To try the
production paths without the cloud, set `EMOTEXT_STORAGE_BACKEND=s3` against MinIO, or
`EMOTEXT_AUTH_MODE=jwt` with a Clerk development instance.
