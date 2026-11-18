import hashlib
import hmac
import json
import time
import uuid

import jwt
import pytest
from conftest import requires_ffmpeg
from fastapi.testclient import TestClient

from emotext import worker
from emotext.api import app
from emotext.settings import get_settings

client = TestClient(app)


def as_user(email: str | None = None) -> dict:
    return {"X-Dev-User": email or f"{uuid.uuid4().hex[:8]}@test"}


def upload(path, headers, name=None, rights=True):
    with path.open("rb") as fh:
        return client.post("/v1/projects", files={"file": (name or path.name, fh)},
                           data={"rights_confirmed": str(rights).lower()}, headers=headers)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_new_user_gets_signup_credits():
    me = client.get("/v1/me", headers=as_user()).json()
    assert me["credits"] == get_settings().signup_credits and me["auth_mode"] == "dev"


def test_options_lists_engines_and_voices():
    opts = client.get("/v1/options").json()
    tone = next(p for p in opts["providers"] if p["name"] == "tone")
    assert tone["available"] and {"mid", "low"} <= {v["id"] for v in tone["voices"]}
    assert "heuristic" in opts["directors"]


def test_upload_parses_chapters_and_is_private(sample_pdf):
    user = as_user()
    resp = upload(sample_pdf, user)
    assert resp.status_code == 201
    project = resp.json()
    assert project["title"] == "The Silent Road" and project["status"] == "draft"
    assert [c["title"] for c in project["chapters"]] == ["Chapter 1: The Letter",
                                                         "Chapter 2: The Road"]
    assert [p["id"] for p in client.get("/v1/projects", headers=user).json()] == [project["id"]]
    assert client.get(f"/v1/projects/{project['id']}", headers=as_user()).status_code == 404


def test_rejects_unsupported_type():
    resp = client.post("/v1/projects", files={"file": ("x.exe", b"MZ")},
                       data={"rights_confirmed": "true"}, headers=as_user())
    assert resp.status_code == 415


@requires_ffmpeg
def test_preview_is_free_and_files_need_a_valid_signature(sample_txt):
    user = as_user()
    pid = upload(sample_txt, user).json()["id"]
    resp = client.post(f"/v1/projects/{pid}/render", headers=user,
                       json={"provider": "tone", "preview": True})
    assert resp.status_code == 202
    project = resp.json()
    assert project["status"] == "done", project  # Celery runs eagerly in tests
    assert project["result"]["preview"] and project["result"]["m4b"] is None
    assert client.get("/v1/me", headers=user).json()["credits"] == get_settings().signup_credits

    url = project["result"]["chapters"][0]["url"]
    assert len(client.get(url).content) > 1000
    assert client.get(url.replace("sig=", "sig=x")).status_code == 403
    assert client.get(f"/v1/projects/{pid}/files/manifest.json").status_code == 403

    events = client.get(f"/v1/projects/{pid}/events", headers=user).text
    assert json.loads(events.split("data: ")[1])["status"] == "done"
    script = client.get(f"/v1/projects/{pid}/script", headers=user).json()
    assert script[0]["lines"]

    assert client.delete(f"/v1/projects/{pid}", headers=user).status_code == 204
    assert client.get(f"/v1/projects/{pid}", headers=user).status_code == 404


@requires_ffmpeg
def test_full_render_charges_words(sample_txt):
    user = as_user()
    project = upload(sample_txt, user).json()
    words = project["chapters"][0]["words"]
    done = client.post(f"/v1/projects/{project['id']}/render", headers=user,
                       json={"provider": "tone", "max_chapters": 1}).json()
    assert done["status"] == "done" and done["charged"] == words
    assert client.get("/v1/me", headers=user).json()["credits"] == (
        get_settings().signup_credits - words)
    reasons = [e["reason"] for e in client.get("/v1/billing/ledger", headers=user).json()]
    assert reasons == ["render", "signup"]


def test_render_without_enough_credits_is_402(sample_txt, monkeypatch):
    monkeypatch.setattr(get_settings(), "signup_credits", 5)
    user = as_user()
    pid = upload(sample_txt, user).json()["id"]
    resp = client.post(f"/v1/projects/{pid}/render", headers=user, json={"provider": "tone"})
    assert resp.status_code == 402
    assert client.get(f"/v1/projects/{pid}", headers=user).json()["status"] == "draft"


def test_failed_render_is_refunded(sample_txt, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(worker, "build_book", boom)
    user = as_user()
    pid = upload(sample_txt, user).json()["id"]
    project = client.post(f"/v1/projects/{pid}/render", headers=user,
                          json={"provider": "tone"}).json()
    assert project["status"] == "failed" and "engine exploded" in project["error"]
    assert client.get("/v1/me", headers=user).json()["credits"] == get_settings().signup_credits


# -- billing -----------------------------------------------------------------------------------


def stripe_post(event: dict, secret: str):
    body = json.dumps(event).encode()
    t = int(time.time())
    sig = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return client.post("/v1/billing/webhook", content=body,
                       headers={"Stripe-Signature": f"t={t},v1={sig}"})


def test_stripe_webhook_grants_credits_once(monkeypatch):
    monkeypatch.setattr(get_settings(), "stripe_webhook_secret", "whsec_test")
    user = as_user()
    me = client.get("/v1/me", headers=user).json()
    event = {"id": f"evt_{uuid.uuid4().hex}", "type": "checkout.session.completed",
             "data": {"object": {"payment_status": "paid",
                                 "metadata": {"user_id": me["id"], "credits": "100000"}}}}
    assert stripe_post(event, "whsec_test").json() == {"granted": True}
    assert stripe_post(event, "whsec_test").json() == {"granted": False}  # retried delivery
    assert stripe_post(event, "wrong").status_code == 400
    assert client.get("/v1/me", headers=user).json()["credits"] == me["credits"] + 100_000


def test_checkout_without_stripe_is_a_clear_error():
    resp = client.post("/v1/billing/checkout", json={"pack": "starter"}, headers=as_user())
    assert resp.status_code == 400 and "not configured" in resp.json()["detail"]


# -- jwt auth ----------------------------------------------------------------------------------


@pytest.fixture
def jwt_mode(monkeypatch):
    monkeypatch.setattr(get_settings(), "auth_mode", "jwt")
    monkeypatch.setattr(get_settings(), "auth_jwt_secret", "s3cret-for-tests-only-0123456789")


def token(sub: str, secret="s3cret-for-tests-only-0123456789", exp_in=600) -> str:
    return jwt.encode({"sub": sub, "email": f"{sub}@x.io", "exp": int(time.time()) + exp_in},
                      secret, algorithm="HS256")


def test_jwt_mode_requires_a_valid_token(jwt_mode):
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers=as_user("dev@localhost")).status_code == 401
    bad = {"Authorization": f"Bearer {token('u1', secret='another-secret-of-enough-length!')}"}
    assert client.get("/v1/me", headers=bad).status_code == 401
    expired = {"Authorization": f"Bearer {token('u1', exp_in=-60)}"}
    assert client.get("/v1/me", headers=expired).status_code == 401
    ok = client.get("/v1/me", headers={"Authorization": f"Bearer {token('user_42')}"})
    assert ok.status_code == 200 and ok.json()["email"] == "user_42@x.io"


def test_untitled_text_upload_is_named_after_the_file(sample_txt):
    project = upload(sample_txt, as_user(), name="my_great_novel.txt").json()
    assert project["title"] == "my great novel"
    assert project["chapters"][0]["title"] != project["id"]


# -- limits ------------------------------------------------------------------------------------


def test_upload_requires_rights_confirmation(sample_txt):
    assert upload(sample_txt, as_user(), rights=False).status_code == 422


def test_daily_upload_limit(sample_txt, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_uploads_per_day", 1)
    user = as_user()
    assert upload(sample_txt, user).status_code == 201
    assert upload(sample_txt, user).status_code == 429


def test_book_word_limit(sample_txt, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_book_words", 10)
    assert upload(sample_txt, as_user()).status_code == 413


def test_concurrent_render_limit(sample_txt, monkeypatch):
    from emotext import db

    monkeypatch.setattr(get_settings(), "max_active_renders", 1)
    user = as_user()
    busy, other = (upload(sample_txt, user).json()["id"] for _ in range(2))
    with db.session() as s:
        s.get(db.Project, busy).status = "rendering"
        s.commit()
    resp = client.post(f"/v1/projects/{other}/render", headers=user,
                       json={"provider": "tone", "preview": True})
    assert resp.status_code == 429


def test_ready_checks_the_database():
    assert client.get("/ready").json() == {"status": "ready"}
