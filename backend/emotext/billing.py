"""Credit packs sold through Stripe Checkout, granted by a signed webhook.

Uses Stripe's REST API directly (two calls), so there is no SDK dependency.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from emotext.db import CreditEntry, User
from emotext.settings import get_settings

STRIPE_API = "https://api.stripe.com/v1"


class BillingError(RuntimeError):
    pass


def create_checkout(user: User, pack: str) -> str:
    """Create a Stripe Checkout session for a credit pack and return its URL."""
    s = get_settings()
    if not s.stripe_secret_key:
        raise BillingError("Payments are not configured (EMOTEXT_STRIPE_SECRET_KEY).")
    if pack not in s.credit_packs:
        raise BillingError(f"Unknown pack '{pack}'")
    credits, cents = s.credit_packs[pack]
    form = {
        "mode": "payment",
        "success_url": f"{s.public_url}/billing?status=success",
        "cancel_url": f"{s.public_url}/billing?status=cancelled",
        "client_reference_id": user.id,
        "line_items[0][quantity]": "1",
        "line_items[0][price_data][currency]": "usd",
        "line_items[0][price_data][unit_amount]": str(cents),
        "line_items[0][price_data][product_data][name]": f"{credits:,} Emo_Text credits",
        "metadata[user_id]": user.id,
        "metadata[credits]": str(credits),
        "metadata[pack]": pack,
    }
    if user.email:
        form["customer_email"] = user.email
    resp = httpx.post(f"{STRIPE_API}/checkout/sessions", data=form,
                      auth=(s.stripe_secret_key, ""), timeout=20)
    if resp.status_code >= 400:
        raise BillingError(f"Stripe error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["url"]


def verify_webhook(payload: bytes, header: str | None, secret: str,
                   tolerance_s: int = 300) -> dict:
    """Check a Stripe-Signature header (t=...,v1=...) and return the parsed event."""
    if not header:
        raise BillingError("Missing Stripe-Signature header")
    parts: dict[str, list[str]] = {}
    for item in header.split(","):
        key, _, value = item.partition("=")
        parts.setdefault(key.strip(), []).append(value.strip())
    try:
        timestamp = int(parts["t"][0])
    except (KeyError, ValueError) as exc:
        raise BillingError("Malformed Stripe-Signature header") from exc
    if abs(time.time() - timestamp) > tolerance_s:
        raise BillingError("Webhook timestamp outside tolerance")
    expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + payload,
                        hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, sig) for sig in parts.get("v1", [])):
        raise BillingError("Bad webhook signature")
    return json.loads(payload)


def handle_event(db: Session, event: dict) -> bool:
    """Grant credits for a paid checkout. Idempotent on the event id; returns True if granted."""
    if event.get("type") != "checkout.session.completed":
        return False
    obj = event["data"]["object"]
    if obj.get("payment_status") != "paid":
        return False
    meta = obj.get("metadata") or {}
    user_id, credits = meta.get("user_id"), int(meta.get("credits") or 0)
    if not user_id or credits <= 0 or db.get(User, user_id) is None:
        return False
    if db.scalar(select(CreditEntry).where(CreditEntry.external_ref == event["id"])):
        return False  # Stripe retries webhooks; grant once
    try:
        db.add(CreditEntry(user_id=user_id, delta=credits, reason="purchase",
                           external_ref=event["id"]))
        db.commit()
    except IntegrityError:
        db.rollback()
        return False
    return True
