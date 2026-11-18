"""Who is calling: dev header or a verified JWT from the auth provider. Plus signed links."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from functools import lru_cache

import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from emotext.db import User, get_db, get_or_create_user
from emotext.settings import get_settings


@lru_cache
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, cache_keys=True)


def verify_token(token: str) -> dict:
    """Return the token's claims, or raise HTTPException(401)."""
    s = get_settings()
    options = {"require": ["sub", "exp"], "verify_aud": bool(s.auth_audience)}
    try:
        if s.auth_jwks_url:
            key = _jwks_client(s.auth_jwks_url).get_signing_key_from_jwt(token).key
            algorithms = ["RS256", "ES256"]
        elif s.auth_jwt_secret:
            key, algorithms = s.auth_jwt_secret, ["HS256"]
        else:
            raise HTTPException(500, "auth_mode=jwt needs EMOTEXT_AUTH_JWKS_URL or "
                                     "EMOTEXT_AUTH_JWT_SECRET")
        return jwt.decode(token, key, algorithms=algorithms, audience=s.auth_audience,
                          issuer=s.auth_issuer, options=options)
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"Invalid token: {exc}") from exc


def current_user(
    authorization: str | None = Header(None),
    x_dev_user: str | None = Header(None),
    db: Session = Depends(get_db),
) -> User:
    s = get_settings()
    if s.auth_mode == "dev":
        email = (x_dev_user or "dev@localhost").strip().lower()
        return get_or_create_user(db, f"dev:{email}", email)
    if s.auth_mode != "jwt":
        raise HTTPException(500, f"Unknown auth_mode '{s.auth_mode}'")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    claims = verify_token(authorization.split(" ", 1)[1].strip())
    return get_or_create_user(db, claims["sub"], claims.get("email"))


# -- signed links ------------------------------------------------------------------------------
# <audio src> and download links cannot carry an Authorization header, so file URLs carry a
# short-lived HMAC signature over the path instead.


def _sig(path: str, expires: int) -> str:
    mac = hmac.new(get_settings().secret_key.encode(), f"{path}|{expires}".encode(),
                   hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac[:18]).decode().rstrip("=")


def sign_path(path: str, ttl_s: int = 6 * 3600) -> str:
    expires = int(time.time()) + ttl_s
    return f"{path}?expires={expires}&sig={_sig(path, expires)}"


def check_signature(path: str, expires: int, sig: str) -> bool:
    return expires >= time.time() and hmac.compare_digest(_sig(path, expires), sig)
