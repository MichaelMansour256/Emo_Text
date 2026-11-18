"""Error reporting. Sentry starts only when EMOTEXT_SENTRY_DSN is set."""

from __future__ import annotations

import logging

from emotext import __version__
from emotext.settings import get_settings

log = logging.getLogger(__name__)


def init_sentry(component: str) -> bool:
    s = get_settings()
    if not s.sentry_dsn:
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=s.sentry_dsn,
        environment=s.environment,
        release=f"emotext@{__version__}",
        traces_sample_rate=0.05,
        send_default_pii=False,  # uploads are people's books: never attach request bodies
        server_name=component,
    )
    log.info("Sentry enabled for %s (%s)", component, s.environment)
    return True
