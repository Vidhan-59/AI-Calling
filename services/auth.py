"""Shared FastAPI dependency guarding the dialer admin surface
(routers/campaigns.py) and the call analytics surface (routers/analytics.py)
- both expose real contact phone numbers, call transcripts, and controls
like the kill switch, none of which should be reachable by anyone outside
a trusted operator.

Deliberately not applied to routers/campaigns.py's Twilio call-status
webhook (/dialer/call-status) - Twilio cannot send our custom header, that
endpoint is protected by knowing an unguessable CallSid instead.

Outside development, an unset ADMIN_API_KEY is refused at server boot (see
main.py's startup check) rather than just logged - "forgot to set it" must
not be able to silently ship an unauthenticated admin surface to production.
The warn-and-allow path below only remains reachable when
ENVIRONMENT=development, where main.py's boot check doesn't apply.
"""

import logging

from fastapi import Header, HTTPException

from config import settings

logger = logging.getLogger("auth")

_warned_unset = False


async def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    global _warned_unset
    if not settings.admin_api_key:
        if not _warned_unset:
            logger.warning(
                "ADMIN_API_KEY is not set - admin/analytics endpoints are running with NO "
                "authentication. Set ADMIN_API_KEY in .env before exposing this server "
                "outside a trusted network."
            )
            _warned_unset = True
        return
    if x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
