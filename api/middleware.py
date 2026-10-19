"""Bound request memory, per-client traffic and privacy-preserving request logs."""

import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("phishlens.api")


class RequestGuard:
    """One event-loop-owned, bounded fixed-window limiter per application process."""

    def __init__(
        self,
        app: ASGIApp,
        limit: int = 30,
        window: float = 60,
        max_bytes: int = 1_000_000,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bound per-process client state and request bytes before JSON parsing."""
        self.app = app
        self.limit = limit
        self.window = window
        self.max_bytes = max_bytes
        self.max_clients = max_clients
        self.clock = clock
        self.clients: OrderedDict[str, tuple[float, int]] = OrderedDict()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Guard HTTP requests while forwarding other ASGI events unchanged."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        started = self.clock()
        status = 500

        async def wrapped_send(message: Message) -> None:
            """Attach a traceable ID and prevent caching sensitive prediction results."""
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        async def reject(code: int, detail: str, headers: dict[str, str] | None = None) -> None:
            """Return guard failures with the same privacy headers as normal responses."""
            await JSONResponse({"detail": detail}, code, headers=headers)(
                scope, receive, wrapped_send
            )

        try:
            if scope["path"].startswith("/predict/"):
                client = (scope.get("client") or ("unknown", 0))[0]
                began, count = self.clients.pop(client, (started, 0))
                if started - began >= self.window:
                    began, count = started, 0
                self.clients[client] = (began, count + 1)
                if len(self.clients) > self.max_clients:
                    self.clients.popitem(last=False)
                if count >= self.limit:
                    await reject(
                        429,
                        "Rate limit exceeded",
                        {"Retry-After": str(max(1, math.ceil(self.window - (started - began))))},
                    )
                    return
            buffered = bytearray()
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body = message.get("body", b"")
                size += len(body)
                if size > self.max_bytes:
                    await reject(413, "Request body exceeds byte limit")
                    return
                buffered.extend(body)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay() -> Message:
                """Deliver the bounded request body once to downstream validation."""
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(buffered), "more_body": False}

            await self.app(scope, replay, wrapped_send)
        finally:
            route = scope.get("route")
            logger.info(
                json.dumps(
                    {
                        "event": "request",
                        "request_id": request_id,
                        "method": scope["method"],
                        "route": getattr(route, "path", "unmatched"),
                        "status": status,
                        "duration_ms": round((self.clock() - started) * 1000, 2),
                    }
                )
            )
