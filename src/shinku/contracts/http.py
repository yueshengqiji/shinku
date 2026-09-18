"""Shinku's JSON response policy for its private HTTP surface.

Each Shinku host serves live process state, not cacheable documents, so every
JSON reply leaves with an explicit cache directive attached.  ``no-store`` is
the default; a caller may override it, and may add headers of its own, but may
not leave the directive off entirely.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi.responses import JSONResponse

__all__ = ["json_response"]

_CACHE_CONTROL_HEADER = "Cache-Control"


def json_response(
    payload: Any,
    status_code: int = 200,
    *,
    cache_control: str = "no-store",
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Render ``payload`` as JSON with Shinku's cache policy applied.

    ``status_code`` and ``cache_control`` are coerced with ``int`` and ``str``
    so callers may pass values straight out of configuration.  Caller-supplied
    ``headers`` are merged on top of the directive, which means an explicit
    ``Cache-Control`` entry there wins.
    """
    response_headers: dict[str, str] = {_CACHE_CONTROL_HEADER: str(cache_control)}
    if headers:
        response_headers.update({str(name): str(value) for name, value in headers.items()})
    return JSONResponse(payload, status_code=int(status_code), headers=response_headers)
