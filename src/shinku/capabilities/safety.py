"""Bounds that every capability a provider declares has to stay inside.

A manifest is an outside file, so any value in it that could reach a network
call or a subprocess is filtered through one of these first.  The module
imports nothing but :mod:`re` on purpose: loaders pull it in early, and a table
of constants should never be the reason an import fails.
"""

from __future__ import annotations

import re

__all__ = ["LOOPBACK_HOSTS", "MCP_SECRET_MARKERS", "MCP_SAFE_TYPE_RE"]


# Hosts a provider may be reached at when its manifest asks for loopback-only
# traffic.  Compared against ``urlparse(...).hostname`` after lower-casing, so
# entries stay lower-case here.
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Secret *names* are declared in a manifest; secret *values* are not.  A name
# that is one of these words carries no information a provider could not guess,
# so it is rejected as a placeholder rather than a real key.
MCP_SECRET_MARKERS = ("api_key", "password", "secret", "token")

# Provider type strings end up in log lines and adapter lookup keys, so they are
# restricted to a conservative alphabet instead of being accepted verbatim.
MCP_SAFE_TYPE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
