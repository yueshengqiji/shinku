"""Optional ReportLab font selection with explicit overrides and glyph checks."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import threading

_LOCK = threading.RLock()
_REGISTERED = {}


class PDFFontError(RuntimeError):
    pass


def _candidates(role, env):
    override = env.get(f"SHINKU_PDF_{role}_FONT", "").strip()
    if override:
        return [Path(override).expanduser()]
    if role == "CJK":
        return [Path(p) for p in (
            "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/TTF/uming.ttc", "/usr/share/fonts/TTF/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        )]
    return [Path(p) for p in (
        "C:/Windows/Fonts/seguisym.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    )]


@lru_cache(maxsize=16)
def _load_font(name, path, index, size, modified):
    from reportlab.pdfbase.ttfonts import TTFont
    return TTFont(name, path, subfontIndex=index)


def register_pdf_fonts(environ=None):
    """Select embeddable fonts; incompatible overrides fail instead of falling back.

    System candidates are tried in order. Existing CFF-only collections may be
    skipped; a font must support its role before registration. No downloads or
    substitution for an unsupported character happen during PDF generation.
    """
    from reportlab.pdfbase import pdfmetrics
    env = os.environ if environ is None else environ
    roles = (("CJK", "ShinkuTranslation", "中文译文"), ("SYMBOL", "ShinkuSymbols", "∼✓✗∗"))
    selected = []
    with _LOCK:
        for role, name, required in roles:
            try:
                index = int(env.get(f"SHINKU_PDF_{role}_FONT_INDEX", "0"))
                if not 0 <= index <= 31:
                    raise ValueError
            except (TypeError, ValueError):
                raise PDFFontError(f"pdf_font_index_invalid:{role.lower()}") from None
            for path in _candidates(role, env):
                try:
                    stat = path.stat()
                    signature = (str(path.resolve()), index, stat.st_size, stat.st_mtime_ns)
                    font = _load_font(name, *signature)
                    if any(ord(char) not in font.face.charWidths for char in required):
                        continue
                except (OSError, ValueError, KeyError):
                    continue
                except Exception as exc:
                    # ReportLab rejects unsupported TTC/CFF outline variants.
                    from reportlab.pdfbase.ttfonts import TTFError
                    if not isinstance(exc, TTFError):
                        raise
                    continue
                selected.append((name, font, signature))
                break
            else:
                raise PDFFontError(f"pdf_font_unavailable:{role.lower()}")
        # Resolve both roles first so an invalid override cannot half-register.
        for name, font, signature in selected:
            if _REGISTERED.get(name) != signature:
                pdfmetrics.registerFont(font)
                _REGISTERED[name] = signature
        return {name: font.face.charWidths for name, font, _ in selected}
