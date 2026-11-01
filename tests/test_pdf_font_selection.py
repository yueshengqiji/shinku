"""Font overrides must retain Chinese and symbols, including missing-glyph errors."""
import pytest

pytest.importorskip("reportlab")
from shinku.pdf_fonts import PDFFontError, register_pdf_fonts


def test_system_fonts_cover_chinese_and_required_symbols():
    fonts = register_pdf_fonts({})
    assert all(ord(c) in fonts["ShinkuTranslation"] for c in "中文全文译文")
    assert all(any(ord(c) in widths for widths in fonts.values()) for c in "中文 10∼20 ✓ ✗ ∗ α β")


@pytest.mark.parametrize("role", ["CJK", "SYMBOL"])
def test_invalid_override_never_reuses_previous_registered_font(role, tmp_path):
    register_pdf_fonts({})
    with pytest.raises(PDFFontError, match=f"pdf_font_unavailable:{role.lower()}"):
        register_pdf_fonts({f"SHINKU_PDF_{role}_FONT": str(tmp_path / "missing.ttf")})


@pytest.mark.parametrize("index", ["not-a-number", "-1", "32"])
def test_invalid_collection_index_is_rejected(index):
    with pytest.raises(PDFFontError, match="pdf_font_index_invalid:cjk"):
        register_pdf_fonts({"SHINKU_PDF_CJK_FONT_INDEX": index})
