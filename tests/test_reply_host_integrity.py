"""Protect atomic links/inline code and preserve code-block bodies during shaping."""
import pytest

from shinku.qq.reply_host import ReplyHost, clean_voice_text


@pytest.mark.parametrize("url", [
    "https://example.test/search?q=abc&lang=zh",
    "https://example.test/search?q=one!two;three&ok=1",
    "https://example.test/file?signature=abc%3D&expires=1800000000",
])
def test_query_url_is_never_split_at_sentence_punctuation(url):
    pieces = ReplyHost(strip_final_periods=True).shape_text("参考这个地址：" + url + "。然后继续。")
    assert pieces == ["参考这个地址：" + url, "然后继续"]


def test_oversize_url_is_atomic_even_when_it_exceeds_the_soft_prose_target():
    url = "https://example.test/download?signature=" + "a" * 700
    pieces = ReplyHost(max_segment_chars=120).shape_text("下载地址 " + url + " 后面继续说明")
    assert sum(url in piece for piece in pieces) == 1
    assert "".join(pieces).replace(" ", "") == ("下载地址" + url + "后面继续说明")


def test_inline_command_punctuation_is_not_a_sentence_boundary():
    pieces = ReplyHost(strip_final_periods=True).shape_text("执行 `echo one; echo two`。然后继续。")
    assert pieces == ["执行 `echo one; echo two`", "然后继续"]


def test_atomic_url_and_inline_code_are_not_rewritten_by_voice_cleaning():
    text = "https://example.test/——slug `echo ——`。你说呢？"
    assert clean_voice_text(text) == "https://example.test/——slug `echo ——`。"


def body(piece):
    lines = piece.split("\n")
    assert lines[0] == "```sh" and lines[-1] == "```"
    return "\n".join(lines[1:-1])


def test_multiline_code_has_balanced_fences_in_every_piece_and_exact_body():
    raw_body = "#" + "a" * 580 + "\n#" + "b" * 580 + "\necho .\n#" + "c" * 580
    pieces = ReplyHost(strip_final_periods=True, max_segment_chars=600).shape_text("```sh\n" + raw_body + "\n```")
    assert len(pieces) > 1 and all(len(piece) <= 600 for piece in pieces)
    assert "\n".join(body(piece) for piece in pieces) == raw_body
    assert "echo ." in "\n".join(pieces)


def test_code_indentation_and_blank_lines_survive_chunking():
    raw_body = "\n  #" + "a" * 90 + "\n\n  echo .\n\n  #" + "b" * 90 + "\n"
    pieces = ReplyHost(strip_final_periods=True, max_segment_chars=120).shape_text("```sh\n" + raw_body + "\n```")
    assert "\n".join(body(piece) for piece in pieces) == raw_body


def test_indivisible_code_line_stays_intact_with_balanced_fences():
    code = "```python\n" + "x" * 700 + "\n```"
    # 600 is a prose/chunk target, not permission to change an executable line.
    assert ReplyHost(strip_final_periods=True, max_segment_chars=600).shape_text(code) == [code]


def test_short_fenced_code_remains_byte_equal_with_sentence_cleanup_on():
    code = "```sh\necho .\n```"
    assert ReplyHost(strip_final_periods=True).shape_text(code) == [code]


def test_long_prose_still_obeys_segment_target_without_losing_characters():
    text = "普通说明文字" * 100
    pieces = ReplyHost(max_segment_chars=120).shape_text(text)
    assert all(len(piece) <= 120 for piece in pieces)
    assert "".join(pieces) == text
