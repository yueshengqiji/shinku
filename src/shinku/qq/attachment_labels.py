"""附件的类型占位说明；仅描述收到的引用，不代表正文已读取。"""
from collections.abc import Mapping


def attachment_label(attachments, *, quoted=False):
    """未知类型保持一般附件，避免把文件/音频误导成待加载图片。"""
    names = {"image": "图片", "file": "文件", "audio": "音频", "video": "视频"}
    kinds = dict.fromkeys(str(row.get("kind", "")) for row in attachments if isinstance(row, Mapping))
    return "".join(f"[{'引用' if quoted else ''}{names.get(kind, '')}附件]" for kind in kinds)
