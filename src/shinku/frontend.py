"""Locate the control center in an installed package or a source checkout."""
from pathlib import Path


def control_center_root() -> Path:
    package = Path(__file__).resolve().parent
    packaged = package / "_frontend" / "control-center"
    if (packaged / "index.html").is_file():
        return packaged
    return package.parents[1] / "frontend" / "control-center"


def angelos_root() -> Path:
    """独立工作台的资源入口，兼容源码运行与 wheel 安装。"""
    package = Path(__file__).resolve().parent
    packaged = package / "_frontend" / "angelos"
    return packaged if (packaged / "index.html").is_file() else package.parents[1] / "frontend" / "angelos"
