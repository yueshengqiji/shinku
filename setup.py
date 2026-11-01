"""把两套正式前端打入 wheel，源码仍只维护 frontend 下的一份资源。"""
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        for frontend in ("control-center", "angelos"):
            source = root / "frontend" / frontend
            destination = Path(self.build_lib) / "shinku" / "_frontend" / frontend
            self.mkpath(str(destination))
            names = ("index.html", "app.css", "app.js")
            if frontend == "angelos":
                names += ("context.js", "layout.js", "activity.js", "log-view.js", "sessions.js", "usage.js", "workspace.js", "editor.js", "editor/THIRD_PARTY_LICENSES.txt")
            for name in names:
                asset = source / name
                if not asset.is_file():
                    raise FileNotFoundError(f"required_frontend_asset_missing:{frontend}/{name}")
                self.mkpath(str((destination / name).parent))
                self.copy_file(str(asset), str(destination / name))


setup(cmdclass={"build_py": BuildPy})
