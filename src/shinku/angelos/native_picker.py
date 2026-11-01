"""本机桌面选取路径；只返回选择，不写工作区、不授予 Agent 文件权限。

路由持有一个实例，等待窗口时不占工作台锁。超时、连接断开或服务取消时
回收子进程；固定 Zenity 参数，不接受浏览器传入命令或任意启动目录。
"""
import asyncio
import os
from pathlib import Path
import shutil


class NativePicker:
    def __init__(self, timeout=150):
        self.timeout = timeout
        self.busy = False

    @property
    def available(self):
        return bool(shutil.which('zenity') and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))

    async def choose(self, kind, directory, disconnected):
        if kind not in {'file', 'directory'}:
            raise ValueError('invalid_picker_kind')
        if self.busy:
            raise ValueError('native_picker_busy')
        if not self.available:
            raise ValueError('native_picker_unavailable')
        start = Path(directory) if directory else Path.home()
        if kind == 'directory' and not start.is_dir():
            start = Path.home()
        if not start.is_absolute() or not start.is_dir():
            raise ValueError('native_picker_directory_missing')
        # 同一事件循环内先占位再 await，避免双击或两个标签页弹出多个窗口。
        self.busy = True
        process = pending = None
        try:
            args = [shutil.which('zenity'), '--file-selection',
                    '--title=Angelos — '+('选择文件' if kind == 'file' else '选择工作区目录'),
                    '--filename='+str(start).rstrip('/')+'/']
            if kind == 'directory':
                args.append('--directory')
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            pending = asyncio.create_task(process.communicate())
            deadline = asyncio.get_running_loop().time()+self.timeout
            while not pending.done():
                if await disconnected():
                    raise ValueError('native_picker_disconnected')
                if asyncio.get_running_loop().time() >= deadline:
                    raise ValueError('native_picker_timeout')
                await asyncio.wait({pending}, timeout=.1)
            output, _ = pending.result()
            if process.returncode == 1:
                return {'cancelled': True}
            if process.returncode != 0:
                raise ValueError('native_picker_failed')
            # Zenity 只追加一个换行，不能 strip 掉合法文件名首尾空格。
            path = os.fsdecode(output.removesuffix(b'\n'))
            selected = Path(path)
            if not selected.is_absolute() or not (selected.is_file() if kind == 'file' else selected.is_dir()):
                raise ValueError('native_picker_invalid_selection')
            return {'cancelled': False, 'path': path}
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            if pending is not None:
                await pending
            self.busy = False
