"""限定工作区的搜索、分行读取和预览后修改，复用 ToolRegistry 执行边界。

宿主提供固定根目录与实时授权回调；读取兼容根内绝对路径，模型不能选择授权根。
修改仍使用相对路径及原预览凭据，不借读取兼容扩大写入协议。
文件版本是原始字节 SHA256，读取回执不把分页冒充全文。预览计划只存在于
本 handler，绑定请求作用域；原文不写入全局缓存。替换使用目录 fd、拒绝符号
链接，并在落盘前复查版本。锁协调本进程；外部编辑器应在预览后重新确认，
不能把普通文件系统的版本检查宣传为对任意外部写入者的跨进程事务。
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import threading
import time
from types import SimpleNamespace
import uuid


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def scope_key(context):
    scope = context.get('agent_request_scope') if isinstance(context, dict) else None
    return (str(getattr(scope, 'requester', '')), str(getattr(scope, 'session', '')),
            str(getattr(scope, 'primary_message_id', '')))


def tool_result(name, data, *, error=''):
    return {'tool_type': name, 'followup_context': json.dumps(data, ensure_ascii=False),
            'state_updates': {**data, **({'tool_error': True, 'error_code': error} if error else {})},
            **({'status': 'error', 'error': error} if error else {})}


class WorkspaceAccess:
    """路径与授权边界由宿主持有，所有操作（含预览）每次重新检查。"""
    max_bytes = 4 * 1024 * 1024

    def __init__(self, root, *, authorize):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir() or not callable(authorize):
            raise ValueError('invalid_workspace_host_configuration')
        self.authorize = authorize
        self.lock = threading.RLock()
        self.previews = OrderedDict()

    def check(self, context, write=False):
        if not self.authorize(context, write) or not all(scope_key(context)):
            raise PermissionError('workspace_not_authorized')

    def parts(self, path):
        if not isinstance(path, str) or '\\' in path or '\x00' in path:
            raise ValueError('invalid_workspace_path')
        p = PurePosixPath(path)
        if p.is_absolute() or not p.parts or any(x in {'.', '..', '.git'} for x in p.parts):
            raise ValueError('workspace_path_outside_scope')
        return p.parts

    def relative_read_path(self, path):
        """只把固定根内的绝对读取路径转为相对路径，既不寻址也不跟随链接。"""
        if not isinstance(path, str) or '\\' in path or '\x00' in path:
            raise ValueError('invalid_workspace_path')
        candidate = PurePosixPath(path)
        if candidate.is_absolute():
            try:
                # 按路径段比较，拒绝同名前缀的兄弟目录；不能用 resolve 先走一遍
                # 不可信路径。后续仍由原 openat/O_NOFOLLOW 边界访问和核验。
                candidate = candidate.relative_to(self.root.as_posix())
            except ValueError:
                raise ValueError('workspace_path_outside_scope') from None
        return PurePosixPath(*self.parts(candidate.as_posix())).as_posix()

    @contextmanager
    def parent(self, path):
        # 逐层 openat 且禁跟随链接，防止检查后目录被换成越界链接。
        parts = self.parts(path)
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = nxt
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def read(self, path):
        with self.parent(path) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > self.max_bytes:
                    raise ValueError('unsupported_or_oversized_file')
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    raw = stream.read(self.max_bytes + 1)
                after = os.fstat(fd)
                if len(raw) > self.max_bytes or (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    raise ValueError('file_changed_during_read')
                return raw, after
            finally:
                os.close(fd)

    def entries(self, parts):
        """使用目录 fd 枚举，路径被换成链接时失败，连越界文件名也不暴露。"""
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = nxt
            rows = []
            with os.scandir(fd) as stream:
                for entry in stream:
                    rows.append((entry.name, entry.is_dir(follow_symlinks=False), entry.is_file(follow_symlinks=False)))
            return sorted(rows)
        finally:
            os.close(fd)

    def replace(self, path, expected, raw, context):
        self.check(context, True)
        old, oldstat = self.read(path)
        if digest(old) != expected:
            raise ValueError('file_version_conflict')
        with self.parent(path) as (parent, name):
            temp = '.agent-patch-' + uuid.uuid4().hex
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, 'wb', closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(fd)
                os.fchmod(fd, stat.S_IMODE(oldstat.st_mode) & 0o777)
                current, currentstat = self.read(path)
                if digest(current) != expected or currentstat.st_ino != oldstat.st_ino:
                    raise ValueError('file_version_conflict')
                self.check(context, True)
                os.replace(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(fd)
                try:
                    os.unlink(temp, dir_fd=parent)
                except FileNotFoundError:
                    pass


class WorkspaceTool:
    """四个明确权限的 handler，共用受控路径访问与结果协议。"""
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_observation_receipt_fields = ('observation', 'path', 'sha256', 'preview_id')

    def __init__(self, access, operation):
        self.access, self.operation = access, operation
        from .observation_receipts import ObservationReceipts
        self.observations = ObservationReceipts()
        self.tool_type = 'workspace_' + operation
        if operation == 'apply_patch':
            self.agent_permission_class, self.agent_progress_class = 'change', 'change'
            self.agent_completion_effects = ('write',)
        elif operation == 'read_lines':
            self.agent_completion_effects = ('read',)
            # 只有覆盖全文且当前字节仍匹配的结果能进入本地分析账本；分页结果
            # 仍由 read_complete=False 拦住，不能借新接口绕过必需原文读取。
            self.agent_local_answer_source = True
            self.agent_local_answer_path_field = 'source_path'
        self.agent_completion_target_arguments = ('path',)

    def build_prompt_instruction(self):
        return {'read_lines': '按行读取工作区 UTF-8 文本，返回行号、原始字节版本与是否全文；不跟随链接。'
                    'path可用工作区相对路径或授权工作区内的绝对路径；回执统一返回相对path与绝对source_path。'
                    '精确引用使用实际start_line与偏移核对；长文可先用workspace_search定位，再读取相邻行，勿猜行号。',
                'search': '完整搜索工作区内指定path子目录，默认忽略大小写；mode=path查文件和目录名，'
                          'content查正文。limit只限制本页返回数量，不限制扫描；按next_offset和snapshot翻页。'
                          '搜索范围、排除项和读取错误随结果返回；无命中只说明该范围，不证明整个项目不存在。',
                'preview_patch': '用文件 SHA256 和唯一原文片段生成局部替换差异及预览凭据，不修改文件。',
                'apply_patch': '应用本请求已预览的局部补丁；需原路径、SHA256 和预览凭据，版本变化即拒绝。'}[self.operation]

    def tool_metadata(self):
        props = {
            'read_lines': {'path': {'type': 'string', 'description': '工作区相对路径，或授权工作区内的绝对路径；禁止越界和符号链接。'},
                           'start_line': {'type': 'integer', 'minimum': 1},
                           'line_count': {'type': 'integer', 'minimum': 1, 'maximum': 400}},
            'search': {'query': {'type': 'string'}, 'glob': {'type': 'string'},
                       'path': {'type': 'string', 'description': '搜索起点，默认工作区根；可用根内相对或绝对目录。'},
                       'ignore_case': {'type': 'boolean', 'description': '默认true；false为区分大小写的字面匹配。'},
                       'mode': {'type': 'string', 'enum': ['content', 'path']},
                       'offset': {'type': 'integer', 'minimum': 0}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                       'snapshot': {'type': 'string'}},
            'preview_patch': {'path': {'type': 'string'}, 'expected_sha256': {'type': 'string'},
                              'edits': {'type': 'array', 'minItems': 1, 'maxItems': 50, 'items': {'type': 'object',
                                  'properties': {'old': {'type': 'string'}, 'new': {'type': 'string'}},
                                  'required': ['old', 'new'], 'additionalProperties': False}}},
            'apply_patch': {'path': {'type': 'string'}, 'expected_sha256': {'type': 'string'}, 'preview_id': {'type': 'string'}},
        }[self.operation]
        required = {'read_lines': ['path'], 'search': ['query'],
                    'preview_patch': ['path', 'expected_sha256', 'edits'],
                    'apply_patch': ['path', 'expected_sha256', 'preview_id']}[self.operation]
        return SimpleNamespace(input_schema={'type': 'object', 'properties': props, 'required': required,
                                             'additionalProperties': False})

    def normalize_call(self, call):
        if not isinstance(call, dict):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not k.startswith('_tool_')}
        schema = self.tool_metadata().input_schema
        if set(args) - schema['properties'].keys() or any(k not in args for k in schema['required']):
            return None
        for key, value in args.items():
            kind = schema['properties'][key]['type']
            if kind == 'string' and (not isinstance(value, str) or len(value) > 4096):
                return None
            if kind == 'boolean' and type(value) is not bool:
                return None
            maximum = schema['properties'][key].get('maximum',
                float('inf') if self.operation == 'search' and key == 'offset' else 100000)
            if kind == 'integer' and (type(value) is not int or value < schema['properties'][key].get('minimum', 0)
                                      or value > maximum):
                return None
        return args

    def execute(self, *, call, context):
        from .workspace_search import SearchAuthorizationLost
        args = self.normalize_call(call)
        try:
            if args is None:
                raise ValueError('invalid_workspace_arguments')
            self.access.check(context, self.operation == 'apply_patch')
            with self.access.lock:
                data = getattr(self, '_' + self.operation)(args, context)
                if self.operation in {'search', 'preview_patch'}:
                    data['observation'] = self.observations.issue(data, tool=self.tool_type, context=context)
            return tool_result(self.tool_type, data)
        except SearchAuthorizationLost:
            return tool_result(self.tool_type, {'status': 'error', 'code': 'workspace_not_authorized'},
                               error='workspace_not_authorized')
        except (ValueError, OSError, UnicodeError) as exc:
            # 不把异常路径、权限配置或原文泄露到跨作用域错误中。
            code = str(exc) if isinstance(exc, ValueError) else 'workspace_access_failed'
            return tool_result(self.tool_type, {'status': 'error', 'code': code}, error=code)

    def agent_verify_result(self, *, receipt, context):
        """完成门重读实际文件；预览永远不能证明 write，分页不能证明全文 read。"""
        try:
            self.access.check(context, self.operation == 'apply_patch')
            if self.operation == 'search':
                return {'result_valid': self.observations.verify(receipt, tool=self.tool_type, context=context)}
            if self.operation == 'preview_patch' and not self.observations.verify(receipt, tool=self.tool_type, context=context):
                return {'result_valid': False}
            if (self.operation == 'read_lines' and receipt.get('source_path') is not None
                    and receipt['source_path'] != str(self.access.root / receipt['path'])):
                return {'result_valid': False}
            raw, _ = self.access.read(receipt['path'])
            valid = digest(raw) == receipt.get('sha256')
            if self.operation == 'read_lines':
                valid = valid and receipt.get('read_complete') is True
            elif self.operation == 'apply_patch':
                # 循环只保留公共紧凑回执字段；applied 随独立 artifact 回执保留。
                applied = receipt.get('applied') is True or any(
                    isinstance(r, dict) and r.get('applied') is True and r.get('path') == receipt.get('path')
                    and r.get('sha256') == receipt.get('sha256') for r in receipt.get('artifact_receipts', []))
                valid = valid and applied
            elif self.operation == 'preview_patch':
                preview = self.access.previews.get(receipt.get('preview_id'))
                valid = valid and bool(preview and preview[0] == scope_key(context) and preview[4] >= time.monotonic())
            result = {'result_valid': valid, 'sha256': digest(raw), 'bytes': len(raw),
                      'file_text': raw.decode('utf-8') if valid else ''}
            if 'artifact_receipts' in receipt:
                result['artifact_receipts'] = [{**r, **result} for r in receipt['artifact_receipts']]
            return result
        except (KeyError, ValueError, OSError, UnicodeError):
            return {'result_valid': False}

    def _read_lines(self, args, context):
        # 原生调用参数保留原貌；规范路径只进入真实读取回执和原完成门。
        path = self.access.relative_read_path(args['path'])
        raw, _ = self.access.read(path)
        lines = raw.decode('utf-8').splitlines(keepends=True)
        start = args.get('start_line', 1) - 1
        if start > len(lines):
            raise ValueError('line_range_outside_file')
        selected = lines[start:start + args.get('line_count', 120)]
        text = ''.join(selected)
        if len(text) > 40000:
            raise ValueError('line_range_too_large_reduce_count')
        return {'path': path, 'source_path': str(self.access.root / path),
                'sha256': digest(raw), 'start_line': start + 1,
                'end_line': start + len(selected), 'total_lines': len(lines), 'text': text,
                'read_complete': start == 0 and len(selected) == len(lines), 'bytes': len(raw),
                'verified_effects': ['read'] if start == 0 and len(selected) == len(lines) else [],
                'completion_targets': [path, str(self.access.root / path)]}

    def _search(self, args, context):
        # 搜索使用流式读取，不继承原文读取/补丁大小限制；路径授权仍复用本适配器。
        from .workspace_search import search
        return search(self.access, args, context)

    def _preview_patch(self, args, context):
        raw, _ = self.access.read(args['path'])
        version = digest(raw)
        if args['expected_sha256'] != version:
            raise ValueError('file_version_conflict')
        text = raw.decode('utf-8')
        edits = args['edits']
        if not isinstance(edits, list) or not 1 <= len(edits) <= 50:
            raise ValueError('invalid_patch_edits')
        updated = text
        for edit in edits:
            if (not isinstance(edit, dict) or set(edit) != {'old', 'new'}
                    or not all(isinstance(v, str) for v in edit.values()) or not edit['old']):
                raise ValueError('invalid_patch_edit')
            if updated.count(edit['old']) != 1:
                raise ValueError('patch_original_not_unique')
            updated = updated.replace(edit['old'], edit['new'], 1)
        encoded = updated.encode('utf-8')
        if len(encoded) > self.access.max_bytes or text == updated:
            raise ValueError('patch_oversized_or_unchanged')
        diff = ''.join(difflib.unified_diff(text.splitlines(keepends=True), updated.splitlines(keepends=True),
                                          fromfile=args['path'], tofile=args['path']))
        if len(diff) > 40000:
            raise ValueError('patch_diff_too_large')
        identifier = uuid.uuid4().hex
        self.access.previews[identifier] = (scope_key(context), args['path'], version, encoded, time.monotonic() + 300)
        # 受限的进程内待应用数据，不保留跨会话授权；重启后须重新预览。
        while len(self.access.previews) > 16:
            self.access.previews.popitem(last=False)
        return {'path': args['path'], 'sha256': version, 'new_sha256': digest(encoded),
                'preview_id': identifier, 'diff': diff, 'applied': False}

    def _apply_patch(self, args, context):
        preview = self.access.previews.get(args['preview_id'])
        if not preview or preview[:3] != (scope_key(context), args['path'], args['expected_sha256']) or preview[4] < time.monotonic():
            raise ValueError('patch_preview_expired_or_wrong_scope')
        self.access.replace(args['path'], args['expected_sha256'], preview[3], context)
        del self.access.previews[args['preview_id']]
        current, _ = self.access.read(args['path'])
        if digest(current) != digest(preview[3]):
            raise ValueError('file_changed_after_patch')
        receipt = {'path': args['path'], 'previous_sha256': args['expected_sha256'], 'sha256': digest(current),
                   'bytes': len(current), 'applied': True, 'result_valid': True, 'verified_effects': ['write'],
                   'completion_targets': [args['path'], str(self.access.root / args['path'])]}
        return {**receipt, 'changed': True, 'artifact_receipts': [receipt]}


def build_workspace_tools(root, *, authorize):
    """构建不执行副作用的可选装配；未注册、未获授权时没有任何新能力。"""
    access = WorkspaceAccess(root, authorize=authorize)
    return tuple(WorkspaceTool(access, op) for op in ('search', 'read_lines', 'preview_patch', 'apply_patch'))
