"""工作区完整搜索：流式扫描、分页交付，复用WorkspaceAccess的openat授权边界。

没有扫描时间、文件数、总字节或匹配数截止。只在内存保留当前页和少量诊断；
翻页重新扫描并核对摘要，来源变化须重新开始，不把过时分页拼成一次完整结果。
每个目录/文件/读取块重新核对宿主授权，取消或撤权立即传播，不当成可跳过文件。
不读取私用、隐藏目录或跟随符号链接；这些是明确的搜索范围，不是整台机器的结论。
"""
import codecs
from collections import Counter
import fnmatch
import hashlib
import json
import os
import re
import stat


EXCLUDED = frozenset({'.git', '.venv', 'node_modules', '__pycache__', 'private', 'logs', 'data', 'tmp'})
BREAKS = re.compile(r'\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]')
CHUNK = 64 * 1024


class SearchAuthorizationLost(RuntimeError):
    """与单文件读取错误分开，避免吞掉取消或撤权。"""


def _file(access, path, needle, ignore_case, offset, limit, check):
    """大文件和超长单行按块匹配；只保留本页片段，SHA256覆盖原始文件全部字节。"""
    count, rows, line_no, length, prefix, tail, matched = 0, [], 1, 0, '', '', False

    def segment(text):
        nonlocal length, prefix, tail, matched
        length += len(text)
        prefix += text[:max(0, 400 - len(prefix))]
        folded = text.casefold() if ignore_case else text
        combined = tail + folded
        matched = matched or needle in combined
        tail = combined[-(len(needle) - 1):] if len(needle) > 1 else ''

    def end_line():
        nonlocal count, line_no, length, prefix, tail, matched
        if matched:
            if offset <= count < offset + limit:
                rows.append({'path': path, 'line': line_no, 'text': prefix,
                             'omitted_chars': max(0, length - 400)})
            count += 1
        line_no += 1
        length, prefix, tail, matched = 0, '', '', False

    with access.parent(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('unsupported_file')
            sha, total, decoder, previous_cr = hashlib.sha256(), 0, codecs.getincrementaldecoder('utf-8')(), False
            while True:
                check()
                raw = os.read(fd, CHUNK)
                sha.update(raw)
                total += len(raw)
                text = decoder.decode(raw, final=not raw)
                if '\x00' in text:
                    raise ValueError('binary_file')
                # CRLF可以跨块；行号与read_lines使用的splitlines保持一致。
                if text:
                    if previous_cr and text.startswith('\n'):
                        text = text[1:]
                    previous_cr = text.endswith('\r')
                    pos = 0
                    for match in BREAKS.finditer(text):
                        segment(text[pos:match.start()])
                        end_line()
                        pos = match.end()
                    segment(text[pos:])
                if not raw:
                    break
            if length:
                end_line()
            after = os.fstat(fd)
            if (before.st_mtime_ns, before.st_ctime_ns, before.st_size) != (after.st_mtime_ns, after.st_ctime_ns, after.st_size):
                raise ValueError('file_changed_during_read')
            for row in rows:
                row['sha256'] = sha.hexdigest()
            return count, rows, sha.hexdigest(), total
        finally:
            os.close(fd)


def search(access, args, context):
    """完成所选范围后返回一页；找不到与未能完整读取分别报告。"""
    def check():
        try:
            access.check(context)
        except Exception as exc:
            raise SearchAuthorizationLost() from exc

    check()
    query, mode = args['query'], args.get('mode', 'content')
    if not query or mode not in {'content', 'path'}:
        raise ValueError('invalid_search_query')
    start = args.get('path', '.')
    if start == str(access.root):
        start = '.'
    parts = () if start == '.' else access.parts(access.relative_read_path(start))
    if any(p in EXCLUDED or p.startswith('.') for p in parts):
        raise ValueError('search_path_excluded')
    ignore_case, pattern = args.get('ignore_case', True), args.get('glob', '*')
    needle = query.casefold() if ignore_case else query
    offset, limit = args.get('offset', 0), args.get('limit', 50)
    scope = {'path': '/'.join(parts) or '.', 'mode': mode, 'glob': pattern, 'ignore_case': ignore_case,
             'excluded_directories': sorted(EXCLUDED), 'hidden_paths_excluded': True, 'follow_symlinks': False}
    sha = hashlib.sha256()

    def record(value):
        sha.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode() + b'\n')

    record([scope, query])
    rows, total, scanned, files = [], 0, 0, 0
    skipped, examples = Counter(), []

    def skip(path, reason):
        skipped[reason] += 1
        record(['skipped', path, reason])
        if len(examples) < 5:
            examples.append({'path': path, 'reason': reason})

    # 明确起点不存在/不可读应报错，不能以空搜索结果冒充成功。
    stack = [iter([(parts, *entry) for entry in access.entries(parts)])]
    while stack:
        check()
        entry = next(stack[-1], None)
        if entry is None:
            stack.pop()
            continue
        directory, name, is_dir, is_file = entry
        if name in EXCLUDED or name.startswith('.') or not (is_dir or is_file):
            continue
        components = (*directory, name)
        path = '/'.join(components)
        if mode == 'path' and fnmatch.fnmatch(path, pattern):
            record(['path', path, is_dir])
            if needle in (path.casefold() if ignore_case else path):
                if offset <= total < offset + limit:
                    rows.append({'path': path, 'kind': 'directory' if is_dir else 'file'})
                total += 1
        if is_dir:
            try:
                entries = access.entries(components)
            except OSError:
                skip(path, 'directory_unreadable')
                continue
            stack.append(iter([(components, *entry) for entry in entries]))
            continue
        if mode == 'path' or not fnmatch.fnmatch(path, pattern):
            continue
        files += 1
        try:
            count, page, version, size = _file(access, path, needle, ignore_case,
                max(0, offset - total), min(limit, max(0, offset + limit - total)), check)
        except UnicodeError:
            skip(path, 'non_utf8')
            continue
        except ValueError as exc:
            skip(path, str(exc))
            continue
        except OSError:
            skip(path, 'file_unreadable')
            continue
        record(['file', path, version])
        total += count
        rows.extend(page)
        scanned += size
    check()
    snapshot = sha.hexdigest()
    if offset and args.get('snapshot') != snapshot:
        raise ValueError('search_snapshot_changed_restart')
    complete = not skipped
    return {'matches': rows, 'total_matches': total, 'snapshot': snapshot,
            'next_offset': offset + len(rows) if offset + len(rows) < total else None,
            'scan_complete': complete, 'scan_bounded': False,
            'search_scope': scope, 'excluded_directories': sorted(EXCLUDED), 'hidden_paths_excluded': True,
            'skipped_counts': dict(skipped), 'skipped_paths': [e['path'] for e in examples],
            'skipped_examples': examples, 'scanned_bytes': scanned, 'scanned_files': files,
            'search_status': 'complete' if complete else 'completed_with_skips',
            'guidance': '搜索已结束；结论仅限search_scope。' if complete else
                        '搜索已结束，但存在未读取文件；不能以空结果断言目标不存在。可定位目录后读取相关文件。',
            'read_complete': False}
