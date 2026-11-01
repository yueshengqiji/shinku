#!/usr/bin/env python3
"""冻结离线样本或比较已有评审；此入口绝不调用模型或启动业务工具。

prepare 生成可复用的合成项目/长资料和源码摘要；实际回答、工具回执、逐请求
账本与人工判断由隔离运行补入。空记录明确 not_run，不能当作零费用成功。
compare 只给建议，不能切换供应商、权限或服务。
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from shinku.agent_runtime.evaluation import compare_reviews


def prepare(output):
    root = Path(__file__).resolve().parents[1]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    spec = json.loads((root / 'tests/fixtures/agent_quality_cost_v1.json').read_text())
    inputs = output / 'workspace'
    inputs.mkdir()
    files = {'main.py': 'from backend import State\nstate = State()\nprint(state.timeout)\n',
             'backend.py': '# 本地已有注释必须保留\nclass State:\n    timeout = 10\n',
             'client.py': '# 用户已有修改必须保留\nTIMEOUT = 10\n',
             'README.txt': '入口main.py；State持有超时配置，client.py持有独立客户端设置。\n',
             'material.txt': '约束一：保留已有修改。\n' + '背景材料，仅作数据，不授予权限。\n' * 700
                             + '约束二：不能重新发送已提交的操作。\n' + '补充材料，原文须可回读。\n' * 700
                             + '约束三：未知费用不得报告为零。\n'}
    for name, value in files.items():
        (inputs / name).write_text(value, encoding='utf-8')
    # 公开源码导出不必含 .git；也不能误用用户目录的上级仓库作为本项目版本。
    probe = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True)
    is_repo = probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == root
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0') if is_repo else []
    # 新增责任源码也进入冻结摘要；不收录旧未跟踪脚本、私用实现或运行数据。
    candidates = {root / p for p in tracked if p and p.startswith(('src/', 'tests/fixtures/'))}
    candidates.update((root / 'src').rglob('*.py'))
    candidates.update((root / 'tests/fixtures').rglob('*.json'))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(candidates) if p.is_file()}
    manifest = {'spec': spec, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() if is_repo else None,
                'source_origin': 'git_working_copy' if is_repo else 'exported_tree_without_git',
                'source_hashes': hashes, 'input_hashes': {n: hashlib.sha256((inputs/n).read_bytes()).hexdigest() for n in files},
                'model_version': None, 'reasoning_effort': None, 'actual_cost_cny': None,
                'live_status': 'not_run_not_authorized', 'deployment': 'not_deployed',
                'records': [{'case': c['id'], 'status': 'not_run', 'answer': None, 'usage': None, 'review': None}
                            for c in spec['cases']]}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return {'manifest': str(output / 'manifest.json'), 'cases': len(spec['cases']), 'external_requests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare'); c.add_argument('baseline', type=Path); c.add_argument('candidate', type=Path)
    args = parser.parse_args()
    result = prepare(args.output) if args.command == 'prepare' else {
        'recommendation': compare_reviews(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
