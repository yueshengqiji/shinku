"""显式配置的goozam模型通道探针；默认仅校验，--execute才会产生付费请求。

配置由操作者在确认范围后单独填写，不读取正式工厂、注册表或环境密钥。这里只
执行合成探针或显式选定的最多三个实验任务提案；费用授权记录不能由模型生成，
本命令也不自动批准后续评估或启用。
"""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from goozam.model_gateway import BatchLimits, ModelRoute, serve_gateway, summarize_batch
from goozam.sandbox import plain_tree, run


def settings(path, lab):
    """配置/密钥必须为独立受限文件，且不可位于沙箱的任何可见目录。"""
    path = plain_tree(path)
    exposed = (lab, ROOT / 'src/goozam', Path(sys.base_prefix), Path(sys.prefix), Path('/usr'))
    if not path.is_file() or any(path.is_relative_to(p) for p in exposed):
        raise ValueError('goozam_external_config_required')
    if path.stat().st_uid != os.getuid() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('goozam_private_config_permissions_required')
    value = json.loads(path.read_text())
    if set(value) != {'route', 'limits', 'key_file'} or 'fixture_http' in value['route']:
        raise ValueError('goozam_explicit_config_required')
    route, limits = ModelRoute(**value['route']), BatchLimits(**value['limits'])
    key_path = plain_tree(Path(value['key_file']))
    if not key_path.is_file() or any(key_path.is_relative_to(p) for p in exposed):
        raise ValueError('goozam_external_key_required')
    if key_path.stat().st_uid != os.getuid() or stat.S_IMODE(key_path.stat().st_mode) & 0o077:
        raise ValueError('goozam_private_key_permissions_required')
    key = key_path.read_text().strip()
    if not key or '\n' in key or '\r' in key:
        raise ValueError('goozam_explicit_key_required')
    return route, limits, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--lab', type=Path, default=ROOT / 'private/experiments/goozam/angelos2')
    parser.add_argument('--venv', type=Path, default=Path(sys.prefix))
    parser.add_argument('--action', choices=('model-probe', 'learn-propose'), default='model-probe')
    parser.add_argument('--sources', type=Path, help='显式任务引用JSON列表，仅learn-propose使用')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    lab = plain_tree(args.lab)
    route, limits, key = settings(args.config, lab)
    batch = plain_tree(args.batch_dir)
    if batch.exists() or batch.is_relative_to(lab) or lab.is_relative_to(batch):
        raise ValueError('goozam_new_external_batch_directory_required')
    refs = None
    if args.action == 'learn-propose':
        if args.sources is None:
            raise ValueError('goozam_explicit_proposal_refs_required')
        refs = json.loads(plain_tree(args.sources).read_text())
        if (not isinstance(refs, list) or not 1 <= len(refs) <= 3
                or any(not isinstance(r, dict) or set(r) != {'session', 'task_id'}
                       or any(not isinstance(v, str) or not v or len(v) > 256 for v in r.values())
                       for r in refs)):
            raise ValueError('goozam_explicit_proposal_refs_required')
    elif args.sources is not None:
        raise ValueError('goozam_unexpected_proposal_refs')
    if not args.execute:
        print(json.dumps(dict(status='validated_not_executed', model=route.model,
                              limits=vars(limits)), ensure_ascii=False, indent=2))
        return
    # 真实调用授权在执行前由操作者核实；不把配置中一个字符串当作审批系统。
    result, error = None, None
    try:
        with serve_gateway(batch, route, limits, key) as sock:
            result = run(lab, args.venv, action=args.action, relay_socket=sock, proposal_refs=refs,
                         model_profile=route.public(), timeout=limits.seconds)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        error = exc
        if batch.exists():
            for name, value in (('stdout', exc.stdout), ('stderr', exc.stderr)):
                if value is not None:
                    (batch / ('probe.' + name)).write_bytes(value.encode() if isinstance(value, str) else value)
            (batch / 'probe-error.json').write_text(json.dumps({'error': type(exc).__name__}))
    if result is not None:
        (batch / 'probe.stdout').write_text(result.stdout)
        (batch / 'probe.stderr').write_text(result.stderr)
    summary = summarize_batch(batch)
    (batch / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if error is not None:
        raise error
    report = json.loads(result.stdout)
    if args.action == 'learn-propose':
        if (not summary['ledger_complete']
                or (summary['requests'] and summary.get('estimated_cny') is None)
                or any(r['status'] not in {'candidate', 'skipped'} for r in report['proposals'])):
            raise RuntimeError('goozam_proposals_not_accepted_see_batch_records')
        print(json.dumps(dict(status='candidates_not_learning_gain', batch=str(batch),
                              proposals=report['proposals']), ensure_ascii=False))
        return
    terminal = (report.get('turn') or {}).get('result') or {}
    if (not summary['requests'] or not summary['ledger_complete'] or summary['estimated_cny'] is None
            or summary['statuses'] != {'completed': summary['requests']}
            or terminal.get('execution', {}).get('status') != 'completed'
            or terminal.get('verification', {}).get('accepted') is not True):
        raise RuntimeError('goozam_probe_not_accepted_see_batch_records')
    print(json.dumps(dict(status='probe_completed_not_learning_gain', batch=str(batch),
                         requests=summary['requests'], task_id=report.get('task_id')), ensure_ascii=False))


if __name__ == '__main__':
    main()
