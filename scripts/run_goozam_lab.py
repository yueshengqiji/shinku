"""运行Angelos2的隔离离线探针/学习演练；不提供正式工厂或真实模型选项。"""
import argparse
from pathlib import Path
import sys

# 脚本明确加载本仓实验模块，沙箱内再显式映射该模块与固定Angelos源码。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from goozam.sandbox import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('probe', 'exercise', 'plugin-status', 'plugin-off', 'plugin-learn', 'plugin-use'), default='probe')
    parser.add_argument('--lab', type=Path, default=ROOT / 'private/experiments/goozam/angelos2')
    parser.add_argument('--venv', type=Path, default=Path(sys.prefix))
    args = parser.parse_args()
    result = run(args.lab, args.venv, action=args.action)
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)


if __name__ == '__main__':
    main()
