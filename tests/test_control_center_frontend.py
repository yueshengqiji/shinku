"""正式中枢的结构/隔离前端回归；全部使用公开源码和合成数据。"""
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.anchors = []
        self.dialogs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if tag == 'a' and attrs.get('href', '').startswith('#'):
            self.anchors.append(attrs['href'][1:])
        if tag == 'dialog':
            self.dialogs.append(attrs.get('id'))


class ControlCenterFrontendTests(unittest.TestCase):
    def test_original_feature_targets_and_anchor_integrity(self):
        parser = Elements()
        parser.feed((ROOT / 'frontend/control-center/index.html').read_text())
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertFalse(set(parser.anchors) - set(parser.ids))
        retained = {'service', 'agent', 'memory', 'napcat', 'desktop-pet', 'runtime-config', 'metrics', 'model-services', 'metrics-recent', 'attention-traces', 'memory-routing', 'traces', 'qq-diagnostics', 'agent-tasks', 'agent-actions', 'pet-actions', 'capabilities', 'approval-conversation', 'approval-inspect', 'approval-result', 'memory-conversation', 'memory-type', 'memory-mode', 'memory-query', 'memory-inspect', 'memory-result', 'logs', 'updated', 'refresh-now', 'auto-refresh'}
        self.assertFalse(retained - set(parser.ids))
        self.assertEqual(set(parser.dialogs), {'model-settings', 'agent-settings', 'pet-settings', 'display-settings'})

    @unittest.skipUnless(shutil.which('node'), 'Node is required for isolated frontend state tests')
    def test_frontend_state_regressions(self):
        result = subprocess.run(['node', '--test', str(ROOT / 'tests/control_center_frontend.test.cjs')], cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
