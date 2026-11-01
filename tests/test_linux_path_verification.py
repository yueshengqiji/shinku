"""Path syntax must retain the identity of the executor's actual target."""
import pytest

from shinku.agent.loop import AgentState
from shinku.agent.verification import SmartCompletionVerifier as Verifier


def test_relative_target_uses_trusted_tool_root_and_still_refreshes_bytes(tmp_path):
    import hashlib
    from shinku.agent_runtime.demo import DemoFileTool
    root = tmp_path/'workspace'; root.mkdir()
    tool = DemoFileTool(root)
    class BoundTool:
        agent_completion_effects = ('write',)
        def agent_resolve_completion_target(self, *, target, context):
            path = root / target
            if not path.resolve().is_relative_to(root):
                raise PermissionError('outside_root')
            return str(path.resolve())
        agent_verify_result = tool.agent_verify_result
    path = root/'src'/'report.txt'; path.parent.mkdir(); path.write_text('verified')
    tool.path = path
    receipt = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'completion_targets': [str(path)], 'verified_effects': ['write']}
    call = {'name': 'writer', 'status': 'ok', 'executed': True,
            'arguments': {'path': str(path)}, 'data': {'state_updates': receipt}}
    verifier = Verifier(tool_capabilities={'writer': {'effects': ['write'], 'target_arguments': ['path']}},
                        result_handlers={'writer': BoundTool()})
    state = AgentState(context={}, tool_calls=[call])
    requirement = {'effect': 'write', 'targets': ['src/report.txt'], 'checks': {'contains': 'verified'}}
    assert verifier._verify_requirement(state, requirement, [call])
    assert requirement['targets'] == ['src/report.txt']
    # 同名异根及后续文件变化均不能借相对路径匹配获得成功。
    other = tmp_path/'other'/'src'/'report.txt'; other.parent.mkdir(parents=True); other.write_text('verified')
    call['data']['state_updates']['completion_targets'] = [str(other)]
    assert not verifier._verify_requirement(state, requirement, [call])
    call['data']['state_updates']['completion_targets'] = [str(path)]
    path.write_text('changed')
    assert not verifier._verify_requirement(state, requirement, [call])


def test_preflight_relative_deliverable_uses_same_host_identity(tmp_path):
    from types import SimpleNamespace
    handler = SimpleNamespace(agent_completion_effects=('write',),
        agent_resolve_completion_target=lambda target, context: str(tmp_path/target))
    verifier = Verifier(result_handlers={'writer': handler})
    contract = {'deliverables': ['src/output.py'], 'requirements': [
        {'id': 'r1', 'effect': 'write', 'targets': [str(tmp_path/'src/output.py')]}]}
    assert verifier.target_coverage_gaps(contract, ['保存 src/output.py']) == []
    contract['requirements'][0]['targets'] = [str(tmp_path.parent/'other/src/output.py')]
    assert verifier.target_coverage_gaps(contract, ['保存 src/output.py'])


def test_relative_resolution_requires_unambiguous_host_and_never_suffix_guesses(tmp_path):
    from types import SimpleNamespace
    from shinku.agent.target_identity import resolve_target
    def handler(root):
        return SimpleNamespace(agent_completion_effects=('write',),
            agent_resolve_completion_target=lambda target, context: str(root/target))
    assert resolve_target('src/a.py', 'write', {}) == 'src/a.py'
    assert resolve_target('src/a.py', 'write', {'a': handler(tmp_path), 'b': handler(tmp_path/'other')}) == 'src/a.py'
    assert resolve_target('src/a.py', 'read', {'a': handler(tmp_path)}) == 'src/a.py'
    assert resolve_target('a.py', 'write', {'a': handler(tmp_path)}) == 'a.py'
    assert resolve_target('https://example.invalid/a.py', 'write', {'a': handler(tmp_path)}) == 'https://example.invalid/a.py'


@pytest.mark.parametrize("requested,actual,matches", [
    ("/tmp/Report.txt", "/tmp/report.txt", False),
    ("/tmp/目录/File.txt", "/tmp/目录/file.txt", False),
    ("/tmp/Report.txt", "/tmp/./Report.txt", True),
    ("/tmp/Report.txt", "/other/Report.txt", False),
    ("Report.txt", "/tmp/Report.txt", True),
    ("Report.txt", "/tmp/report.txt", False),
    ("folder/Report.txt", "/tmp/folder/Report.txt", False),
    ("C:/Work/Report.txt", r"c:\work\REPORT.TXT", True),
    (r"\\server\share\Report.txt", "//server/share/report.txt", True),
    ("Report.txt", r"C:\Work\REPORT.TXT", True),
    ("https://example.test/Report", "https://example.test/report", False),
])
def test_targets_match_by_their_path_semantics(requested, actual, matches):
    expected = Verifier._normalize_target_values(requested)[0]
    observed = Verifier._normalize_target_values(actual)[0]
    assert Verifier._targets_match(expected, observed) is matches


@pytest.mark.parametrize("kind,value,path,matches", [
    ("file_name", "Report.txt", "/tmp/report.txt", False),
    ("file_stem", "Report", "/tmp/report.txt", False),
    ("file_name", "Report.txt", "/tmp/Report.txt", True),
    ("file_name", "Report.txt", "C:/tmp/REPORT.TXT", True),
])
def test_selector_preserves_posix_names(kind, value, path, matches):
    assert Verifier._selector_matches({"kind": kind, "value": value}, path) is matches


@pytest.mark.parametrize("suffixes", [None, [], [""], ".txt"])
def test_invalid_selector_suffixes_do_not_match(suffixes):
    assert not Verifier._selector_matches(
        {"kind": "file_stem", "value": "Report", "suffixes": suffixes}, "/tmp/Report.txt")


@pytest.mark.parametrize("path", ["/tmp/Report.txt", "/tmp/Report", "/tmp/目录/输入"])
def test_quoted_posix_targets_are_retained(path):
    assert Verifier._extract_targets(f'读取文件 "{path}"') == [path]


def test_posix_selector_uses_an_absolute_executor_receipt():
    requirement = {"id": "file", "description": "目标文件已删除", "effect": "delete", "targets": [],
                   "target_selector": {"kind": "file_name", "value": "Report.txt"}}
    call = {"name": "private_file_delete", "status": "ok", "arguments": {"path": "/tmp/Report.txt"},
            "data": {"state_updates": {"verified_effects": ["delete"], "completion_targets": ["/tmp/Report.txt"]}}}
    state = AgentState(context={"agent_completion_contract": {"goal": "删除目标", "deliverables": ["结果"],
        "requirements": [requirement]}}, tool_calls=[call])
    verdict = Verifier().evaluate(state, "已删除。")
    assert verdict.accepted
