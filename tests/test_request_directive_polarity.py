"""否定只作用于本分句；后续肯定操作仍需真实执行，不授予任何权限。"""
import pytest

from shinku.agent.verification import SmartCompletionVerifier as Verifier


@pytest.mark.parametrize('text', [
    '分析项目，只分析，不修改文件。',
    '读取 input.txt 后解释，不写文件或发送消息。',
    '分析项目，不要修改 A.py，也不要发送 B.txt。',
    '分析项目，不得修改 A.py。',
    '分析项目，不需要再修改文件。',
    '分析项目，别再修改文件。',
    '分析项目，不再发送消息。',
    'Analyze input.txt; do not write output.txt or send result.txt.',
])
def test_negated_changes_are_not_execution_requirements(text):
    assert not Verifier.request_requires_action(text)


@pytest.mark.parametrize('text', [
    '请修改 A.py，不要发送 B.txt。',
    '不要修改 A.py，但请发送 B.txt。',
    '禁止发送 B.txt，然后修改 A.py。',
    '不能不修改 A.py。',
    '不得不修改 A.py。',
    '不要不修改 A.py。',
    '把别名修改成新名称。',
    '请对 config.json 进行“修改”',
    '分析“不修改”这个说法，并修改 A.py。',
])
def test_positive_or_double_negative_actions_remain_strict(text):
    assert Verifier.request_requires_action(text)


def test_negated_target_role_does_not_hide_positive_read_or_write():
    roles = Verifier._explicit_target_roles(['读取 input.txt，不修改 output.txt；保存 result.txt。'])
    assert 'read' in roles.get('input.txt', set())
    assert 'write' not in roles.get('output.txt', set())
    assert 'write' in roles.get('result.txt', set())
