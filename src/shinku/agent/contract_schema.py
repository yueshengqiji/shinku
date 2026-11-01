"""声明工具的规范输入形状；语义、权限和效果仍由原验证器裁决。

与 task_state.validate_contract 的字段及边界同步，避免只向模型暴露空 object。
这里不创建另一份契约状态，也不把示例效果当成运行时能力目录。
"""


def contract_input_schema():
    text = {'type': 'string', 'minLength': 1, 'maxLength': 2000}
    texts = {'type': 'array', 'maxItems': 16, 'items': text}
    checks = {key: text for key in ('contains', 'sha256', 'page_contains', 'command')}
    checks.update({key: {'type': 'boolean'} for key in ('receipt_required', 'require_dates', 'unchanged')})
    checks.update({key: {'type': 'integer', 'minimum': 0}
                   for key in ('min_bytes', 'http_status', 'exit_code')})
    checks.update(required_fields={**texts, 'minItems': 1, 'maxItems': 8},
        min_sources={'type': 'integer', 'minimum': 1, 'maximum': 32},
        source_pages={'type': 'integer', 'minimum': 1, 'maximum': 100},
        date_kinds={'type': 'array', 'minItems': 1, 'maxItems': 3,
                    'items': {'type': 'string', 'enum': ['event', 'publication', 'updated']}})
    selector = {'type': 'object', 'additionalProperties': False, 'required': ['kind', 'value'],
        'properties': {'kind': {'type': 'string', 'enum': ['file_name', 'file_stem']},
            'value': {**text, 'maxLength': 255},
            'suffixes': {**texts, 'minItems': 1,
                'items': {'type': 'string', 'pattern': r'^(?:\.[A-Za-z0-9]{1,16}){1,3}$'}}}}
    requirement = {'type': 'object', 'additionalProperties': False,
        'required': ['id', 'description', 'effect'], 'properties': {
            'id': {**text, 'maxLength': 64}, 'description': text,
            'effect': {**text, 'maxLength': 64, 'description':
                '验收效果，不是kind。使用当前工具登记的效果，例如read/write、tool:<只读工具名>；'
                'local_answer要求checks.required_fields且targets为空；声明契约本身不是执行证据。'},
            'targets': {**texts, 'description': '真实文件/URL目标；代码成员放description，不另列成文件。'},
            'checks': {'type': 'object', 'additionalProperties': False, 'properties': checks,
                'description': '按效果校验：read的unchanged；send的receipt_required；'
                    'command_run必须只有command和exit_code=0；local_answer只允许required_fields/min_sources。'},
            'target_selector': selector}}
    return {'type': 'object', 'additionalProperties': False,
        'required': ['goal', 'deliverables', 'requirements'], 'properties': {
            'goal': text, 'deliverables': {**texts, 'minItems': 1}, 'scope': texts,
            'requirements': {'type': 'array', 'minItems': 1, 'maxItems': 24, 'items': requirement}}}
