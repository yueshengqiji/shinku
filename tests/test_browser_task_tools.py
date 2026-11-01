"""浏览器真实页面反例：目标语义、动态条件、标签页与任务资源归属。"""
from pathlib import Path

import pytest

from shinku.agent.request_scope import RequestScope
from shinku.tools.browser import build_browser_tools
from shinku.tools.execution import execute_invocation
from shinku.tools.invocation import ToolInvocation


def context(turn='turn'):
    return {'agent_request_scope': RequestScope('owner', 'session', turn, ((turn, 'fixture'),)),
            'agent_permission_mode': 'full'}


@pytest.fixture
def browser(tmp_path):
    pw = pytest.importorskip('playwright.sync_api')
    with pw.sync_playwright() as p:
        if not Path(p.chromium.executable_path).is_file():
            pytest.skip('matching local Chromium unavailable; no installation')
        runtime = p.chromium.launch(headless=True)
        ctx = runtime.new_context(accept_downloads=True)
        ctx.route('**/*', lambda route: route.abort())
        page = ctx.new_page()
        tools = {t.tool_type: t for t in build_browser_tools(page_provider=lambda c: page,
            run=lambda f: f(), authorize=lambda c, a, u: True, download_root=tmp_path)}
        try:
            yield page, tools
        finally:
            ctx.close(); runtime.close()


def invoke(tools, name, args=None, ctx=None):
    return execute_invocation(ToolInvocation('browser_' + name, args or {}), handlers=tools,
                              context=ctx or context())[1]


def observe(tools, **kwargs):
    value = invoke(tools, 'observe', kwargs)
    assert value.status == 'ok', value.model_feedback
    return value.data['state_updates']['browser_observation']


def act(tools, state, action, **kwargs):
    return invoke(tools, 'interact', {'state_id': state['state_id'], 'request_key': action,
                                     'action': action, **kwargs})


def test_semantic_labels_open_shadow_and_iframe_are_observable(browser):
    page, tools = browser
    page.set_content('''<label for="title">标题</label><input id="title">
      <div role="button" tabindex="0" aria-labelledby="name"><span id="name">自定义保存</span></div>
      <div id="shadow"></div><iframe srcdoc="<button>框架保存</button>"></iframe>''')
    page.locator('#shadow').evaluate("e=>e.attachShadow({mode:'open'}).innerHTML='<button>影子保存</button>'")
    page.frame_locator('iframe').get_by_role('button').wait_for()
    state = observe(tools)
    assert {'标题', '自定义保存', '影子保存', '框架保存'} <= {t['label'] for t in state['targets']}
    assert len(state['frames']) == 2


def test_target_and_text_pagination_reaches_the_tail(browser):
    page, tools = browser
    page.set_content('<main>' + '正文' * 9000 + '</main>' + ''.join(f'<button>button-{i}</button>' for i in range(105)))
    first = observe(tools)
    assert first['targets_complete'] is False and first['page_text_complete'] is False
    tail = observe(tools, target_offset=100, text_offset=16000)
    assert tail['targets'][-1]['label'] == 'button-104' and tail['next_target_offset'] is None
    assert tail['text_offset'] == 16000 and tail['next_text_offset'] is None


def test_unrelated_text_does_not_stale_target_but_target_replacement_does(browser):
    page, tools = browser
    page.set_content('<span id="clock">1</span><button onclick="this.dataset.clicked=1">Save</button>')
    state = observe(tools)
    page.locator('#clock').evaluate('e=>e.textContent="2"')
    assert act(tools, state, 'click', ref=state['targets'][0]['ref']).status == 'ok'
    fresh = observe(tools)
    page.locator('button').evaluate('e=>e.outerHTML=e.outerHTML')
    result = act(tools, fresh, 'fill', ref=fresh['targets'][0]['ref'], value='no')
    assert result.status == 'error' and 'target' in result.model_feedback


def test_new_element_and_disappearing_overlay_waits(browser):
    page, tools = browser
    page.set_content('<div id="loading">Loading</div>')
    state = observe(tools)
    page.evaluate('''setTimeout(()=>{document.querySelector('#loading').remove();
      document.body.insertAdjacentHTML('beforeend','<output id="result">Saved</output>')},150)''')
    hidden = act(tools, state, 'wait', condition='hidden', selector='#loading')
    assert hidden.status == 'ok', hidden.model_feedback
    state = hidden.data['state_updates']['browser_observation']
    found = invoke(tools, 'interact', {'action':'wait','condition':'text','selector':'#result','value':'Saved',
        'state_id':state['state_id'],'request_key':'result'})
    assert found.status == 'ok', found.model_feedback


def test_href_change_is_not_authorized_by_unchanged_label(browser):
    page, tools = browser
    page.set_content('<a href="https://one.example/">Same label</a>')
    state = observe(tools)
    page.locator('a').evaluate('e=>e.href="https://two.example/"')
    result = act(tools, state, 'click', ref=state['targets'][0]['ref'])
    assert result.status == 'error' and 'target_changed' in result.model_feedback
    assert page.url == 'about:blank'


def test_observation_receipt_rejects_modified_payload(browser):
    from copy import deepcopy
    page, tools = browser
    page.set_content('<button>Original</button>')
    result = invoke(tools, 'observe').data['state_updates']
    assert tools['browser_observe'].agent_verify_result(receipt=result, context=context())['result_valid']
    forged = deepcopy(result)
    forged['browser_observation']['text'] = 'invented success'
    assert not tools['browser_observe'].agent_verify_result(receipt=forged, context=context())['result_valid']


def test_shadow_and_frame_actions_use_real_observed_nodes(browser):
    page, tools = browser
    page.set_content('<div id="shadow"></div><iframe srcdoc="<input aria-label=Frame><output></output>"></iframe>')
    page.locator('#shadow').evaluate("e=>e.attachShadow({mode:'open'}).innerHTML='<input aria-label=Shadow>'")
    page.frame_locator('iframe').get_by_label('Frame').wait_for()
    state = observe(tools)
    for label in ('Shadow', 'Frame'):
        target = next(t for t in state['targets'] if t['label'] == label)
        result = invoke(tools, 'interact', {'action':'fill','value':label+' value','ref':target['ref'],
            'state_id':state['state_id'],'request_key':label})
        assert result.status == 'ok', result.model_feedback
        state = result.data['state_updates']['browser_observation']
    assert page.get_by_label('Shadow').input_value() == 'Shadow value'
    assert page.frame_locator('iframe').get_by_label('Frame').input_value() == 'Frame value'


def test_form_changes_and_cross_turn_references_are_rejected(browser):
    page, tools = browser
    page.set_content('<form><input aria-label="amount" value="1"><button>Submit</button></form>')
    state = observe(tools)
    target = next(t for t in state['targets'] if t['label'] == 'Submit')
    page.get_by_label('amount').fill('999')
    result = act(tools, state, 'click', ref=target['ref'])
    assert result.status == 'error' and 'target_changed' in result.model_feedback
    result = invoke(tools,'interact',{'action':'click','ref':target['ref'],'state_id':state['state_id'],
        'request_key':'other'},context('other-turn'))
    assert result.status == 'error'


def test_wait_timeout_is_observation_failure_not_uncertain_effect(browser):
    page, tools = browser
    page.set_content('<button>Ready</button>')
    state = observe(tools)
    result = act(tools,state,'wait',selector='#absent',condition='visible',timeout_ms=30)
    assert result.status == 'error' and 'wait_timeout' in result.model_feedback
    assert not tools['browser_interact'].access.uncertain
    assert act(tools,state,'click',ref=state['targets'][0]['ref']).status == 'ok'


def test_keyboard_checkbox_hover_and_scroll_use_observed_elements(browser):
    page, tools = browser
    page.set_content('''<input aria-label="Entry" onkeydown="if(event.key==='Enter')this.dataset.enter='yes'">
        <input type="checkbox" aria-label="Choice"><button onmouseover="this.dataset.hover='yes'">Hover</button>
        <div style="height:2000px"></div><button>Tail</button>''')
    state = observe(tools)
    for action,label,value in [('press','Entry','Enter'),('check','Choice','true'),('hover','Hover',None),('scroll','Tail',None)]:
        result = act(tools,state,action,ref=next(t['ref'] for t in state['targets'] if t['label']==label),
                     **({'value':value} if value is not None else {}))
        assert result.status == 'ok',result.model_feedback
        state = result.data['state_updates']['browser_observation']
    assert page.get_by_label('Entry').get_attribute('data-enter') == 'yes'
    assert page.get_by_label('Choice').is_checked()
    assert page.get_by_text('Hover',exact=True).get_attribute('data-hover') == 'yes'
    assert page.evaluate('scrollY') > 1000
