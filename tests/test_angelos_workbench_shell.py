"""工作台布局复用真实入口；视图、快捷操作不得清除草稿或中断任务。"""
from test_angelos_coding_view import coding


def command(page, text):
    page.keyboard.press('Control+k')
    page.locator('#command-dialog').wait_for()
    page.locator('#command-search').fill(text)
    page.locator('#command-search').press('Enter')


def test_commands_and_detail_tabs_preserve_editor_draft(coding):
    c,p=coding,coding.page
    p.locator('#code-mode').select_option('edit')
    editor=p.locator('#code-editor .cm-content').last
    editor.click();p.keyboard.press('Control+End');p.keyboard.insert_text('# KEEP_DRAFT\n')
    command(p,'查看工具调用')
    p.locator('#tools-panel').wait_for(state='visible')
    assert p.locator('#changes-panel').is_hidden()
    p.locator('#tools-panel [data-detail-tab=files]').click()
    p.locator('#changes-panel').wait_for(state='visible')
    assert p.locator('#tools-panel').is_hidden()
    assert 'KEEP_DRAFT' in editor.inner_text()
    assert p.locator('#code-mode').input_value()=='edit'
    assert p.locator('#code-location').get_attribute('title')==str(c.project)
    assert p.locator('#code-path').get_attribute('title')==str(c.project/'sample.py')
    assert p.locator('#copy-code-path').is_enabled()
    assert not c.calls and c.tool.calls==0


def test_command_new_conversation_keeps_running_task(coding):
    c,p=coding,coding.page
    p.locator('#message-input').fill('HOLD_TASK 合成任务')
    p.locator('#send-message').click()
    assert c.entered.wait(8)
    p.wait_for_function('()=>current.turns.length>0')
    original=p.evaluate('current.id')
    command(p,'新建对话')
    p.wait_for_function('(old)=>current.id!==old',arg=original)
    assert not c.release.is_set() and len(c.calls)==1
    assert c.bench.store.read('conversation',original)['turns'][0]['stage']=='executing'
    assert p.locator('#running-list [data-conversation="'+original+'"]').count()==1
    c.release.set()


def test_statusbar_real_totals_and_mobile_controls_remain_visible(coding):
    c,p=coding,coding.page
    assert p.locator('.workbench-statusbar #permission-select').count()==1
    assert p.locator('.workbench-statusbar #usage-instance-hit').count()==1
    p.locator('#usage-instance-hit').click()
    assert p.locator('[data-usage-scope=instance]').get_attribute('aria-pressed')=='true'
    p.locator('[aria-label="关闭用量统计"]').click()
    p.set_viewport_size({'width':390,'height':844})
    p.wait_for_function('()=>document.querySelector(".changes-panel").getBoundingClientRect().top>=document.querySelector(".topbar").getBoundingClientRect().bottom')
    assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
    footer=p.locator('.workbench-statusbar').bounding_box();composer=p.locator('#composer').bounding_box()
    assert composer['y']+composer['height']<=footer['y']+1
    assert p.locator('#permission-select').is_visible()
    assert p.locator('#code-editor').bounding_box()['height']>=50
    p.locator('#view-chat').click()
    assert p.locator('#changes-panel').is_hidden()
    assert p.locator('#message-input').is_visible()
    assert not c.calls
