"""回复相对路径的真实浏览器回归：子项目归属、歧义、错误与草稿边界。"""
import json

from test_angelos_coding_view import coding


def register(c, name, root):
    root.mkdir(parents=True, exist_ok=True)
    return c.bench.store.workspace({'name': name, 'path': str(root)})


def reply_link(c, path):
    # 保存合成回复，再用实际正文渲染与点击委托；不发起模型任务。
    c.bench.store.change('conversation',c.conversation['id'],lambda row:row.update(turns=[{
        'id':'file-reply','stage':'replied','text':'查看文件','reply':'请查看 `'+path+'`',
        'attachments':[],'created':1,'result':{},
    }]))
    c.page.evaluate('() => refresh(true)')
    c.page.locator('#view-chat').click()
    c.page.locator('[data-open-path='+json.dumps(path)+']').click()


def test_reply_relative_paths_find_nested_project_and_deduplicate_roots(coding):
    c=coding
    root=c.project/'nested'/'repo'
    register(c,'子项目',root)
    register(c,'相同目录',root)
    paths=['src/shinku/angelos/execution_log.py','src/shinku/api/angelos.py','frontend/angelos/log-view.js']
    for i, path in enumerate(paths):
        f=root/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(f'CONTENT_{i}\n')
    c.page.evaluate('() => bootstrap()')
    for i,path in enumerate(paths):
        reply_link(c,path)
        c.page.locator('#code-editor .cm-content').last.filter(has_text=f'CONTENT_{i}').wait_for()
        assert c.page.evaluate('activeWorkspaceId')==c.workspace['id']
    assert not c.calls


def test_existing_current_file_wins_and_ambiguous_missing_file_requires_selection(coding):
    c=coding
    for name in ['one','two']:
        root=c.project/name;register(c,name,root)
        (root/'shared.py').write_text(name+'\n')
        (root/'sample.py').write_text('OTHER_SAMPLE\n')
    c.page.evaluate('() => bootstrap()')
    reply_link(c,'sample.py')
    assert 'class Example' in c.page.locator('#code-editor .cm-content').last.inner_text()
    reply_link(c,'shared.py')
    choices=c.page.locator('.path-open-notice button')
    choices.first.wait_for();assert choices.count()==2
    assert c.page.locator('#code-path').inner_text()=='sample.py'
    choices.filter(has_text='two ·').click()
    c.page.locator('#code-editor .cm-content').last.filter(has_text='two').wait_for()
    assert c.page.evaluate('activeWorkspaceId')==c.workspace['id']


def test_missing_file_fallback_preserves_draft_and_propagates_nonmissing_error(coding):
    c=coding;root=c.project/'repo';register(c,'子项目',root)
    (root/'next.py').write_text('NEXT_FILE\n')
    c.page.evaluate('() => bootstrap()')
    c.page.locator('#code-mode').select_option('edit')
    view=c.page.locator('#code-editor .cm-content').last
    view.click();c.page.keyboard.press('Control+End');c.page.keyboard.insert_text('# DRAFT\n')
    reply_link(c,'next.py')
    c.page.locator('#discard-code-dialog').wait_for()
    c.page.locator('#discard-code-cancel').click()
    assert 'DRAFT' in view.inner_text()
    reply_link(c,'next.py');c.page.locator('#discard-code-confirm').click()
    c.page.locator('#code-editor .cm-content').last.filter(has_text='NEXT_FILE').wait_for()
    requests=[]
    def reject(route):
        requests.append(route.request.url)
        route.fulfill(status=403,content_type='application/json',body=json.dumps({'ok':False,'reason':'file_path_unavailable'}))
    c.page.route('**/file?path=blocked.py',reject)
    reply_link(c,'blocked.py')
    c.page.locator('#code-status').filter(has_text='符号链接').wait_for()
    assert len(requests)==1


def test_late_fallback_does_not_replace_newly_opened_file(coding):
    c=coding;root=c.project/'repo';target=register(c,'子项目',root)
    (root/'slow.py').write_text('LATE_FILE\n')
    c.page.evaluate('() => bootstrap()')
    pending=[]
    c.page.route('**/workspaces/'+target['id']+'/file?path=slow.py',lambda route:pending.append(route))
    reply_link(c,'slow.py')
    c.page.wait_for_timeout(250)
    assert len(pending)==1
    c.page.locator('#open-code-path').fill('sample.py')
    c.page.locator('#open-code-form button').filter(has_text='打开文件').click()
    c.page.locator('#code-editor .cm-content').last.filter(has_text='class Example').wait_for()
    pending[0].fulfill(status=200,content_type='application/json',body=json.dumps({'ok':True,'file':{'exists':True}}))
    c.page.wait_for_timeout(100)
    assert c.page.locator('#code-path').inner_text()=='sample.py'
    assert 'LATE_FILE' not in c.page.locator('#code-editor').inner_text()
