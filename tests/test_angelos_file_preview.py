"""改动列表目录切换与大目录单文件预览；不借扫描降级放宽文件边界。"""
import subprocess
from pathlib import Path
import pytest
from test_angelos_files import files


def test_large_directory_preview_does_not_scan_or_forge_baseline(files, monkeypatch):
    f,w,r=files;(r/'real.py').write_text('value = 1\n')
    def cannot_scan(*args):raise ValueError('workspace_scan_limit')
    monkeypatch.setattr(f,'walk',cannot_scan)
    with pytest.raises(ValueError,match='workspace_scan_limit'):f.changes(w)
    row=f.open(w,'real.py')
    assert row['content']=='value = 1\n' and row['exists'] and row['editable']
    assert row['comparisonAvailable'] is False and row['original']==row['content']
    assert f.store.list('file_baseline')==[]
    saved=f.save(w,'real.py','value = 2\n',row['version'])
    assert saved['content']=='value = 2\n' and saved['comparisonAvailable'] is False
    assert f.store.list('manual_file_edit')[0]['before']=='value = 1\n'
    with pytest.raises(ValueError,match='file_not_found'):f.open(w,'missing.py')
    with pytest.raises(ValueError,match='file_path_outside_workspace'):f.open(w,'../real.py')


def test_existing_snapshot_and_deleted_file_still_have_comparison(files):
    f,w,r=files;(r/'real.py').write_text('before');f.changes(w)
    (r/'real.py').write_text('after');row=f.open(w,'real.py')
    assert row['comparisonAvailable'] is True and row['original']=='before'
    (r/'real.py').unlink();row=f.open(w,'real.py')
    assert row['exists'] is False and row['original']=='before' and row['comparisonAvailable'] is True


def test_change_list_is_bound_to_workspace_even_during_request_and_confirmation(tmp_path):
    source=Path(__file__).parents[1]/'frontend/angelos/workspace.js'
    script=tmp_path/'preview.cjs'
    script.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const s=fs.readFileSync(process.argv[2],'utf8'),nodes=new Map(),handlers={};
const ui=id=>{if(!nodes.has(id))nodes.set(id,{dataset:{},textContent:'',innerHTML:'',replaceChildren(){this.innerHTML=''},addEventListener(name,fn){handlers[id+':'+name]=fn}});return nodes.get(id)};
let pending,opened=[],confirm;
const ctx={ui,labels:{modified:'修改'},renderSelection(){},activeWorkspaceId:'a',root:'/a',workspace(){return {path:ctx.root}},selectedWorkspace:'',refreshing:false,online:true,agent:{workbenchControls:true},opened:null,esc:String,
 status(t){ctx.statusText=t},afterDiscard(fn){confirm=fn},openFile(wid,path){opened.push({wid,path})},api(){return new Promise((r,j)=>pending={r,j})}};
vm.createContext(ctx);
vm.runInContext(s.slice(s.indexOf(' async function refreshChanges()'),s.indexOf(" window.addEventListener('beforeunload'")),ctx);
vm.runInContext(s.slice(s.indexOf(" ui('changed-files').addEventListener"),s.indexOf(' // 视图切换')),ctx);
const file={path:'same.py',status:'modified',added:1,removed:0};
const result={changes:{files:[file],skipped:[],baseline:'HEAD',added:1,removed:0}};
(async()=>{
let p=ctx.refreshChanges();pending.r(result);await p;
assert(ui('changed-files').innerHTML.includes('same.py'));
let old=ctx.refreshChanges();const late=pending;
ctx.activeWorkspaceId='b';ctx.root='/b';await ctx.refreshChanges();
assert.equal(ui('changed-files').innerHTML,'');assert.equal(ui('changed-files').dataset.workspaceId,'b');
late.r(result);await old;assert.equal(ui('changed-files').innerHTML,'');
p=ctx.refreshChanges();pending.j(new Error('scan failed'));await p;
assert.equal(ui('changed-files').innerHTML,'');assert.equal(ui('changes-note').textContent,'scan failed');
p=ctx.refreshChanges();pending.r(result);await p;
handlers['changed-files:click']({target:{closest(){return {dataset:{codePath:'same.py'}}}}});
confirm();assert.deepEqual(opened,[{wid:'b',path:'same.py'}]);
// 用户确认丢弃草稿期间换目录，也不能执行原来排队的打开。
handlers['changed-files:click']({target:{closest(){return {dataset:{codePath:'same.py'}}}}});
ctx.root='/b-new';confirm();assert.equal(opened.length,1);
// 同一工作区ID改关联目录，旧响应仍然无效。
p=ctx.refreshChanges();const oldRoot=pending;ctx.root='/b-later';await ctx.refreshChanges();
oldRoot.r(result);await p;assert.equal(ui('changed-files').innerHTML,'');
})().catch(e=>{console.error(e);process.exit(1)});
''')
    result=subprocess.run(['node',str(script),str(source)],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
