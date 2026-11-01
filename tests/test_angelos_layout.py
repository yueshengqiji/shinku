"""分栏拖动/持久化和跨会话运行标记，不触发模型或任务操作。"""
from pathlib import Path
import subprocess
from test_angelos import env, new


def test_bootstrap_activity_without_body_or_extra_reads(env):
    client,bench,host=env
    ids=[new(client) for _ in range(4)]
    for cid,stage,status in zip(ids,['preparing','executing','finished','executing'],['preparing','running','completed','waiting_background']):
        bench.store.change('conversation',cid,lambda c,stage=stage,status=status:c['turns'].append({'id':'t','text':'合成输入','stage':stage,'view':{'status':status},'result':{'private':'do not project'}}))
    rows={c['id']:c for c in client.get('/angelos/api/bootstrap').json()['conversations']}
    assert rows[ids[0]]['activity']['stage']=='preparing'
    assert rows[ids[1]]['activity']['status']=='running'
    assert rows[ids[2]]['activity'] is None
    assert rows[ids[3]]['activity']['status']=='waiting_background'
    assert all('turns' not in c and 'result' not in c for c in rows.values())
    bench.store.change('conversation',ids[1],lambda c:c['turns'][0].update(stage='finished'))
    rows={c['id']:c for c in client.get('/angelos/api/bootstrap').json()['conversations']}
    assert rows[ids[1]]['activity'] is None and host.preparations==0
    assert client.get('/angelos/assets/layout.js').status_code==200


def test_splitter_pointer_keyboard_limits_and_reload(tmp_path):
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[2],'utf8');
const saved=new Map();
function boot(){
 const events={},panels={},styles={},classes=new Set();
 function element(){return {events:{},attrs:{},children:[],capture:null,addEventListener(n,fn){this.events[n]=fn},appendChild(c){this.children.push(c)},setAttribute(k,v){this.attrs[k]=v},setPointerCapture(p){this.capture=p},hasPointerCapture(p){return this.capture===p},releasePointerCapture(){this.capture=null}}}
 for(const id of ['sidebar','changes-panel','tools-panel'])panels[id]=element();
 const window={innerWidth:1400,innerHeight:900,addEventListener(n,f){events[n]=f}};
 const document={documentElement:{style:{setProperty(k,v){styles[k]=v},removeProperty(k){delete styles[k]}}},body:{classList:{add(v){classes.add(v)},remove(v){classes.delete(v)}}},getElementById(id){return panels[id]},createElement:element};
 vm.runInNewContext(source,{window,document,localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)}});
 return {window,events,panels,styles,classes};
}
let b=boot(),side=b.panels.sidebar.children[0],editor=b.panels['changes-panel'].children[0];
const e=(x,y=0)=>({button:0,pointerId:1,clientX:x,clientY:y,preventDefault(){}});
side.events.pointerdown(e(242));side.events.pointermove(e(350));side.events.pointerup(e(350));
assert.equal(b.styles['--side'],'350px');assert.equal(side.capture,null);assert(!b.classes.size);
b=boot();assert.equal(b.styles['--side'],'350px');side=b.panels.sidebar.children[0];editor=b.panels['changes-panel'].children[0];
side.events.keydown({key:'End',preventDefault(){}});assert.equal(b.styles['--side'],'420px');
side.events.dblclick();assert.equal(b.styles['--side'],'242px');
editor.events.pointerdown(e(800));editor.events.pointermove(e(-900));editor.events.pointercancel(e(-900));
assert.equal(b.styles['--editor-width'],'858px');assert(!b.classes.size);
b.window.innerWidth=700;b.events.resize();assert.equal(b.styles['--side'],undefined);assert.equal(editor.attrs['aria-orientation'],'horizontal');
editor.events.pointerdown(e(0,300));editor.events.pointermove(e(0,350));editor.events.pointerup(e(0,350));assert.equal(b.styles['--editor-height'],'446px');
editor.events.pointerdown(e(0,350));b.events.blur();assert(!b.classes.size);
'''
    path=tmp_path/'layout.cjs';path.write_text(script)
    result=subprocess.run(['node',str(path),str(Path(__file__).resolve().parents[1]/'frontend/angelos/layout.js')],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
