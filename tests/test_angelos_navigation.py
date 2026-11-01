"""新对话/工作区切换行为；不执行模型、取消任务或覆盖已有草稿。"""
from pathlib import Path
import subprocess


def test_navigation_no_duplicate_or_draft_loss(tmp_path):
    app=Path(__file__).resolve().parents[1]/'frontend/angelos/app.js'
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let source=fs.readFileSync(process.argv[2],'utf8');
source=source.slice(source.indexOf('async function newConversation('),source.indexOf('let refreshing=false;'));
const context={console};vm.createContext(context);
vm.runInContext(`
let sending=false,creatingConversation=false,selection=0,activeWorkspaceId='a';
let workspaces=[{id:'a'},{id:'b'}],conversations=[],current={id:'empty',workspaceId:'a',turns:[],draft:'',pending:[]};
let calls=[],saved=[],focus=0,rendered=0,resolveCreate;
const elements=new Map();function $(id){if(!elements.has(id))elements.set(id,{disabled:false,value:'',close(){},focus(){focus++}});return elements.get(id)}
function renderSidebar(){rendered++}function closeSidebar(){}function saveDraft(c){saved.push({...c})}
function readLocal(){return null}function upsert(c){conversations.push(c)}
async function api(path,body){calls.push({path,body});return await new Promise(r=>resolveCreate=r)}
async function selectConversation(id){selection++;current=conversations.find(c=>c.id===id);activeWorkspaceId=current.workspaceId}
`+source,context);
(async()=>{
 await vm.runInContext('newConversation()',context);assert.equal(vm.runInContext('calls.length',context),0);
 vm.runInContext("current.draft='保留此草稿';current.turns=[{stage:'executing'}]",context);
 const first=vm.runInContext('newConversation()',context);
 await vm.runInContext('newConversation()',context);
 assert.equal(vm.runInContext('calls.length',context),1);
 assert.equal(vm.runInContext('saved[0].draft',context),'保留此草稿');
 vm.runInContext("resolveCreate({conversation:{id:'new',workspaceId:'a',turns:[],draft:'',pending:[]}})",context);await first;
 assert.equal(vm.runInContext('current.id',context),'new');
 assert.equal(vm.runInContext('creatingConversation',context),false);
 const switching=vm.runInContext("selectWorkspace('b')",context);
 assert.equal(vm.runInContext('calls[1].body.workspaceId',context),'b');
 vm.runInContext("resolveCreate({conversation:{id:'bnew',workspaceId:'b',turns:[],draft:'',pending:[]}})",context);await switching;
 assert.equal(vm.runInContext('activeWorkspaceId',context),'b');
 await vm.runInContext("selectWorkspace('a')",context);assert.equal(vm.runInContext('current.id',context),'new');
 assert.equal(vm.runInContext('calls.length',context),2);
 // 新建回执迟到时不能把后来选中的会话切走。
 vm.runInContext("current.draft='another draft'",context);
 const late=vm.runInContext('newConversation()',context);
 await vm.runInContext("selectConversation('bnew')",context);
 vm.runInContext("resolveCreate({conversation:{id:'late',workspaceId:'a',turns:[],draft:'',pending:[]}})",context);await late;
 assert.equal(vm.runInContext('current.id',context),'bnew');
 assert(vm.runInContext("calls.every(c=>c.path==='/conversations')",context));
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    path=tmp_path/'navigation.cjs';path.write_text(script)
    result=subprocess.run(['node',str(path),str(app)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
