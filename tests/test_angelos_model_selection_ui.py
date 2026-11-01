"""执行真实前端函数，确定性覆盖保存/轮询竞争，不调用供应商。"""
import subprocess
from pathlib import Path


def test_model_selector_transaction_and_polling(tmp_path):
    script = tmp_path / 'selector.cjs'
    script.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const src=fs.readFileSync(process.argv[2],'utf8');
const nodes=new Map();
const $=id=>{if(!nodes.has(id))nodes.set(id,{value:'',dataset:{},classList:{toggle(){}},close(){}});return nodes.get(id)};
const A={providerId:'a',modelId:'alpha',effort:'auto'},B={...A,modelId:'beta'};
const catalog={available:true,selected:A,providers:[{id:'a',configured:true,models:[{id:'alpha'},{id:'beta'}]},{id:'b',configured:true,models:[{id:'bad'},{id:'good'}]}]};
const ctx={console,$,document:{activeElement:null},window:{},esc:x=>x||'',active:()=>false,
 current:{id:'one',turns:[],modelSettings:A},conversations:[],agent:{model:'alpha'},modelCatalog:catalog,
 modelSaving:false,modelRevision:0,modelDraft:null,modelStatus:null,sending:false,selection:0,
 deletedConversations:new Set(),notify(){},updateComposer(){},pruneDeletedDrafts(){},renderTaskNotices(){},taskNotices:{observe(){}},
 refreshing:false,creatingConversation:false,conversationDeleting:false,online:true,
 renderConversation(){},renderMessages(){},showError(e){throw e},clearError(){},adopt(c){ctx.current=c},
 readLocal(){return {}},closeSidebar(){},loadWorkspaceInstructions(){}};
vm.createContext(ctx);
vm.runInContext(src.slice(src.indexOf('const effortLabels='),src.indexOf("$('settings-provider').addEventListener")),ctx);
function load(name,next){vm.runInContext(src.slice(src.indexOf('async function '+name),src.indexOf(next,src.indexOf('async function '+name))),ctx)}
load('bootstrap()', 'async function selectConversation');
load('selectConversation(', 'async function'); // 截取下一函数前的真实导航实现。
load('refresh(', '\nasync function');
const run=s=>vm.runInContext(s,ctx),tick=()=>new Promise(r=>setImmediate(r));
let resolve,reject,calls=0;ctx.api=()=>{calls++;return new Promise((r,j)=>{resolve=r;reject=j})};
(async()=>{
ctx.conversations=[ctx.current];ctx.document.activeElement=$('model-select');
let pending=ctx.applyModelSelection(B);
assert.equal($('model-select').value,'beta');assert.equal($('model-select').disabled,true);
ctx.renderModelSelectors();assert.equal($('model-select').value,'beta');
assert.equal(await ctx.applyModelSelection(A),undefined);assert.equal(calls,1);
resolve({models:{...catalog,selected:B}});assert.equal(await pending,true);
assert.equal(ctx.current.modelSettings.modelId,'beta');assert.equal($('model-select').value,'beta');
assert.match($('session-model-status').textContent,/已应用.*beta/);
// 失败也强制修正当前有焦点的下拉框，错误留在模型栏而非向外抛成连接错误。
pending=ctx.applyModelSelection(A);reject(new Error('synthetic refusal'));assert.equal(await pending,false);
assert.equal($('model-select').value,'beta');assert.match($('session-model-status').textContent,/alpha.*synthetic refusal/);
// 更换供应商只建立草稿，不自动提交它的第一个（可能不可用）模型。
$('provider-select').value='b';const before=calls;ctx.changeModelProvider();ctx.renderModelSelectors();
assert.equal(calls,before);assert.equal($('model-select').value,'');assert.equal($('effort-select').disabled,true);
assert.equal(ctx.modelSelectionForCurrent().providerId,'b');
pending=ctx.applyModelSelection({providerId:'b',modelId:'good',effort:'auto'});
resolve({models:{...catalog,selected:{providerId:'b',modelId:'good',effort:'auto'}}});await pending;
assert.equal($('provider-select').value,'b');assert.equal($('model-select').value,'good');
// 引导请求跨越保存代际时，其旧模型副本不能覆盖新配置。
let bootstrapResolve;ctx.api=()=>new Promise(r=>bootstrapResolve=r);let poll=ctx.bootstrap();
ctx.modelRevision++;bootstrapResolve({token:'test',conversations:[{id:'one',modelSettings:A}],workspaces:[],agent:{}});await poll;
assert.equal(ctx.current.modelSettings.modelId,'good');
// 目录请求迟到，同样不得写回旧模型。
ctx.api=async path=>path==='/bootstrap'?{token:'test',conversations:ctx.conversations,workspaces:[],agent:ctx.agent}:new Promise(r=>bootstrapResolve=r);
poll=ctx.bootstrap();await tick();ctx.modelRevision++;bootstrapResolve({models:catalog});await poll;
assert.equal(ctx.current.modelSettings.modelId,'good');
// 会话轮询已经发出，保存结束后才返回，不能再 adopt 旧会话配置。
ctx.bootstrap=async()=>{};ctx.api=()=>new Promise(r=>bootstrapResolve=r);poll=ctx.refresh();await tick();ctx.modelRevision++;
bootstrapResolve({conversation:{id:'one',modelSettings:A}});await poll;assert.equal(ctx.current.modelSettings.modelId,'good');
// 离开会话后，旧会话保存结果只能更新旧记录，不能改新会话模型或状态。
ctx.api=()=>new Promise(r=>resolve=r);pending=ctx.applyModelSelection(B);
ctx.current={id:'two',turns:[],modelSettings:A};ctx.modelDraft=null;ctx.modelStatus=null;
resolve({models:{...catalog,selected:B}});await pending;
assert.equal(ctx.current.modelSettings.modelId,'alpha');assert.equal(ctx.conversations[0].modelSettings.modelId,'beta');
assert.equal($('session-model-status').hidden,true);
console.log('selector transaction, provider draft, focused rollback, stale polls and isolation passed');
})().catch(e=>{console.error(e);process.exit(1)});
''')
    source = Path(__file__).parents[1] / 'frontend/angelos/app.js'
    result = subprocess.run(['node', str(script), str(source)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
