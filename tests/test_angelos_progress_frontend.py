"""实际前端函数的离线执行：SSE优先、乱序保护、无进度时也显示等待状态。"""
from pathlib import Path
import subprocess


def test_live_progress_selection_and_rendering():
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const scope=vm.createContext({window:{}});
vm.runInContext(fs.readFileSync('frontend/angelos/activity.js','utf8'),scope);
const select=scope.window.AngelosActivity.progressFor;
const turn={id:'turn',task_id:'task',stage:'executing',live_progress:{
  taskId:'task',attempt:1,version:2,waitSeconds:0,phase:'model',available:true}};
const live={taskId:'task',attempt:1,version:3,phase:'tool',available:true,
  updates:{rows:[{id:1,text:'发现旧版本未加载，正在检查回滚记录。'}]},text:'正在搜索项目文件。',events:[{state:'ok',text:'读取代码：已取得回执'}],totalEvents:1};
assert.strictEqual(select(turn,{progress:live}),live);
assert.strictEqual(select(turn,{progress:{...live,taskId:'foreign'}}),turn.live_progress);
assert.strictEqual(select(turn,{progress:{...live,attempt:0}}),turn.live_progress);
assert.strictEqual(select(turn,{progress:{...live,version:1}}),turn.live_progress);
assert.strictEqual(select(turn,{progress:{available:false}}),turn.live_progress);
assert.strictEqual(select(turn,{progress:{available:true,phase:'preparing'}}),turn.live_progress);
turn.live_progress.waitSeconds=40;
assert.strictEqual(select(turn,{progress:{...turn.live_progress,waitSeconds:20}}),turn.live_progress);
const app=fs.readFileSync('frontend/angelos/app.js','utf8');
scope.activityDisplay={progress:t=>select(t,{progress:live})};scope.online=true;
scope.active=t=>t.stage==='executing'||t.stage==='preparing';scope.esc=s=>String(s??'');
vm.runInContext(app.slice(app.indexOf('function progressHTML('),app.indexOf('function taskOverviewHTML(')),scope);
assert(scope.progressHTML(turn).includes('正在搜索项目文件'));
assert(scope.progressHTML(turn).includes('读取代码'));
assert(scope.progressHTML(turn).includes('发现旧版本未加载'));
assert(scope.progressHTML({...turn,stage:'finished'}).includes('发现旧版本未加载'));
assert(!scope.progressHTML({...turn,stage:'finished'}).includes('role="status"'));
scope.activityDisplay=undefined;
assert(scope.progressHTML({stage:'preparing'}).includes('正在接收要求'));
assert(scope.progressHTML({stage:'executing'}).includes('正在连接进度记录'));
assert.strictEqual(scope.progressHTML({stage:'finished'}),'');
console.log('SSE progress rendering and stale response checks passed');
'''
    result = subprocess.run(['node', '-e', script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
