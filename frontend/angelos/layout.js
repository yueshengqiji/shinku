/* 工作台分栏尺寸由浏览器持有；拖动只改变布局，不发送请求或操作任务。
 * 指针捕获保证离开细边框后仍可拖动；取消/失焦结束拖动，键盘可替代鼠标。
 */
(()=>{
 const key='angelos:panel-sizes',root=document.documentElement;
 let saved={};try{saved=JSON.parse(localStorage.getItem(key))||{}}catch{}
 const clamp=(v,min,max)=>Math.min(Math.max(v,min),Math.max(min,max));
 const finite=(v,fallback)=>Number.isFinite(v)?v:fallback;
 let drag=null;
 const handles=[];
 function range(name){
  const w=window.innerWidth,h=window.innerHeight,side=w>760?clamp(finite(saved.side,242),200,Math.min(420,w*.35)):0;
  if(name==='side')return {value:side,min:200,max:Math.min(420,w*.35),axis:'x',sign:1};
  if(name==='tools')return {value:clamp(finite(saved.tools,390),280,Math.min(720,w-side-180)),min:280,max:Math.min(720,w-side-180),axis:'x',sign:-1};
  if(w<=1000)return {value:clamp(finite(saved.editorHeight,h*.44),150,h-250),min:150,max:Math.max(150,h-250),axis:'y',sign:1};
  return {value:clamp(finite(saved.editor,Math.min(440,(w-side)*.48)),280,w-side-300),min:280,max:Math.max(280,w-side-300),axis:'x',sign:-1};
 }
 function apply(){
  if(window.innerWidth>760)root.style.setProperty('--side',range('side').value+'px');else root.style.removeProperty('--side');
  root.style.setProperty('--tools-width',range('tools').value+'px');
  const r=range('editor');root.style.setProperty(r.axis==='y'?'--editor-height':'--editor-width',r.value+'px');
  for(const {name,node} of handles){const r=range(name);node.setAttribute('aria-orientation',r.axis==='x'?'vertical':'horizontal');for(const [k,v] of Object.entries({min:r.min,max:r.max,now:r.value}))node.setAttribute('aria-value'+k,String(Math.round(v)));}
 }
 function write(name,value){const r=range(name);saved[name==='editor'&&r.axis==='y'?'editorHeight':name]=clamp(value,r.min,r.max);apply()}
 function persist(){try{localStorage.setItem(key,JSON.stringify(saved))}catch{}}
 function finish(){if(!drag)return;const {node,pointer}=drag;drag=null;document.body.classList.remove('resizing-panels');if(node.hasPointerCapture(pointer))node.releasePointerCapture(pointer);persist()}
 for(const [name,id,label] of [['side','sidebar','调整会话侧栏宽度'],['editor','changes-panel','调整代码与对话分栏大小'],['tools','tools-panel','调整工具调用栏宽度']]){
  const panel=document.getElementById(id);if(!panel)continue;
  const node=document.createElement('div');node.className='panel-resizer resizer-'+name;node.tabIndex=0;node.setAttribute('role','separator');node.setAttribute('aria-label',label);node.setAttribute('aria-controls',id);node.title='拖动调整大小；双击恢复默认；方向键微调';panel.appendChild(node);handles.push({name,node});
  node.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();finish();const r=range(name);drag={name,node,pointer:e.pointerId,start:r.axis==='x'?e.clientX:e.clientY,value:r.value,axis:r.axis,sign:r.sign};node.setPointerCapture(e.pointerId);document.body.classList.add('resizing-panels')});
  node.addEventListener('pointermove',e=>{if(!drag||drag.node!==node||drag.pointer!==e.pointerId)return;write(name,drag.value+((drag.axis==='x'?e.clientX:e.clientY)-drag.start)*drag.sign)});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])node.addEventListener(event,finish);
  node.addEventListener('dblclick',()=>{delete saved[name];if(name==='editor')delete saved.editorHeight;apply();persist()});
  node.addEventListener('keydown',e=>{const r=range(name),directions=r.axis==='x'?{ArrowLeft:-1,ArrowRight:1}:{ArrowUp:-1,ArrowDown:1};if(e.key==='Home'){e.preventDefault();write(name,r.min)}else if(e.key==='End'){e.preventDefault();write(name,r.max)}else if(e.key in directions){e.preventDefault();write(name,r.value+directions[e.key]*r.sign*(e.shiftKey?40:10))}else return;persist()});
 }
 window.addEventListener('blur',finish);window.addEventListener('resize',()=>{finish();apply()});apply();
})();
