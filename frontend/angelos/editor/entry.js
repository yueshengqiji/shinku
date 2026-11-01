/* 编辑器只持有未保存草稿；实际文件读取/版本校验/保存由宿主 API 负责。 */
import {basicSetup} from 'codemirror';
import {EditorView, keymap} from '@codemirror/view';
import {EditorState, Compartment} from '@codemirror/state';
import {HighlightStyle, syntaxHighlighting} from '@codemirror/language';
import {tags} from '@lezer/highlight';
import {javascript} from '@codemirror/lang-javascript';
import {python} from '@codemirror/lang-python';
import {json} from '@codemirror/lang-json';
import {html} from '@codemirror/lang-html';
import {css} from '@codemirror/lang-css';
import {markdown} from '@codemirror/lang-markdown';
import {cpp} from '@codemirror/lang-cpp';
import {java} from '@codemirror/lang-java';
import {rust} from '@codemirror/lang-rust';
import {yaml} from '@codemirror/lang-yaml';
import {MergeView} from '@codemirror/merge';
// 关键字、函数、类型与字面量各用独立色相，注释保留可读对比；不改变语法或编辑行为。
const colors=HighlightStyle.define([
 {tag:tags.keyword,color:'#c6a0e8'}, {tag:[tags.typeName,tags.className],color:'#66c9ce'},
 {tag:tags.function(tags.variableName),color:'#82aaff'}, {tag:tags.function(tags.propertyName),color:'#82aaff'},
 {tag:[tags.variableName,tags.propertyName],color:'#d5dce7'}, {tag:[tags.string,tags.special(tags.string)],color:'#a3c78f'},
 {tag:tags.comment,color:'#929ba9'}, {tag:[tags.number,tags.bool],color:'#e5b37c'},
 {tag:tags.operator,color:'#e5e2dc'}, {tag:tags.tagName,color:'#c6a0e8'}]);
const theme=EditorView.theme({
 '&':{height:'100%',backgroundColor:'#090b10',color:'#e5e2dc',fontSize:'15px'},
 '.cm-scroller':{overflow:'auto',fontFamily:'"Cascadia Code", "Noto Sans Mono", monospace',lineHeight:'1.65'},
 '.cm-content':{padding:'10px 0',caretColor:'#e5e5e5'},
 '.cm-gutters':{backgroundColor:'#090b10',color:'#929ba9',borderRight:'1px solid #393939'},
 '.cm-activeLine':{backgroundColor:'#ffffff06'}, '.cm-activeLineGutter':{backgroundColor:'#172033'},
 '&.cm-focused .cm-selectionBackground, .cm-selectionBackground':{backgroundColor:'#264f78'},
 '.cm-cursor':{borderLeftColor:'#eee'}, '.cm-panels':{backgroundColor:'#11151d',color:'#ccc'},
 '.cm-tooltip':{backgroundColor:'#1a202a',border:'1px solid #555'}},{dark:true});
function language(path){const ext=path.split('.').pop().toLowerCase();return ({py:python,js:javascript,mjs:javascript,cjs:javascript,ts:()=>javascript({typescript:true}),tsx:()=>javascript({typescript:true,jsx:true}),jsx:()=>javascript({jsx:true}),json,html,htm:html,css,md:markdown,yaml,yml:yaml,c:cpp,h:cpp,cpp,hpp:cpp,java,rs:rust})[ext]?.()||[]}
export function create(parent,{path,content,original='',diff=false,onChange=()=>{},onSave=()=>{},onSelection=()=>{}}){
 const access=new Compartment();
 // 选区归编辑器所有，失焦仍保留；两栏必须标明来源，不能把旧版本当当前文件。
 function selected(state,side){const r=state.selection.main;if(r.empty)return null;const start=state.doc.lineAt(r.from),end=state.doc.lineAt(r.to);return {side,text:state.sliceDoc(r.from,r.to),startLine:start.number,startColumn:r.from-start.from+1,endLine:end.number,endColumn:r.to-end.from+1}}
 const selectionListener=side=>EditorView.updateListener.of(u=>{if(u.selectionSet||u.docChanged||u.focusChanged&&u.view.hasFocus)onSelection(selected(u.state,side))});
 const common=[basicSetup,EditorState.lineSeparator.of(content.includes('\r\n')?'\r\n':'\n'),theme,syntaxHighlighting(colors),language(path),EditorView.contentAttributes.of({'aria-label':'代码编辑器',tabindex:'0'})];
 const extensions=[...common,access.of([EditorState.readOnly.of(true),EditorView.editable.of(false)]),
  EditorView.updateListener.of(u=>{if(u.docChanged)onChange(u.state.sliceDoc())}),selectionListener('current'),keymap.of([{key:'Mod-s',run:()=>{onSave();return true}}])];
 const merge=diff?new MergeView({parent,a:{doc:original,extensions:[...common,EditorState.readOnly.of(true),EditorView.editable.of(false),selectionListener('original')]},b:{doc:content,extensions},highlightChanges:true,gutter:true,diffConfig:{scanLimit:2000,timeout:150}}):null;
 const view=merge?merge.b:new EditorView({parent,doc:content,extensions});
 return {getValue:()=>view.state.sliceDoc(),setEditable(value){view.dispatch({effects:access.reconfigure([EditorState.readOnly.of(!value),EditorView.editable.of(value)])})},destroy(){if(merge)merge.destroy();else view.destroy()},focus(){view.focus()}};
}
