"""两处阶段展示在真实 Chromium 中的只读、转义和窄屏布局验收。"""
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('surface', ['angelos', 'control-center'])
def test_phase_component_in_browser(tmp_path, surface):
    pw = pytest.importorskip('playwright.sync_api')
    perf = {'available': True, 'truncated': True, 'delegatedTaskIds': ['agent:synthetic'], 'phases': [
        dict(phase=name, label=label, totalMs=total, samples=2, measured=2, p50Ms=total/4,
             p95Ms=total*3/4, requests=requests, requestMs=total/2 if requests else None,
             knownInputTokens=1200 if requests else None, knownOutputTokens=80 if requests else None,
             incompleteUsage=0, failed=0, requestFailed=0, unsettled=0, requestUnsettled=0)
        for name,label,total,requests in [('planning','执行规划',8000,2),('tool','工具调用',1200,0),
            ('verification','目标核验',40,0),('expression','结果表达',2400,1)]]}
    errors = []
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1440,'height':900})
        page.on('pageerror', lambda error: errors.append(str(error)))
        # 组件使用真实脚本与样式；所有输入为合成快照，不连接任何运行服务。
        page.set_content('<main style="max-width:1250px;margin:24px"><h2>阶段统计 · 离线合成示例</h2><div id="sample"></div></main>')
        page.add_style_tag(content=(ROOT/'frontend'/surface/'app.css').read_text())
        if surface == 'angelos':
            page.add_script_tag(content=(ROOT/'frontend/angelos/usage.js').read_text())
            page.evaluate('''perf => {
                const summary={requests:3,totalTokens:1280,knownTotalTokens:1280,cacheHitRatio:null,
                    totals:{input_tokens:1200,output_tokens:80,cache_hit_tokens:null},
                    known:{input_tokens:1200,output_tokens:80,cache_hit_tokens:null},performance:perf};
                const display=window.AngelosUsage.create({api:async()=>({usage:{available:true,
                    conversationId:'fixture',sampledAt:0,summary,turns:{}}}),
                    changed:()=>{document.getElementById('sample').innerHTML=display.summaryHTML()}});
                display.select('fixture',true);
            }''', perf)
        else:
            # 中枢脚本在初始化时绑定现有表单；保留真实 DOM，再聚焦本轮组件。
            # 首轮缺少这些元素导致脚本提前中止，不能把测试壳错误算作页面通过。
            html=(ROOT/'frontend/control-center/index.html').read_text()
            html=re.sub(r'<script\b[^>]*>.*?</script>|<link\b[^>]*>', '', html, flags=re.S)
            page.set_content(html)
            page.evaluate('''() => {
                for(const child of document.body.children)child.style.display='none';
                document.body.insertAdjacentHTML('afterbegin','<main style="max-width:1250px;margin:24px"><h2>阶段统计 · 离线合成示例</h2><div id="sample"></div></main>');
            }''')
            page.add_style_tag(content=(ROOT/'frontend/control-center/app.css').read_text())
            page.add_script_tag(content='''if(!crypto.randomUUID)crypto.randomUUID=()=>"fixture";''')
            source=(ROOT/'frontend/control-center/app.js').read_text().split('// 锚点使用浏览器原生历史与滚动')[0]
            page.add_script_tag(content=source)
            page.evaluate('perf=>document.getElementById("sample").innerHTML=phasePerformanceHTML(perf)',perf)
        page.get_by_text('阶段耗时与模型请求', exact=True).click()
        assert page.locator('.phase-table tbody tr').count() == 4
        assert '不可相加' in page.locator('#sample').inner_text()
        assert '采集不完整' in page.locator('#sample').inner_text()
        folder=Path(os.environ.get('TIMING_SCREENSHOT_DIR', str(tmp_path)))
        folder.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(folder/(surface+'-wide.png')))
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(folder/(surface+'-narrow.png')))
        assert not errors, errors
        browser.close()
