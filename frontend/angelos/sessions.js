'use strict';
/* 终态来自宿主，不从回答猜测。每个任务尝试只提醒一次；首次访问只建立基线。
   已观察的运行任务即使页面重载/隐藏后完成也会留下未读提醒。只保存ID/标题/状态。 */
class AngelosTaskNotices {
  constructor(saved,persist){this.persist=persist;this.state=saved&&saved.version===2?saved:{version:2,initialized:false,seen:{},notices:[]};this.fresh=[];
    // 旧提醒只有状态，没有具体缺项；改为中性的结束提示，保留已读标记与去重身份。
    for(const n of this.state.notices)if(n.label==='任务部分完成')n.label='本轮已结束';
  }
  observe(conversations){
    const labels={completed:'任务完成',replied:'已回复',failed:'任务失败',interrupted:'任务中断',partial:'本轮已结束',blocked:'任务受阻',waiting_user:'等待主管回答',canceled:'任务已取消',unverified:'任务未通过验收'};
    for(const c of conversations){
      const seen=this.state.seen[c.id]||{};const turns=c.taskStates||(c.latestTask?[c.latestTask]:[]);
      const next={};
      for(const t of turns){const key=JSON.stringify([c.id,t.id,t.attempt]),status=t.status,previous=seen[t.id];
        // worker尚未结束/结果尚未归档时，不能依据中间视图提前发终态提醒。
        const settled=!['preparing','submitting','executing'].includes(t.stage);
        const signature=JSON.stringify([key,status,settled]);
        if(this.state.initialized&&settled&&labels[status]&&previous!==signature){
          const notice={key:JSON.stringify([key,status]),cid:c.id,title:c.title,label:labels[status],at:Date.now()/1000};
          if(!this.state.notices.some(n=>n.key===notice.key)){this.state.notices.push(notice);this.fresh.push(notice);}
        }
        next[t.id]=signature;
      }
      this.state.seen[c.id]=next;
    }
    const ids=new Set(conversations.map(c=>c.id));
    for(const id of Object.keys(this.state.seen))if(!ids.has(id))delete this.state.seen[id];
    this.state.notices=this.state.notices.filter(n=>ids.has(n.cid)).slice(-100);
    this.state.initialized=true;this.persist(this.state);
  }
  unread(){return this.state.notices.filter(n=>!n.read).slice().reverse();}
  takeFresh(){const rows=this.fresh;this.fresh=[];return rows;}
  read(key){for(const n of this.state.notices)if(n.key===key)n.read=true;this.persist(this.state);}
  readAll(){for(const n of this.state.notices)n.read=true;this.persist(this.state);}
}
