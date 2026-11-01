"""Optional local locator. Queries never rebuild or wait for archive indexing.

Only derived tables have triggers. A missing FTS table cannot break canonical
memory writes. Read-time joins and canonical validation reject stale entries;
bounded SQL fallback can find new/unindexed/corrected sources immediately.
"""
from __future__ import annotations

import hashlib
import heapq
import re
import sqlite3
import threading
import time

from .contracts import Deadline
from .sources import connection
from .jobs import SingleFlight
from .contracts import digest
from ..semantic_scan import cosine_scores
from ..ranked_candidates import fuse_ranked_handles
from .. import retrieval_trace


def query_parts(query):
    return tuple(_parts(str(query)[:1000], 48))


def _parts(text, limit):
    parts = {}
    for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_./-]{2,}", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for i in range(len(token)-1):
                parts[token[i:i+2]] = None
                if i+2 < len(token):
                    parts[token[i:i+3]] = None
                if len(parts)>=limit:
                    return list(parts)[:limit]
        else:
            parts[token.lower()] = None
        if len(parts)>=limit:
            break
    return list(parts)[:limit]


def encoded_terms(text):
    # ASCII encodings let stock SQLite FTS5 index Chinese bigrams as whole
    # tokens without an extra tokenizer dependency. Originals remain separate.
    return " ".join("t"+p.encode("utf-8").hex() for p in _parts(str(text)[:50000],4096))


class ArchiveLocator:
    def __init__(self, store, policy):
        self.store, self.policy = store, policy
        self._stop = threading.Event()
        self._worker = None
        self._maintenance_lock = threading.Lock()
        self._watch_lock = threading.Lock()
        self._watch = sqlite3.connect(store.path, timeout=.1, check_same_thread=False)
        self._watch.execute("PRAGMA query_only=ON")
        self._ready_version = None
        self._pass_version = None
        self._pass_dirty = False
        self._pass_finished = set()
        self._encoding = SingleFlight(entries=32, ttl=120)
        self.last_status = {"state":"not_started"}
        try:
            self._install()
        except sqlite3.Error as exc:
            self.last_status = {"state":"fallback", "error_type":type(exc).__name__}
        if not hasattr(store, "_projection_observers"):
            store._projection_observers = []
        store._projection_observers.append(self)

    def before_commit(self):
        return self._version()

    def after_commit(self, before, source_write):
        # Audit/usage writes do not dirty a complete source index. Unknown
        # writes from other store instances still change data_version and force
        # conservative fallback. These callbacks never alter canonical data.
        after = self._version()
        if source_write:
            self._ready_version = None
            self._pass_dirty = True
        elif before == self._ready_version:
            self._ready_version = after
        if not source_write and before == self._pass_version:
            self._pass_version = after

    def _version(self):
        with self._watch_lock:
            return self._watch.execute("PRAGMA data_version").fetchone()[0] if self._watch else -1

    def _install(self):
        # No backfill here, and no production service is started by construction.
        with self.store._lock, self.store._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_projection_documents (
                    id INTEGER PRIMARY KEY, record_key TEXT UNIQUE NOT NULL,
                    scope TEXT NOT NULL, body TEXT NOT NULL, tokens TEXT NOT NULL DEFAULT '');
                CREATE INDEX IF NOT EXISTS idx_projection_documents_scope
                    ON memory_projection_documents(scope);
                CREATE TABLE IF NOT EXISTS memory_projection_watermarks (
                    kind TEXT PRIMARY KEY, cursor INTEGER NOT NULL DEFAULT 0);
            """)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_projection_documents)")}
            if "tokens" not in columns:
                conn.execute("ALTER TABLE memory_projection_documents ADD COLUMN tokens TEXT NOT NULL DEFAULT ''")
            old = conn.execute("SELECT sql FROM sqlite_master WHERE name='memory_projection_fts'").fetchone()
            if old and "trigram" in str(old[0]):
                # Rebuild only the disposable index, incrementally; no canonical
                # writes or full backfill occur while constructing the service.
                conn.executescript("DROP TRIGGER IF EXISTS projection_doc_insert; DROP TRIGGER IF EXISTS projection_doc_delete; "
                    "DROP TRIGGER IF EXISTS projection_doc_update; DROP TABLE memory_projection_fts; "
                    "UPDATE memory_projection_documents SET tokens=''; DELETE FROM memory_projection_watermarks;")
            conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_projection_fts
                    USING fts5(tokens, tokenize='unicode61');
                CREATE TRIGGER IF NOT EXISTS projection_doc_insert AFTER INSERT ON memory_projection_documents BEGIN
                    INSERT INTO memory_projection_fts(rowid,tokens) VALUES(new.id,new.tokens);
                END;
                CREATE TRIGGER IF NOT EXISTS projection_doc_delete AFTER DELETE ON memory_projection_documents BEGIN
                    DELETE FROM memory_projection_fts WHERE rowid=old.id;
                END;
                CREATE TRIGGER IF NOT EXISTS projection_doc_update AFTER UPDATE ON memory_projection_documents BEGIN
                    DELETE FROM memory_projection_fts WHERE rowid=old.id;
                    INSERT INTO memory_projection_fts(rowid,tokens) VALUES(new.id,new.tokens);
                END;
            """)

    def maintain(self):
        """Incremental, restartable pass. Never called from a search request."""
        if not self._maintenance_lock.acquire(blocking=False):
            return {"state":"busy","written":0}
        try:
            before = self._version()
            if self._ready_version == before:
                return {"state":"ready", "written":0}
            if self._pass_version is not None and before != self._pass_version:
                self._pass_dirty = True
            deadline = Deadline(1000)
            written = 0
            with connection(self.store, deadline, write=True) as conn:
                for kind, table, pk in (("event","memory_events","event_id"),("fact","memory_facts","fact_id")):
                    row=conn.execute("SELECT cursor FROM memory_projection_watermarks WHERE kind=?",(kind,)).fetchone()
                    cursor=int(row[0]) if row else 0
                    rows=conn.execute(f"SELECT rowid AS scan_id,{pk} AS rid,scope,text,status FROM {table} WHERE rowid>? ORDER BY rowid LIMIT ?",
                                      (cursor,self.policy.maintenance_batch)).fetchall()
                    for row in rows:
                        deadline.check()
                        key=kind+":"+row["rid"]
                        if row["status"] == "active":
                            conn.execute("""INSERT INTO memory_projection_documents(record_key,scope,body,tokens) VALUES(?,?,?,?)
                                ON CONFLICT(record_key) DO UPDATE SET scope=excluded.scope,body=excluded.body,tokens=excluded.tokens
                                WHERE scope!=excluded.scope OR body!=excluded.body OR tokens!=excluded.tokens
                                  OR NOT EXISTS (SELECT 1 FROM memory_projection_fts WHERE rowid=memory_projection_documents.id)""",
                                (key,row["scope"],row["text"],encoded_terms(row["text"])))
                        else:
                            conn.execute("DELETE FROM memory_projection_documents WHERE record_key=?",(key,))
                        written+=1
                    # Cycle to refresh corrected old sources after the initial pass.
                    cursor = int(rows[-1]["scan_id"]) if rows else 0
                    if not rows:
                        self._pass_finished.add(kind)
                    conn.execute("INSERT INTO memory_projection_watermarks VALUES(?,?) ON CONFLICT(kind) DO UPDATE SET cursor=excluded.cursor",(kind,cursor))
            self._pass_version = self._version()
            ready = self._pass_finished == {"event", "fact"}
            if ready:
                if not self._pass_dirty:
                    self._ready_version = self._pass_version
                self._pass_dirty = False
                self._pass_finished.clear()
            self.last_status={"state":"ready" if self._ready_version == self._pass_version else "indexing","written":written}
        except (sqlite3.Error, TimeoutError) as exc:
            self.last_status={"state":"fallback","error_type":type(exc).__name__,"written":0}
        finally:
            self._maintenance_lock.release()
        return dict(self.last_status)

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        def run():
            delay=.1
            while not self._stop.wait(delay):
                result=self.maintain()
                delay=1 if result.get("written") else (10 if result["state"]=="fallback" else 3)
        self._worker=threading.Thread(target=run,name="projection-archive",daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        self._encoding.stop()
        if self in getattr(self.store, "_projection_observers", ()):
            self.store._projection_observers.remove(self)
        if self._worker:
            self._worker.join(timeout=1.5)
        with self._watch_lock:
            if self._watch:
                self._watch.close()
                self._watch = None

    def invalidate(self, *, scope, record_ids):
        try:
            with connection(self.store, Deadline(200), write=True) as conn:
                if record_ids:
                    conn.executemany("DELETE FROM memory_projection_documents WHERE scope=? AND record_key=?",[(scope,rid) for rid in record_ids])
                else:
                    conn.execute("DELETE FROM memory_projection_documents WHERE scope=?" if scope else "DELETE FROM memory_projection_documents",
                                 (scope,) if scope else ())
        except (sqlite3.Error, TimeoutError):
            # Source revalidation still rejects revoked entries even if cleanup fails.
            self.last_status={"state":"cleanup_pending"}

    def search(self, request, deadline, *, limit):
        parts=query_parts(request.search_query)
        if not parts:
            return (), {"state":"no_query_terms","complete":False}
        handles=[]; coverage={"state":"indexed","complete":False,"candidate_limit":limit,
                              "source_types":["event","fact"],"time_filter":[request.start_at,request.end_at],
                              "maintenance_in_request":0,
                              "lexical_coverage":"CJK_2_3grams_ASCII_terms_source_prefix_50000_chars_4096_tokens"}
        scopes=','.join('?' for _ in request.allowed_scopes)
        indexed=list(parts)
        index_ready = self._ready_version == self._version()
        coverage["index_ready"] = index_ready
        with connection(self.store, deadline) as conn:
            if indexed:
                match=' OR '.join('"t'+p.encode("utf-8").hex()+'"' for p in indexed)
                try:
                    rows=conn.execute(f"""SELECT d.record_key,d.scope FROM memory_projection_fts f
                        JOIN memory_projection_documents d ON d.id=f.rowid
                        LEFT JOIN memory_events e ON d.record_key LIKE 'event:%' AND e.event_id=substr(d.record_key,7)
                        LEFT JOIN memory_facts t ON d.record_key LIKE 'fact:%' AND t.fact_id=substr(d.record_key,6)
                        WHERE memory_projection_fts MATCH ? AND d.scope IN ({scopes})
                        AND ((e.status='active' AND d.body=e.text AND d.scope=e.scope)
                             OR (t.status='active' AND d.body=t.text AND d.scope=t.scope))
                        AND (? IS NULL OR coalesce(e.created_at,t.created_at)>=?)
                        AND (? IS NULL OR coalesce(e.created_at,t.created_at)<?)
                        ORDER BY rank LIMIT ?""",
                        (match,*request.allowed_scopes,request.start_at,request.start_at,request.end_at,request.end_at,limit+1)).fetchall()
                    handles.extend((r[0],r[1],"") for r in rows[:limit])
                    coverage["truncated"]=len(rows)>limit
                except sqlite3.OperationalError:
                    deadline.check()
                    coverage["state"]="index_unavailable"
            # Corrected and not-yet-indexed originals must be discoverable too.
            # This is relevance filtering before LIMIT, with a hard SQL deadline.
            conditions=' OR '.join('instr(lower(text),?)>0' for _ in parts)
            fallback_tables = () if index_ready and len(indexed)==len(parts) and coverage["state"]=="indexed" else (
                ("event","memory_events","event_id"),("fact","memory_facts","fact_id"))
            coverage["source_fallback_scan"] = bool(fallback_tables)
            for kind,table,pk in fallback_tables:
                deadline.check()
                rows=conn.execute(f"""SELECT {pk},scope,created_at FROM {table}
                    WHERE scope IN ({scopes}) AND status='active' AND ({conditions})
                    AND (? IS NULL OR created_at>=?) AND (? IS NULL OR created_at<?)
                    ORDER BY created_at DESC LIMIT ?""",
                    (*request.allowed_scopes,*(p.lower() for p in parts),request.start_at,request.start_at,
                     request.end_at,request.end_at,limit+1)).fetchall()
                coverage[kind+"_truncated"]=len(rows)>limit
                handles.extend((kind+":"+r[0],r[1],"") for r in rows[:limit])
            coverage["state"]="source_checked" if coverage["state"]=="indexed" else "bounded_source_fallback"
            coverage["candidate_set_untruncated"]=not any(coverage.get(k) for k in ("truncated","event_truncated","fact_truncated"))
        unique=tuple(dict.fromkeys(handles))
        coverage["candidate_count"]=len(unique)
        if len(unique)>limit:
            coverage.update(candidate_set_untruncated=False,truncated=True)
        return unique[:limit],coverage

    def discover(self, request, deadline, *, limit):
        """One shared bounded lexical/vector locator contract for all readers."""
        retrieval_trace.query("archive.request", request.search_query, limit=limit)
        rankings, coverage, gaps = [], {}, []
        for name, operation, count in (("lexical", self.search, limit),
                                       ("vector", self.semantic, min(12, limit))):
            started = time.monotonic()
            try:
                handles, status = operation(request, deadline, limit=count)
                retrieval_trace.observe("archive." + name, handles,
                    vector_truncated=bool(status.get("vector_truncated")),
                    lexical_truncated=bool(status.get("truncated") or status.get("event_truncated")
                                           or status.get("fact_truncated")),
                    state=str(status.get("state", status.get("semantic", ""))))
                rankings.append(handles)
                coverage.update(status)
                coverage[name+"_candidate_count"] = len(handles)
                if name == "vector" and status.get("vector_truncated"):
                    gaps.append({"reason": "budget_exhausted", "stage": "vector_discovery"})
            except (sqlite3.Error, TimeoutError, ValueError) as exc:
                rankings.append(())
                retrieval_trace.observe("archive." + name, reason=type(exc).__name__)
                gaps.append({"reason":"budget_exhausted" if isinstance(exc, TimeoutError) or not deadline.remaining
                             else "source_read_failed", "stage":name+"_discovery", "error_type":type(exc).__name__})
            coverage[name+"_ms"] = round((time.monotonic()-started)*1000, 3)
        handles, count = fuse_ranked_handles(*rankings, limit=limit)
        retrieval_trace.observe("archive.fused", handles, candidate_count=count, limit=limit)
        coverage.update(fusion="reciprocal_rank_v1", discovered_unique=count,
                        fusion_truncated=count > len(handles), complete=False,
                        absence_means="unknown", maintenance_in_request=0)
        return handles, coverage, gaps

    def semantic(self, request, deadline, *, limit):
        """Search retained vectors across time, never embed/rebuild corpus here."""
        provider=self.store._embedding_provider
        if not self.store.semantic_enabled or provider is None:
            return (), {"semantic":"unavailable"}
        deadline.check()
        def encode():
            acquired=self.store._embedding_lock.acquire(timeout=.1)
            if not acquired:
                raise TimeoutError("encoder_busy")
            try:
                return provider.embed_text(request.search_query)
            finally:
                self.store._embedding_lock.release()
        vector, encoding = self._encoding.run(digest([self.store.embedding_model_key, request.search_query]),
                                              encode, min(.8, deadline.remaining))
        if not vector:
            return (), {"semantic":"encoder_"+encoding["status"]}
        deadline.check()
        model_key=self.store.embedding_model_key
        scope_sql=','.join('?' for _ in request.allowed_scopes)
        top=[]; scanned=0; truncated=False
        # 固定行数前缀与相关性无关。复用 Store 的批量余弦计算，遍历授权范围，
        # 内存只保留一批向量及 top-k；整体工作仍受本次 Deadline 约束。
        # 旧 vector_scan_limit 参数保留兼容，现作为每批取数上限，最多 256 条。
        batch_size = min(256, self.policy.vector_scan_limit)
        try:
            with connection(self.store,deadline) as conn:
                cursor=conn.execute(f"""SELECT v.*,coalesce(e.text,t.text) AS body,
                coalesce(e.created_at,t.created_at) AS source_time FROM memory_embeddings v
                LEFT JOIN memory_events e ON v.record_type='event' AND v.record_id=e.event_id
                LEFT JOIN memory_facts t ON v.record_type='fact' AND v.record_id=t.fact_id
                WHERE v.scope IN ({scope_sql}) AND v.model_key=?
                AND ((e.status='active' AND e.scope=v.scope) OR (t.status='active' AND t.scope=v.scope))
                AND (? IS NULL OR coalesce(e.created_at,t.created_at)>=?)
                AND (? IS NULL OR coalesce(e.created_at,t.created_at)<?)
                ORDER BY v.embedding_id""",
                (*request.allowed_scopes,model_key,request.start_at,request.start_at,request.end_at,request.end_at))
                try:
                    while True:
                        deadline.check()
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        eligible = [row for row in rows if
                            hashlib.sha256(row["body"].encode()).hexdigest() == row["content_hash"]]
                        for row, similarity in zip(eligible, cosine_scores(vector, eligible)):
                            if similarity is None or similarity < self.store.semantic_min_score:
                                continue
                            entry=(similarity,row["record_type"]+":"+row["record_id"],row["scope"])
                            if len(top)<limit:
                                heapq.heappush(top,entry)
                            elif entry>top[0]:
                                heapq.heapreplace(top,entry)
                        scanned += len(rows)
                finally:
                    cursor.close()
        except TimeoutError:
            truncated = True
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_INTERRUPT:
                raise
            truncated = True
        # 超时只返回已比较部分的定位符，并显式报告缺口；来源读取仍必须再次验证。
        return tuple((rid,scope,"") for _,rid,scope in sorted(top,reverse=True)),{
            "semantic":"retained_vectors","vector_rows":scanned, "vector_batch_size":batch_size,
            "vector_truncated":truncated,"vector_coverage":"indexed_sources_only"}
