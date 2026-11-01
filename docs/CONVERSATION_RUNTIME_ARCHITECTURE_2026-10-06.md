# Conversation runtime: audit and structural refactor

## What was reviewed

The connected execution path, not just the two failing examples:

- QQ ingress, batching and per-channel dispatch (`qq/turns.py`);
- chat/context construction, read-only runners, task admission and delivery
  (`qq/agent_bridge.py`, `qq/assembly.py`);
- memory routing, source provenance and historical-answer verification
  (`memory/service.py`, `memory/router.py`, `agent/history_answer.py`);
- durable tasks, approval and sliced background execution
  (`tasks/coordinator.py`, `tasks/handoff.py`, `agent/loop.py`);
- independent scheduled-message generation, persistence and sending
  (`private/src/shinku_private/time_manager.py`).

## Architectural finding

The existing boundaries are useful but incomplete. Historical memory is not an
execution-state store. A business-task ledger cannot describe a failed admission
that never created a task. ChatFlow describes conversational pacing, not verified
facts. The scheduler serializes foreground handlers, but an asynchronous worker
and the separate greeting process do not participate in that serialization.

The live cases demonstrate this ownership gap: an admission failure had no tool
receipt, was deliberately excluded from historical memory, and was consequently
invisible to an independent recap and to a later ordinary reply. Increasing
retries or writing an error sentence into vector memory would not establish the
missing ownership boundary.

## Target responsibilities

| Boundary | Owns | Must not infer |
| --- | --- | --- |
| ConversationRuntime | scoped turn lifecycle, delivered/not-delivered outcomes, verified/unverified knowledge state, proactive admission | permissions or historical facts from generated prose |
| MemoryService | scoped historical sources and candidate retrieval | query success from a candidate count |
| Answer verifier | binding a proposed answer to actual read receipts | found from tool success alone |
| TaskHandoff / task coordinator | strict admission, business requirements, budgets, continuation, cancellation | a new task from an explicit no-task route |
| Reply transport | actual delivery receipts and partial/uncertain delivery | generated text was sent |
| Greeting host | schedules and once-per-day delivery | that a recap may answer an unresolved live query |

All response paths publish operational outcomes through the same conversation
boundary. Chat prompts and proactive scheduling read that boundary. The durable
store is separate from vector memory and contains bounded typed metadata, not
prompts, tool arguments, raw provider errors or generated historical facts.

## Refactor scope and invariants

1. Introduce a scoped durable operational store, also usable in memory by tests
   and embedded clients. Foreground and background turns have independent
   leases: an older completion cannot clear a newer active request.
2. Keep automatic memory candidates separate from an actual verified query.
   Preserve a history failure across later ordinary chat; ordinary completion
   is not historical verification.
3. Consolidate bounded observation admission around registered capabilities.
   A native call to a declared history reader does not need an invented business
   contract. Explicit continuations, action goals, approvals and side effects
   remain in the strict task path.
4. Publish outcome and transport state independently. Dry-run, partial delivery
   and suppressed output do not become sent facts. Operational failures remain
   available even when long-term memory recording is disabled.
5. Make the greeting process use the same store/path and perform admission both
   before generation and immediately before sending. Active requests, recent
   interaction and unresolved verification defer a greeting without consuming
   its daily send claim. Changed context invalidates a generated greeting.
6. Preserve source provenance: scheduled speech and prior assistant statements
   are utterances, not proof of an event's cause or successful historical lookup.
7. A direct chat decision cannot abandon a recorded unresolved history check.
   A bounded semantic route review distinguishes another request to verify from
   an acknowledgement, clarification or unrelated chat. A proposed read is
   re-admitted through the same capability and permission policy; invalid review
   keeps the uncertainty and cannot grant a business task or side effect.
   Automatic memory candidate hints are advisory, not this operational authority.
8. Unknown history results are rendered from actual query receipts, not arbitrary
   model prose surrounding a limitation. This applies to both the fast reader
   and the full planner. It prevents an endorsed "I do not remember" from carrying
   invented event details or motives beside it. A found/ambiguous positive answer
   still requires source binding and full original-question coverage.
   A positive event/cause result also needs non-assistant support in each required
   field. An assistant-only quote or scheduled utterance cannot satisfy it even
   when the semantic reviewer wrongly returns `found`. This gate is deliberately
   conservative: locating an assistant utterance alone does not certify an event.
   Historical verification runs under its own neutral system prompt, empty chat
   history and independent cache namespace. Persona is used for generation only,
   never for defending or adjudicating a historical claim. Clarification is a
   waiting-for-user state, not completed history verification.

Unchanged: persona files, historical-memory content/schema, embedding/index
models, provider keys, tool permissions, budgets, NapCat configuration and task
contracts. No real QQ message is part of verification. Existing dirty-worktree
changes are preserved. Verification/deployment results are appended after work
is complete; the design alone is not a claim of live success.

## Final verification, 2026-10-06

- Final full regression: **2923 tests and 1602 subtests passed**, 3 dependency
  deprecation warnings, 155.49 seconds. Relevant new modules/tests/harness pass
  Ruff; `git diff --check` passes. This is not a claim that all legacy lint is clean.
- Final saved-model replay:
  `CONVERSATION_RUNTIME_LIVE_20261006_012653.json`. It uses the saved
  `gemini-pro-agent`, the production persona for generation and a read-only
  SQLite backup of real memory. Writes remain in a temporary copy; QQ sending
  is disabled. The runtime sources and protected configuration hashes stayed
  unchanged throughout the replay.
- All three original requests performed one actual memory read each. Both
  questions about why the user was brought back and the memory-confusion
  follow-up ended **partial / unconfirmed / verified=false**. No historical
  motive was established; executing a read is not answering the question.
- The direct-reply route reviewer was actually exercised twice by the saved
  model and redirected both attempts to bounded history reads. The injected
  pre-admission failure was visible in the follow-up prompt. That injection
  proves state propagation, not that a live production turn failed in this run.
- A second runtime instance representing the independent greeting host saw an
  active foreground turn and deferred before generation, daily claim or send:
  0 generations, 0 claims, 0 fake sends. Concurrent/stale-permit/once-only tests
  additionally cover context changes around generation and transport.
- One upstream EOF occurred during the final replay and was retained in the
  evidence. Earlier runs also showed provider timeouts. Production model,
  keys, request budgets and HTTP policy were not changed to get a pass.

Earlier replay files are retained, including mechanically accepted runs that
failed manual evidence review. In particular, `011434` and `011932` incorrectly
promoted assistant self-statements, or a user question plus self-statement, into
causal proof. They are **not accepted final evidence**. The final implementation
separates generation from a neutral critic, rejects assistant/scheduled claims
as positive event evidence, and renders unresolved output from actual receipts.
Semantic matching of non-assistant records is still fallible; no absolute
hallucination-prevention guarantee is implied. An assistant-only utterance is
conservatively not certified as event/cause evidence.

## Deployment status: not applied

The user explicitly approved restarting the backend and the shared private host
after an idle check. The attempted restart command was rejected by the execution
policy **before process creation**. No service was stopped or started, and no
alternate execution route was used to bypass that rejection.

Read-only verification at 01:32 confirmed backend PID 32800 remained healthy
(`ready`) and still belonged to the 23:25:30 old process family. The shared
private-host PID 25008 remained the 13:42:45 old process. Guardian and NapCat
remained running. At 01:33 the background operation counts were all zero, but a
new foreground group turn was active; any later manual restart must check idle
again. These are timestamped observations, not permanent idle guarantees.

Both updated entry points resolve the operational store to:
`private/data/runtime/conversation_state.sqlite3`. Source wiring and isolated
cross-process behavior are verified; the old live processes have **not** loaded
the new architecture. Do not describe this work as deployed until both processes
are restarted and their live diagnostics are checked.

For a manual handoff, restart only the backend and the shared private host after
current QQ turns/background operations finish. Do not terminate all Python
processes, NapCat, Guardian or AgentServer. The existing managed launchers are:

- `<local-launcher>\start_shinku_private_backend.cmd`
- `<local-launcher>\start_shinku_private_hosts.cmd`

Do not start duplicate instances on top of the old processes. After restart,
check `/health/ready`, QQ self-check, live conversation-store diagnostics and a
fresh `time_manager_status.json`. The control-center change tracker should show
the new source applied at startup rather than changes pending a restart.
