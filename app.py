"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", layout="wide")
st.title("🤖 HITL PR Review Agent")
st.caption("LangGraph Human-in-the-Loop PR Review with SQLite Audit Trail")


# ─── Sidebar — recent sessions & calibration stats ─────────────────────────
with st.sidebar:
    st.header("Recent sessions")

    async def get_recent_sessions():
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 15
                """
            ) as cur:
                return await cur.fetchall()

    async def get_calibration_stats():
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT COUNT(DISTINCT thread_id) AS total_sessions,
                       AVG(confidence)           AS avg_conf,
                       COUNT(CASE WHEN decision = 'approve' THEN 1 END) AS approved_count,
                       COUNT(CASE WHEN decision = 'reject' THEN 1 END) AS rejected_count,
                       COUNT(CASE WHEN decision = 'auto' THEN 1 END) AS auto_count,
                       COUNT(CASE WHEN decision = 'escalate' THEN 1 END) AS escalated_count
                  FROM audit_events
                 WHERE action IN ('route', 'commit', 'auto_approve')
                """
            ) as cur:
                return await cur.fetchone()

    try:
        recent = asyncio.run(get_recent_sessions())
        if recent:
            for r in recent:
                tid = r["thread_id"]
                pr_num = r["pr_url"].rstrip("/").split("/")[-1]
                risk = r["worst_risk"]
                btn_label = f"PR #{pr_num} [{risk}] - {tid[:8]}"
                if st.button(btn_label, key=f"session_{tid}"):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = r["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
        else:
            st.caption("No previous sessions recorded yet.")

        # Bonus 2: Calibration stats
        st.divider()
        st.subheader("📊 Calibration Analytics")
        stats = asyncio.run(get_calibration_stats())
        if stats and stats["total_sessions"]:
            st.metric("Total Sessions", stats["total_sessions"])
            avg_c = stats["avg_conf"] or 0.0
            st.metric("Average Confidence", f"{avg_c:.1%}")
            col_a, col_b = st.columns(2)
            col_a.write(f"✅ Approved: {stats['approved_count']}")
            col_b.write(f"⚡ Auto: {stats['auto_count']}")
            col_a.write(f"❌ Rejected: {stats['rejected_count']}")
            col_b.write(f"⚠️ Escalated: {stats['escalated_count']}")
    except Exception:
        st.caption("Database not initialized yet.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(payload.get("summary", ""))

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback / Edit instructions (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload.get("summary", ""))

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(q, key=f"q_{i}")
        if st.form_submit_button("Submit answers"):
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(
                Command(resume=resume_value), cfg
            )
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = `{st.session_state.thread_id}`  ·  Replay CLI: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")

# Render Audit timeline if thread is active
if st.session_state.thread_id:
    with st.expander("🔍 Session Audit Trail & Replay Timeline", expanded=False):
        async def fetch_events(tid):
            return await replay_events(tid)
        evs = asyncio.run(fetch_events(st.session_state.thread_id))
        if evs:
            st.dataframe(evs, use_container_width=True)
