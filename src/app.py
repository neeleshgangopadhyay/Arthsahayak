"""
ArthSahayak - Streamlit frontend (Step 9)
 
A thin presentation layer over the existing LangGraph agent in agents.py.
No agent logic lives here - this just calls build_agent() and renders the
result, including the routing decision and groundedness/citation status,
since that's the most differentiated part of the project and otherwise
invisible to anyone just reading a plain chat transcript.
 
Run from the project root:
    streamlit run src\\app.py
(same import style as eval_agent.py - Streamlit adds this script's own
directory to sys.path, so `from agents import build_agent` resolves the
same way it does there.)
"""
 
import inspect
import time
 
# Workaround for a known Streamlit/huggingface_hub interaction: Streamlit's
# startup code calls inspect.stack() (to detect whether it's running in a
# REPL), which walks every frame on the stack and, for each one, calls
# inspect.getmodule() to find the module that frame's code belongs to.
# That crashes with "'function' object has no attribute 'endswith'" when a
# lazily-loaded module (huggingface_hub's _LazyModule, pulled in by
# sentence-transformers/embeddings libraries) has left a non-standard
# entry in sys.modules whose __file__ isn't a normal string.
#
# The first attempt at this patch wrapped inspect.stack() itself and fell
# back to an empty list on failure - that avoided the crash but broke a
# different assumption downstream (Streamlit indexes the *last* frame,
# stack()[-1], and an empty list has no last element -> IndexError).
# Patching the actual failing call one level down, inspect.getmodule(),
# is the correct fix: on failure it returns None, which every caller
# (including Python's own inspect.findsource) already treats as a normal
# "couldn't determine the module" case and falls back gracefully from -
# it's the same path taken for code with no module at all (e.g. exec'd
# strings), so this doesn't change behavior for any frame that resolves
# normally; it only stops the one broken frame from taking the whole
# server down.
_original_getmodule = inspect.getmodule
 
 
def _safe_getmodule(*args, **kwargs):
    try:
        return _original_getmodule(*args, **kwargs)
    except Exception:
        return None
 
 
inspect.getmodule = _safe_getmodule
 
import streamlit as st
 
from agents import build_agent
 
st.set_page_config(
    page_title="ArthSahayak",
    page_icon="\U0001F4B0",
    layout="centered",
)
 
# ---------------------------------------------------------------------------
# Theme - a green palette that adapts to the viewer's light/dark preference
# via @media (prefers-color-scheme), rather than Streamlit's single static
# config.toml theme (which can't switch shades on its own). Only touches
# chrome (backgrounds, borders, buttons, headers) - left the semantic
# badge colors (red/orange/green/blue) in ACTION_BADGES alone, since those
# carry meaning (error, unverified, grounded) that a uniform green theme
# would wash out.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --app-bg: #f1f8f4;
        --sidebar-bg: #e3f2e6;
        --accent: #2e7d32;
        --accent-dark: #1b5e20;
        --border: #a5d6a7;
        --card-bg: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #0d1f12;
            --sidebar-bg: #112615;
            --accent: #66bb6a;
            --accent-dark: #a5d6a7;
            --border: #2e5631;
            --card-bg: #16281a;
        }
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--app-bg);
    }
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }
    h1, h2, h3 {
        color: var(--accent-dark);
    }
    .stButton button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        transition: background-color 0.15s ease;
    }
    .stButton button:hover {
        background-color: var(--accent-dark);
        color: white;
    }
    [data-testid="stChatMessage"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 6px 10px;
        margin-bottom: 6px;
    }
    [data-testid="stChatInput"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    hr {
        border-color: var(--border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ---------------------------------------------------------------------------
# Agent setup - built once per server process, not once per message. Without
# caching, every single chat turn would rebuild the whole LangGraph graph
# (and whatever retrievers/clients it wires up) from scratch.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    return build_agent()
 
 
agent = get_agent()
 
ACTION_BADGES = {
    "retrieve": ("\U0001F4C4 Answered from documents", "blue"),
    "news": ("\U0001F4F0 Live news", "orange"),
    "verify": ("\U0001F50D Fact-checked against live data", "violet"),
    "clarify": ("❓ Needs clarification", "gray"),
    "reject": ("\U0001F6AB Blocked by input guardrail", "red"),
    "error": ("⚠️ Something went wrong", "red"),
}
 
EXAMPLE_QUESTIONS = [
    "What does the Economic Survey say about inflation trends?",
    "What's the latest economic news in India today?",
    "The repo rate is currently 7%, right? What does that mean for borrowers?",
    "Tell me about the economy.",
]
 
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
# Separate from "messages" (which also carries display metadata/badges) -
# this is exactly what gets passed to agent.invoke(..., history=...) so the
# agent's contextualize_node can rewrite follow-ups against it. Capped to
# the last few exchanges so prompts built from it don't grow unbounded over
# a long session.
if "history" not in st.session_state:
    st.session_state["history"] = []
MAX_HISTORY_TURNS = 6
 
 
def _session_stats():
    """
    A live, session-only mini-dashboard computed from this conversation's
    own messages - not a replacement for the Step 8 eval suite (which
    measures against a known-correct labeled test set), just a quick
    "what's happened in this chat so far" view, since that's the kind of
    thing an interactive demo benefits from showing.
    """
    assistant_msgs = [
        m for m in st.session_state["messages"]
        if m["role"] == "assistant" and m.get("meta")
    ]
    if not assistant_msgs:
        return None
    total = len(assistant_msgs)
    by_action = {}
    latencies = []
    grounded_count = 0
    groundable = 0
    for m in assistant_msgs:
        meta = m["meta"]
        action = meta.get("action", "unknown")
        by_action[action] = by_action.get(action, 0) + 1
        if meta.get("latency") is not None:
            latencies.append(meta["latency"])
        if meta.get("grounded") is not None:
            groundable += 1
            if meta["grounded"]:
                grounded_count += 1
    return {
        "total": total,
        "by_action": by_action,
        "avg_latency": sum(latencies) / len(latencies) if latencies else None,
        "grounded_count": grounded_count,
        "groundable": groundable,
    }
 
 
# ---------------------------------------------------------------------------
# Sidebar - pitch, example questions, live session stats.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ArthSahayak")
    st.caption(
        "An agentic RAG system over Indian Economics documents. "
        "Routes each question to document retrieval, live news, or "
        "real-time fact-checking, and grounds every answer in its sources."
    )
    st.divider()
    st.markdown("**Try asking:**")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"example-{q}"):
            st.session_state["pending_question"] = q
 
    stats = _session_stats()
    if stats:
        st.divider()
        with st.expander(f"\U0001F4CA Session stats ({stats['total']} answered)", expanded=False):
            action_labels = {
                "retrieve": "Documents", "news": "News", "verify": "Verified",
                "clarify": "Clarified", "reject": "Blocked", "error": "Errors",
            }
            for action, count in sorted(stats["by_action"].items(), key=lambda kv: -kv[1]):
                st.markdown(f"- {action_labels.get(action, action)}: **{count}**")
            if stats["avg_latency"] is not None:
                st.markdown(f"- Avg response time: **{stats['avg_latency']:.1f}s**")
            if stats["groundable"]:
                pct = 100 * stats["grounded_count"] / stats["groundable"]
                st.markdown(f"- Grounded: **{stats['grounded_count']}/{stats['groundable']} ({pct:.0f}%)**")
 
    st.divider()
    if st.button("\U0001F5D1️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["history"] = []
        st.rerun()
 
st.title("ArthSahayak")
st.caption("Ask a question about the Indian economy.")
 
 
def render_sources(retrieved, key_prefix):
    """An expandable panel listing exactly what the answer was grounded in -
    the retrieved document/page or news/web source for each chunk - so the
    citations in the answer text aren't the only visible evidence of where
    it came from."""
    if not retrieved:
        return
    try:
        expander = st.expander(
            f"\U0001F4DA View sources ({len(retrieved)})", key=f"{key_prefix}-sources"
        )
    except TypeError:
        # Older Streamlit versions don't accept a "key" on st.expander -
        # fall back without one rather than crashing the whole message.
        expander = st.expander(f"\U0001F4DA View sources ({len(retrieved)})")
    with expander:
        for chunk in retrieved:
            doc = chunk.get("document", "Unknown source")
            page = chunk.get("page", "-")
            label = doc if page == "-" else f"{doc}, p.{page}"
            snippet = (chunk.get("text") or "").strip()
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."
            st.markdown(f"**{label}**")
            st.caption(snippet)
 
 
def render_feedback(key_prefix):
    """Thumbs up/down on each answer. Session-only (no backend to persist
    to), but it's a real interactive affordance rather than a static
    transcript, and the pattern is the same one a production chat product
    would use to collect signal on answer quality."""
    try:
        st.feedback("thumbs", key=f"{key_prefix}-feedback")
    except Exception:
        # st.feedback needs a recent Streamlit version - degrade silently
        # on an older install rather than crashing the whole message.
        pass
 
 
def render_caption(meta):
    badge_text, badge_color = ACTION_BADGES.get(meta.get("action"), ("", "gray"))
    caption_bits = []
    if badge_text:
        caption_bits.append(f":{badge_color}[{badge_text}]")
    if meta.get("grounded") is True:
        caption_bits.append(":green[✅ Grounded in sources]")
    elif meta.get("grounded") is False:
        caption_bits.append(":orange[⚠️ Unverified after retries]")
    if meta.get("attempts") and meta["attempts"] > 1:
        caption_bits.append(f"({meta['attempts']} attempts)")
    if meta.get("latency") is not None:
        caption_bits.append(f"{meta['latency']:.1f}s")
    if caption_bits:
        st.caption(" &nbsp;·&nbsp; ".join(caption_bits))
 
 
# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------
for idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            render_caption(meta)
            render_sources(meta.get("retrieved"), key_prefix=f"hist-{idx}")
            if msg["role"] == "assistant" and meta.get("action") not in ("reject", "error"):
                render_feedback(key_prefix=f"hist-{idx}")
 
 
def run_turn(question: str) -> None:
    """Send one question through the agent and render the response."""
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking through your question..."):
            try:
                start = time.time()
                result = agent.invoke({
                    "question": question,
                    "attempts": 0,
                    "history": st.session_state["history"],
                })
                latency = time.time() - start
            except Exception as e:
                # Same class of failure the eval suite had to harden against -
                # a provider-side error (rate limit, daily token cap) shouldn't
                # crash the whole app, just this one turn.
                error_text = (
                    "Sorry, that request failed - the model provider returned an "
                    f"error ({e}). Try again in a moment."
                )
                st.markdown(error_text)
                st.caption(":red[⚠️ Something went wrong]")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_text,
                    "meta": {"action": "error"},
                })
                return
 
        action = result["action"] if "action" in result else "reject"
        answer = result.get("final_answer") or "I couldn't produce an answer for that question."
 
        st.markdown(answer)
 
        meta = {
            "action": action,
            "grounded": result.get("grounded"),
            "attempts": result.get("attempts"),
            "latency": latency,
            "retrieved": result.get("retrieved"),
        }
        render_caption(meta)
        new_idx = len(st.session_state["messages"])
        render_sources(meta.get("retrieved"), key_prefix=f"new-{new_idx}")
        if action not in ("reject", "error"):
            render_feedback(key_prefix=f"new-{new_idx}")
        st.toast("Answer ready", icon="✅")
 
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "meta": meta,
        })
 
        # Record this exchange using the raw question the user actually
        # typed (not any rewritten standalone version) and the final
        # rendered answer, so the next turn's contextualize_node sees the
        # conversation the same way a person reading the chat would.
        st.session_state["history"].append({"question": question, "answer": answer})
        st.session_state["history"] = st.session_state["history"][-MAX_HISTORY_TURNS:]
 
 
# A sidebar example button sets this; handle it before the chat_input so a
# click behaves exactly like typing the question and hitting enter.
pending = st.session_state.pop("pending_question", None)
if pending:
    run_turn(pending)
 
prompt = st.chat_input("Ask about the Indian economy...")
if prompt:
    run_turn(prompt)
 