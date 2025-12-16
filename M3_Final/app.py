import streamlit as st
from main import answer_query

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="FPL GraphRAG",
    page_icon="⚽",
    layout="wide",
)

# -----------------------------
# FPL Theme CSS
# -----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .fpl-hero {
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        background: linear-gradient(135deg, #0b3d2e 0%, #0f5f45 45%, #0b3d2e 100%);
        color: #eafff6;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        margin-bottom: 14px;
      }
      .fpl-hero h1 { margin: 0; font-size: 30px; font-weight: 800; }
      .fpl-hero p { margin: 6px 0 0 0; opacity: 0.9; }

      .chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        margin-right: 8px;
        font-size: 12px;
      }

      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071a13 0%, #0b2c21 100%);
      }
      section[data-testid="stSidebar"] * {
        color: #ecfff6 !important;
      }

      [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 8px 12px;
      }

      .stButton>button {
        border-radius: 12px;
        font-weight: 650;
      }

      .metric-row { margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="fpl-hero">
      <h1>⚽ FPL-T69 GraphRAG Assistant ( FantasyTrivia )</h1>
      <p>Ask about players, teams, seasons, gameweeks - powered by our very own Neo4j Knowledge Graph.</p>
      <div style="margin-top:10px;">
        <span class="chip">Baseline (Cypher)</span>
        <span class="chip">Embeddings (Vector)</span>
        <span class="chip">Hybrid</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def _usage_to_dict(usage):
    """Support dict, pydantic CompletionUsage, or None."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    # Pydantic / object with attributes
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

def _format_tokens(usage) -> str:
    u = _usage_to_dict(usage)
    return (
        f"Tokens: prompt={u.get('prompt_tokens')} | "
        f"completion={u.get('completion_tokens')} | "
        f"total={u.get('total_tokens')}"
    )


def _format_retrieval_timing(timing: dict) -> str:
    if not timing:
        return "Retrieval: N/A"
    return (
        f"Retrieval: baseline={timing.get('baseline_s')}s | "
        f"embedding={timing.get('embedding_s')}s"
    )

def _show_metrics(latency_s=None, llm_s=None, usage=None, timing=None):
    cols = st.columns(4)
    cols[0].caption(f"Total latency: {latency_s}s" if latency_s is not None else "Total latency: N/A")
    cols[1].caption(f"LLM: {llm_s}s" if llm_s is not None else "LLM: N/A")
    cols[2].caption(_format_tokens(usage or {}))
    cols[3].caption(_format_retrieval_timing(timing or {}))

# -----------------------------
# Sidebar controls + Quick Questions
# -----------------------------
with st.sidebar:
    st.header("Control Room")

    llm_model = st.selectbox(
        "LLM Model",
        [
            "llama3.1:8b",          # local ollama
            "gemini-2.5-flash",     # gemini
            "gemma-3-4b-it",        # gemma via gemini (if supported in your wrapper)
        ],
        index=0,
    )

    retrieval_type = st.selectbox(
        "Retrieval Method",
        ["Baseline", "Embedding", "Hybrid"],
        index=0,
    )

    embedding_model = st.selectbox(
        "Embedding Model (Embedding/Hybrid)",
        ["mpnet", "minilm"],
        index=0,
    )

    show_debug = st.checkbox("Show debug (KG context + Cypher)", value=False)

    st.divider()
    st.subheader("Quick Questions")

    st.caption("Working questions (should succeed):")
    quick_working = [
        ("player_stats", "Show Mohamed Salah’s stats in the 2022-23 season."),
        ("top_players", "Who are the top 10 players in the 2022-23 season?"),
        ("best_by_position", "Who are the best defenders in the 2022-23 season?"),
        ("compare_players", "Compare Mohamed Salah vs Kevin De Bruyne in the 2022-23 season."),
        ("team_fixtures", "Show Liverpool’s fixtures in the 2022-23 season."),
        ("gameweek_fixtures", "Show fixtures in GW 8 of the 2022-23 season."),
        ("player_clean_sheets", "How many clean sheets did Alisson keep in the 2022-23 season?"),
        ("player_form", "What was Mohamed Salah’s form in the 2022-23 season?"),
        ("player_discipline", "How many yellow and red cards did Mateo Kovacic receive in 2022-23?"),
        ("goalkeeper_stats", "How many saves did Alisson make in the 2022-23 season?"),
        ("recommendation", "Recommend the best midfielders for the 2022-23 season."),
        ("top_players (alt phrasing)", "Top performers in 2022-23"),
    ]

    quick_non_working = [
        ("player_stats", "Show Lionel Messi’s stats in the 2022-23 season."),
        ("top_players", "Who are the top 10 players in the 2018-19 season?"),
        ("best_by_position", "Who are the best wing-backs in the 2022-23 season?"),
        ("compare_players", "Compare Mohamed Salah vs Cristiano Ronaldo in the 2022-23 season."),
        ("team_fixtures", "Show Barcelona’s fixtures in the 2022-23 season."),
        ("gameweek_fixtures", "Show fixtures in GW 50 of the 2022-23 season."),
        ("player_clean_sheets", "How many clean sheets did Liverpool keep in the 2022-23 season?"),
        ("player_form", "What was Mohamed Salah’s form in the 2019-20 season?"),
        ("player_discipline", "How many fouls did Ben Davies commit in 2022-23?"),
        ("goalkeeper_stats", "How many saves did Mohamed Salah make in the 2022-23 season?"),
        ("recommendation", "Recommend the best referees for the 2022-23 season."),
    ]

    if "queued_question" not in st.session_state:
        st.session_state.queued_question = None

    def queue_question(q: str):
        st.session_state.queued_question = q

    for _, q in quick_working:
        st.button(f"{q}", on_click=queue_question, args=(q,), use_container_width=True)

    st.divider()

    st.caption("Non-working tests (should fail gracefully):")

    for _, q in quick_non_working:
        st.button(f"{q}", on_click=queue_question, args=(q,), use_container_width=True)

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.queued_question = None
        st.rerun()

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

        # metrics for assistant messages
        if m["role"] == "assistant" and m.get("metrics"):
            met = m["metrics"]
            usage = met.get("usage") or {}
            timing = met.get("timing") or {}
            _show_metrics(
                latency_s=met.get("latency_s"),
                llm_s=met.get("llm_s"),
                usage=usage,
                timing=timing,
            )

        if show_debug and m.get("debug") is not None and m["role"] == "assistant":
            with st.expander("Debug: KG context + Cypher", expanded=False):
                st.json(m["debug"])
                if "cypher" in m["debug"]:
                    st.code(m["debug"]["cypher"]["query"], language="cypher")
                    st.json(m["debug"]["cypher"]["params"])

# -----------------------------
# Chat input
# -----------------------------
prefill = ""
if st.session_state.queued_question:
    prefill = st.session_state.queued_question
    st.session_state.queued_question = None

user_query = st.chat_input("Type your FPL question…")

if not user_query and prefill:
    user_query = prefill

if user_query:
    # user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking like a scout :D .."):
            out = answer_query(
                user_query,
                retrieval_type=retrieval_type,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )

        st.write(out["answer"])

        # show metrics live
        usage = out.get("usage") or {}
        timing = (out.get("debug") or {}).get("timing") or {}
        _show_metrics(
            latency_s=out.get("latency_s"),
            llm_s=out.get("llm_s"),
            usage=usage,
            timing=timing,
        )

        if show_debug:
            with st.expander("Debug: KG context + Cypher", expanded=False):
                st.json(out["debug"])
                st.code(out["debug"]["cypher"]["query"], language="cypher")
                st.json(out["debug"]["cypher"]["params"])

    # assistant message saved with debug + metrics
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": out["answer"],
            "debug": out.get("debug"),
            "metrics": {
                "latency_s": out.get("latency_s"),
                "llm_s": out.get("llm_s"),
                "usage": out.get("usage"),
                "timing": (out.get("debug") or {}).get("timing"),
            },
        }
    )
