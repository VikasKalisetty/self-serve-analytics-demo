import pandas as pd
import plotly.express as px
import streamlit as st

from engine import AnalyticsEngine

st.set_page_config(
    page_title="Self-Serve Analytics AI | Enterprise Context Demo",
    page_icon="📊",
    layout="wide",
)

# --- Initialize Engine & Session State ---
@st.cache_resource
def load_engine():
    return AnalyticsEngine()

engine = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_grounding" not in st.session_state:
    st.session_state.pending_grounding = None


# --- Helper: Render Plotly Chart ---
def render_chart(df: pd.DataFrame, viz: dict):
    if df.empty or not viz:
        st.info("Query returned 0 rows.")
        return

    chart_type = viz.get("chart_type", "table")
    x_col = viz.get("x_col", "")
    y_col = viz.get("y_col", "")
    title = viz.get("chart_title", "")

    if chart_type == "kpi_card" or (len(df) == 1 and len(df.columns) <= 3):
        cols = st.columns(len(df.columns))
        for idx, col_name in enumerate(df.columns):
            val = df.iloc[0, idx]
            if isinstance(val, (int, float)):
                formatted = f"{val:,.2f}"
            else:
                formatted = str(val)
            cols[idx].metric(label=col_name.replace("_", " ").title(), value=formatted)
    elif chart_type == "bar" and x_col in df.columns and y_col in df.columns:
        fig = px.bar(df, x=x_col, y=y_col, title=title, text_auto=".2s", template="plotly_white")
        fig.update_layout(xaxis_tickangle=-25, height=380)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "line" and x_col in df.columns and y_col in df.columns:
        fig = px.line(df, x=x_col, y=y_col, title=title, markers=True, template="plotly_white")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


# --- Helper: Render Glass-Box Reasoning Drawer ---
def render_assistant_details(msg: dict):
    badge = "🟢 Enterprise Context: ON" if msg["context_on"] else "🟠 Enterprise Context: OFF (Ungrounded)"
    st.caption(badge)

    if msg.get("summary"):
        st.markdown(msg["summary"])

    if msg.get("df") is not None and not msg["df"].empty and msg.get("viz"):
        render_chart(msg["df"], msg["viz"])

    with st.expander("🔍 Glass-Box Reasoning Drawer (Grounding, CoT, SQL & Data)", expanded=False):
        t1, t2, t3, t4 = st.tabs(["Grounding & Citations", "Chain of Thought", "Generated SQL", "Data & CSV"])

        with t1:
            plan = msg.get("plan", {})
            st.markdown(f"**Metric:** `{plan.get('metric_name', 'N/A')}`")
            st.markdown(f"**Definition:** {plan.get('business_definition', 'N/A')}")
            st.markdown(f"**Formula:** `{plan.get('formula_summary', 'N/A')}`")
            filters = plan.get("mandatory_filters", [])
            st.markdown("**Mandatory Filters:** " + (", ".join(f"`{f}`" for f in filters) if filters else "None"))
            st.markdown(f"**Join Strategy:** {plan.get('join_strategy', 'N/A')}")

            retrieved = msg.get("retrieved", {})
            kpi_match = retrieved.get("kpi_match", {})
            if msg["context_on"] and not kpi_match.get("is_exact", False):
                st.warning(
                    "No exact KPI name/synonym match was found. Retrieved KPI documents "
                    "were used as reference context, not as the selected KPI."
                )
            if retrieved.get("kpis"):
                st.divider()
                st.markdown("**KPI Context Selection:**")
                for k in retrieved["kpis"]:
                    details = (
                        "exact name/synonym match"
                        if k.get("match_type") == "exact"
                        else f"reference only; embedding similarity: `{k['similarity']}`"
                    )
                    st.markdown(f"- `{k['id']}` **{k['name']}** ({k['version']} · {k['owner']}) — {details}")

        with t2:
            for idx, step in enumerate(msg.get("cot", []), start=1):
                st.markdown(f"{idx}. {step}")

        with t3:
            st.code(msg.get("sql", ""), language="sql")

        with t4:
            df = msg.get("df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv",
                    key=f"dl_{msg['id']}",
                )
            else:
                st.info("No data returned.")


# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Governance Controls")

    context_on = st.toggle(
        "Enterprise Context & Semantic Layer",
        value=True,
        help="Toggle OFF to see how an ungrounded LLM hallucinates KPI definitions, ignores snapshots, or includes void orders.",
    )
    if context_on:
        st.success("**Context ON:** Enforcing governed KPIs, composite company joins, and ERP filters.")
    else:
        st.warning("**Context OFF:** Raw schema only. Expect KPI drift & snapshot inflation!")

    hitl_mode = st.toggle(
        "Human-in-the-Loop Grounding Gate",
        value=True,
        help="Pause before SQL execution to review & confirm the grounded KPI definition.",
    )

    st.divider()
    st.markdown("### 🏗️ 3-Tier Architecture")
    st.markdown(
        "1. **Lakehouse:** 9 Epicor Parquet tables in DuckDB\n"
        "2. **Context & Semantic Layer:** YAML KPIs + 10 composite joins in ChromaDB\n"
        "3. **Agentic Text-to-SQL:** Sarvam 105B Grounding → CoT → Self-Healing SQL → Plotly"
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_grounding = None
        st.rerun()


# --- Main Header & Curated Demo Prompts ---
st.title("📊 Self-Serve Procurement & Inventory AI")
st.markdown(
    "Ask questions in plain English. Toggle **Enterprise Context ON vs. OFF** in the sidebar to compare governed vs. ungrounded SQL."
)

demo_prompts = [
    ("📦 Inventory Valuation", "What is our Total Inventory Valuation by product group?"),
    ("🚚 OTIF Rate", "What is our On-Time In-Full (OTIF) rate by supplier tier?"),
    ("💰 Purchase Price Variance", "Calculate Purchase Price Variance (PPV) by supplier tier."),
    ("📋 Open PO Value", "What is our Open PO Value by company?"),
    ("🛡️ Guardrail Test", "What is our employee headcount by HR salary band?"),
]

cols = st.columns(len(demo_prompts))
selected_demo = None
for idx, (label, prompt_text) in enumerate(demo_prompts):
    if cols[idx].button(label, use_container_width=True):
        selected_demo = prompt_text

st.divider()

# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg.get("declined"):
            st.warning(msg["content"])
        else:
            render_assistant_details(msg)


# --- Handle Pending Human-in-the-Loop Grounding Confirmation ---
if st.session_state.pending_grounding is not None:
    pending = st.session_state.pending_grounding
    plan = pending["bundle"]["plan"]
    mode_label = "Governed Enterprise Context" if pending["bundle"]["context_on"] else "Ungrounded Raw Schema"

    with st.chat_message("assistant"):
        st.info(f"### ✅ Grounding Verification ({mode_label})")
        kpi_match = pending["bundle"]["retrieved"].get("kpi_match", {})
        if pending["bundle"]["context_on"] and not kpi_match.get("is_exact", False):
            st.warning(
                "No exact KPI name/synonym match was found. The generated metric is "
                "derived using related KPI documents as reference; review its definition and formula."
            )
        st.markdown(f"**Metric:** `{plan['metric_name']}`")
        st.markdown(f"**Definition:** {plan['business_definition']}")
        st.markdown(f"**Formula:** `{plan['formula_summary']}`")
        filters = plan.get("mandatory_filters", [])
        st.markdown("**Mandatory Filters:** " + (", ".join(f"`{f}`" for f in filters) if filters else "None"))
        st.markdown(f"**Tables & Join Strategy:** {', '.join(plan.get('tables_to_use', []))} — {plan.get('join_strategy', '')}")

        c1, c2, _ = st.columns([1.2, 1, 3])
        confirm_clicked = c1.button("▶️ Confirm & Run SQL", type="primary", use_container_width=True)
        cancel_clicked = c2.button("✖️ Cancel", use_container_width=True)

        if confirm_clicked:
            with st.spinner("Generating CoT SQL, executing on DuckDB, and building chart..."):
                exec_res = engine.generate_and_execute_sql(pending["question"], pending["bundle"])

            if exec_res["success"]:
                st.session_state.messages.append({
                    "id": len(st.session_state.messages),
                    "role": "assistant",
                    "context_on": pending["bundle"]["context_on"],
                    "summary": exec_res["viz"]["executive_summary"],
                    "df": exec_res["df"],
                    "viz": exec_res["viz"],
                    "sql": exec_res["sql"],
                    "cot": exec_res["chain_of_thought"],
                    "plan": plan,
                    "retrieved": pending["bundle"]["retrieved"],
                })
            else:
                st.session_state.messages.append({
                    "id": len(st.session_state.messages),
                    "role": "assistant",
                    "declined": True,
                    "content": f"SQL execution failed after retries: `{exec_res['error']}`",
                })
            st.session_state.pending_grounding = None
            st.rerun()

        if cancel_clicked:
            st.session_state.pending_grounding = None
            st.rerun()


# --- Chat Input Handling ---
user_input = st.chat_input("Ask about Inventory Valuation, OTIF, PPV, Open PO Value, or Past-Due POs...")
question = selected_demo or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Retrieving Enterprise Context & verifying grounding..."):
        bundle = engine.create_grounding_plan(question, context_on=context_on)

    plan = bundle["plan"]
    if not plan.get("is_answerable", True):
        st.session_state.messages.append({
            "id": len(st.session_state.messages),
            "role": "assistant",
            "declined": True,
            "content": f"🛡️ **Out-of-Scope Guardrail:** {plan.get('decline_reason', 'This question is outside the Procurement & Inventory lakehouse.')}",
        })
        st.rerun()
    elif hitl_mode:
        st.session_state.pending_grounding = {"question": question, "bundle": bundle}
        st.rerun()
    else:
        with st.spinner("Generating CoT SQL, executing on DuckDB, and building chart..."):
            exec_res = engine.generate_and_execute_sql(question, bundle)

        if exec_res["success"]:
            st.session_state.messages.append({
                "id": len(st.session_state.messages),
                "role": "assistant",
                "context_on": context_on,
                "summary": exec_res["viz"]["executive_summary"],
                "df": exec_res["df"],
                "viz": exec_res["viz"],
                "sql": exec_res["sql"],
                "cot": exec_res["chain_of_thought"],
                "plan": plan,
                "retrieved": bundle["retrieved"],
            })
        else:
            st.session_state.messages.append({
                "id": len(st.session_state.messages),
                "role": "assistant",
                "declined": True,
                "content": f"SQL execution failed: `{exec_res['error']}`",
            })
        st.rerun()
