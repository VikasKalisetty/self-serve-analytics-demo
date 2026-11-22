from html import escape
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import AnalyticsEngine


st.set_page_config(
    page_title="Procurement Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
:root {
    --ink: #172033;
    --muted: #667085;
    --line: #e7ecf3;
    --blue: #315efb;
    --blue-soft: #edf2ff;
    --green: #12a36d;
    --panel: #ffffff;
}

.block-container {
    max-width: 1800px;
    padding: 0 32px 12px !important;
    margin: 0 auto;
}

[data-testid="stApp"] {
    height: 100vh;
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"],
[data-testid="stAppHeader"],
[data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stAppViewBlockContainer"] {
    padding: 0 !important;
    margin: 0 !important;
}
section.main > div {
    padding: 0 !important;
}
section.main,
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
    top: 0 !important;
}
section.main {
    height: 100vh;
    min-height: 100vh;
    overflow-x: hidden;
    overflow-y: auto;
}
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
div[data-testid="column"] {
    min-width: 0;
}
div[data-testid="stChatInput"] {
    margin: 0;
}
[data-testid="stElementToolbar"],
.stElementToolbar,
[data-testid="stElementToolbarButton"] {
    display: none !important;
}

.decision-header {
    height: 112px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding: 12px 0 4px;
    background: white;
}

.decision-title {
    color: #68227a;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 43px;
    line-height: 0.95;
    font-weight: 800;
    letter-spacing: -1.8px;
}

.decision-underline {
    width: 188px;
    height: 7px;
    margin-top: -2px;
    background:
        repeating-linear-gradient(
            90deg,
            #68227a 0,
            #68227a 10px,
            transparent 10px,
            transparent 13px
        );
    clip-path: polygon(0 35%, 100% 35%, 100% 65%, 0 65%);
}

.decision-subtitle {
    color: #8b21ef;
    font-size: 20px;
    line-height: 1.15;
    margin-top: 5px;
    font-weight: 500;
}

[data-testid="stHorizontalBlock"] {
    align-items: stretch;
    height: calc(100vh - 112px);
}

[data-testid="stHorizontalBlock"] [data-testid="column"] {
    height: calc(100vh - 112px);
}

[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1),
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3) {
    border: 1px solid #dfe5ef;
    border-radius: 12px;
    background: white;
    padding: 0 10px 10px;
}

[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) {
    border: 2px dashed #dfe5ef;
    border-radius: 12px;
    background: white;
    padding: 0;
}

[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1)
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3)
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #dfe5ef;
    border-radius: 12px;
    background: white;
}

[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2)
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px dashed #dfe5ef;
    border-radius: 12px;
    background: #fff;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}

.workspace-header {
    display: none;
}

.brand {
    display: flex;
    align-items: center;
    gap: .65rem;
}

.brand-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    color: white;
    background: linear-gradient(135deg, #315efb, #7b61ff);
    font-weight: 800;
}

.brand-name {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 800;
}

.brand-subtitle {
    color: var(--muted);
    font-size: .72rem;
}

.workspace-status {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    border: 1px solid #d7efdf;
    background: #effaf4;
    color: #087d51;
    border-radius: 999px;
    padding: .35rem .65rem;
    font-size: .75rem;
    font-weight: 750;
}

.pane-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--ink);
    font-size: .82rem;
    font-weight: 800;
    letter-spacing: .04em;
    text-transform: uppercase;
    padding: .05rem .1rem .55rem;
}

.dataset-card {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fbfcfe;
    padding: .72rem;
    margin-bottom: .6rem;
}

.dataset-card.active {
    border-color: #aebcff;
    background: var(--blue-soft);
}

.dataset-title {
    color: var(--ink);
    font-size: .82rem;
    font-weight: 750;
    margin-bottom: .25rem;
}

.dataset-meta {
    color: var(--muted);
    font-size: .72rem;
}

.canvas-shell {
    border: 1px dashed #cfd8e6;
    border-radius: 16px;
    background:
        linear-gradient(rgba(49,94,251,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(49,94,251,.025) 1px, transparent 1px),
        #fbfcfe;
    background-size: 24px 24px;
    min-height: calc(100vh - 116px);
    overflow: hidden;
}

.canvas-toolbar {
    border-bottom: 1px solid var(--line);
    background: white;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 .9rem;
}

.canvas-title {
    color: var(--ink);
    font-weight: 800;
    font-size: .88rem;
}

.canvas-hint {
    color: var(--muted);
    font-size: .72rem;
}

.chat-shell {
    display: flex;
    flex-direction: column;
}

.chat-scroll {
    flex: 1;
    overflow-y: auto;
    max-height: calc(100vh - 205px);
    padding-right: .15rem;
}

.chat-user {
    margin-left: 1.5rem;
    border-radius: 14px 14px 4px 14px;
    background: #eef3ff;
    color: #27364f;
    padding: .7rem .78rem;
    font-size: .82rem;
    margin-bottom: .65rem;
}

.chat-assistant {
    margin-right: 1.2rem;
    border: 1px solid var(--line);
    border-radius: 14px 14px 14px 4px;
    background: white;
    color: var(--ink);
    padding: .7rem .78rem;
    font-size: .82rem;
    margin-bottom: .65rem;
}

.chat-status {
    color: var(--muted);
    font-size: .7rem;
    margin-bottom: .38rem;
}

.chat-message-summary {
    line-height: 1.48;
}

.grounding-mini {
    border: 1px solid #dbe5ff;
    background: #f5f7ff;
    border-radius: 10px;
    padding: .55rem;
    margin-top: .55rem;
    color: #344054;
    font-size: .72rem;
}

[data-testid="stChatInput"] {
    border: 1px solid #d9e1ee;
    border-radius: 14px;
    background: white;
    margin: 4px;
    width: calc(100% - 8px);
}

[data-testid="stChatInput"] textarea {
    min-height: 48px;
}

.chat-message-padding-top {
    height: 8px;
}

.chat-message-padding-bottom {
    height: 12px;
}

div[data-testid="stColumn"] {
    padding-left: 3px;
    padding-right: 3px;
}

.conversation-window {
    border: 1px solid #dfe5ef;
    border-radius: 14px;
    background: #ffffff;
    height: calc(100vh - 112px);
    min-height: 580px;
    overflow: hidden;
}

.conversation-scroll {
    height: calc(100vh - 245px);
    min-height: 430px;
    overflow-y: auto;
    padding: 8px 6px 4px;
}

.conversation-controls {
    border-top: 1px solid #eef1f6;
    background: #ffffff;
    padding: 8px 6px 6px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    scroll-margin-top: 0;
}

[data-testid="stVerticalBlock"] [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stMarkdownContainer"]) {
    /* Native Streamlit containers retain their own rounded border. */
}

.stButton > button {
    border-radius: 10px;
    min-height: 38px;
}

[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fbfcfe;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 10px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
    margin-bottom: 3px;
}

.stPopover > button,
[data-testid="stPopover"] > button {
    width: 34px;
    height: 34px;
    min-height: 34px;
    border-radius: 50%;
    padding: 0;
    color: #667085;
    border: 1px solid #dfe5ef;
    background: #ffffff;
    box-shadow: 0 3px 10px rgba(23,32,51,.08);
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
}

[data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar {
    width: 8px;
}

[data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 999px;
}

@media (max-width: 1000px) {
    .chat-scroll { max-height: 480px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return AnalyticsEngine()


engine = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_grounding" not in st.session_state:
    st.session_state.pending_grounding = None
if "generated_datasets" not in st.session_state:
    st.session_state.generated_datasets = []
if "active_dataset_id" not in st.session_state:
    st.session_state.active_dataset_id = None
if "whiteboard_items" not in st.session_state:
    st.session_state.whiteboard_items = []
if "center_view_mode" not in st.session_state:
    st.session_state.center_view_mode = "empty"
if "semantic_context_on" not in st.session_state:
    st.session_state.semantic_context_on = True
if "hitl_grounding_on" not in st.session_state:
    st.session_state.hitl_grounding_on = True


def infer_viz(df: pd.DataFrame, requested_type: str = "") -> dict:
    if df.empty:
        return {"chart_type": "table", "x_col": "", "y_col": "", "chart_title": "Result"}

    numeric_columns = [
        column for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column for column in df.columns
        if column not in numeric_columns
    ]

    if requested_type == "kpi_card":
        chart_type = "kpi_card"
    elif requested_type == "bar" and categorical_columns and numeric_columns:
        chart_type = "bar"
    elif requested_type == "line" and numeric_columns:
        chart_type = "line"
    elif len(df) == 1 and len(df.columns) <= 3:
        chart_type = "kpi_card"
    elif categorical_columns and numeric_columns:
        chart_type = "bar"
    elif numeric_columns and len(df) > 1:
        chart_type = "table"
    else:
        chart_type = "table"

    x_col = categorical_columns[0] if categorical_columns else (
        df.columns[0] if len(df.columns) else ""
    )
    y_col = numeric_columns[0] if numeric_columns else ""
    return {
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "chart_title": "Query result",
    }


def render_native_chart(df: pd.DataFrame, viz: dict, height: int = 300):
    if df.empty:
        st.info("No rows returned.")
        return

    chart_type = viz.get("chart_type", "table")
    x_col = viz.get("x_col", "")
    y_col = viz.get("y_col", "")
    title = viz.get("chart_title", "Query result")

    if chart_type == "kpi_card":
        cols = st.columns(max(1, min(len(df.columns), 4)))
        for idx, column in enumerate(df.columns[:4]):
            value = df.iloc[0, idx]
            if isinstance(value, (int, float)):
                value = f"{value:,.2f}"
            cols[idx].metric(column.replace("_", " ").title(), value)
    elif chart_type == "bar" and x_col in df.columns and y_col in df.columns:
        fig = px.bar(
            df, x=x_col, y=y_col, title=title,
            text_auto=".2s", template="plotly_white",
        )
        fig.update_traces(marker_color="#315efb", marker_line_width=0)
        fig.update_layout(
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=10, t=45, b=35),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "line" and x_col in df.columns and y_col in df.columns:
        fig = px.line(
            df, x=x_col, y=y_col, title=title,
            markers=True, template="plotly_white",
        )
        fig.update_traces(line_color="#315efb", line_width=3)
        fig.update_layout(
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=10, t=45, b=35),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True, height=height)


def create_canvas_card_html(item: dict) -> str:
    title = escape(str(item["title"]))
    chart_type = item["viz"].get("chart_type", "table")
    df = item["df"]

    if df.empty:
        body = '<div class="canvas-empty">No rows returned</div>'
    elif chart_type == "kpi_card":
        cards = []
        for idx, column in enumerate(df.columns[:4]):
            value = df.iloc[0, idx]
            if isinstance(value, (int, float)):
                value = f"{value:,.2f}"
            cards.append(
                f'<div class="metric-card"><span>{escape(str(column).replace("_", " ").title())}</span>'
                f'<strong>{escape(str(value))}</strong></div>'
            )
        body = '<div class="metric-grid">' + "".join(cards) + "</div>"
    elif chart_type in {"bar", "line"}:
        x_col = item["viz"].get("x_col", "")
        y_col = item["viz"].get("y_col", "")
        if x_col in df.columns and y_col in df.columns:
            fig = px.bar(df, x=x_col, y=y_col) if chart_type == "bar" else px.line(
                df, x=x_col, y=y_col, markers=True
            )
            fig.update_layout(
                height=260,
                margin=dict(l=20, r=10, t=20, b=35),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
            )
            body = fig.to_html(
                include_plotlyjs="cdn",
                full_html=False,
                config={"displayModeBar": False, "responsive": True},
            )
        else:
            body = df.head(20).to_html(index=False, classes="canvas-table")
    else:
        body = df.head(20).to_html(index=False, classes="canvas-table")

    return f"""
    <article class="canvas-card" draggable="true" data-card-id="{escape(str(item['id']))}">
        <div class="drag-header">
            <div>
                <span class="drag-grip">⠿</span>
                <strong>{title}</strong>
            </div>
            <span class="card-kind">{escape(chart_type.replace("_", " ").title())}</span>
        </div>
        <div class="canvas-card-body">{body}</div>
    </article>
    """


def _render_canvas_iframe_disabled():
    items = st.session_state.canvas_items
    cards_html = "".join(create_canvas_card_html(item) for item in items) if items else """
    <div class="canvas-empty-state">
        <div class="empty-icon">◆</div>
        <strong>Build your analysis</strong>
        <span>Select a generated dataset on the left, choose a visual, then add it here.</span>
    </div>
    """

    payload = {
        "items": [
            {
                "id": item["id"],
                "title": item["title"],
                "df": item["df"].head(20).where(pd.notna(item["df"].head(20)), None).to_dict("records"),
                "columns": list(item["df"].columns),
                "viz": item["viz"],
            }
            for item in items
        ]
    }

    html = f"""
    <!doctype html>
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background:
                linear-gradient(rgba(49,94,251,.032) 1px, transparent 1px),
                linear-gradient(90deg, rgba(49,94,251,.032) 1px, transparent 1px),
                #fbfcfe;
            background-size: 24px 24px;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        #canvas {{
            position: relative;
            min-height: 660px;
            height: calc(100vh - 94px);
            display: grid;
            grid-template-columns: repeat(2, minmax(280px, 1fr));
            align-content: start;
            gap: 14px;
            padding: 14px;
        }}
        .canvas-card {{
            min-width: 0;
            min-height: 180px;
            border: 1px solid #e3e8f0;
            border-radius: 14px;
            background: white;
            box-shadow: 0 8px 24px rgba(23,32,51,.055);
            overflow: hidden;
            transition: opacity .15s ease, transform .15s ease, border-color .15s ease;
        }}
        .canvas-card.dragging {{ opacity: .45; transform: scale(.985); }}
        .canvas-card.drag-over {{ border-color: #315efb; }}
        .drag-header {{
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #edf0f5;
            padding: 0 10px 0 8px;
            color: #172033;
            cursor: grab;
        }}
        .drag-header:active {{ cursor: grabbing; }}
        .drag-grip {{
            color: #98a2b3;
            margin-right: 5px;
            font-size: 16px;
        }}
        .card-kind {{
            color: #667085;
            background: #f2f4f7;
            border-radius: 999px;
            padding: 4px 7px;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .04em;
        }}
        .canvas-card-body {{ padding: 8px; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 8px;
        }}
        .metric-card {{
            min-height: 110px;
            border: 1px solid #edf0f5;
            border-radius: 10px;
            background: #fbfcfe;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 12px;
        }}
        .metric-card span {{ color: #667085; font-size: 11px; }}
        .metric-card strong {{ color: #172033; font-size: 24px; }}
        .canvas-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        .canvas-table th {{
            position: sticky;
            top: 0;
            background: #f5f7fb;
            color: #344054;
            text-align: left;
        }}
        .canvas-table th, .canvas-table td {{
            border: 1px solid #edf0f5;
            padding: 6px;
            max-width: 140px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .canvas-empty-state {{
            grid-column: 1 / -1;
            min-height: 380px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding-top: 22px;
            color: #667085;
            text-align: center;
        }}
        .canvas-empty-state strong {{
            color: #344054;
            margin: 8px 0 4px;
        }}
        .canvas-empty-state span {{
            max-width: 320px;
            font-size: 12px;
        }}
        .empty-icon {{
            width: 46px;
            height: 46px;
            display: grid;
            place-items: center;
            border: 1px solid #dbe5ff;
            border-radius: 14px;
            background: #edf2ff;
            color: #315efb;
        }}
        .canvas-empty {{ color: #667085; padding: 20px; font-size: 12px; }}
        .canvas-menu {{
            position: absolute;
            z-index: 1000;
            display: none;
            width: 176px;
            border: 1px solid #dde4ef;
            border-radius: 12px;
            background: white;
            box-shadow: 0 14px 36px rgba(23,32,51,.16);
            padding: 6px;
        }}
        .canvas-menu.open {{ display: block; }}
        .canvas-menu button {{
            width: 100%;
            height: 36px;
            border: 0;
            border-radius: 8px;
            background: transparent;
            color: #344054;
            text-align: left;
            padding: 0 9px;
            font: 600 12px Inter, sans-serif;
            cursor: pointer;
        }}
        .canvas-menu button:hover {{
            background: #edf2ff;
            color: #315efb;
        }}
        .canvas-object {{
            position: absolute;
            z-index: 20;
            min-width: 120px;
            min-height: 44px;
            cursor: move;
        }}
        .canvas-text {{
            border: 1px solid transparent;
            border-radius: 8px;
            background: rgba(255,255,255,.92);
            color: #172033;
            padding: 10px 12px;
            font-size: 16px;
            line-height: 1.35;
            box-shadow: 0 4px 14px rgba(23,32,51,.08);
            outline: none;
        }}
        .canvas-text:focus {{ border-color: #315efb; }}
        .canvas-shape {{
            width: 180px;
            height: 110px;
            border: 2px solid #315efb;
            background: rgba(49,94,251,.10);
        }}
        .canvas-shape.circle {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
        }}
        .canvas-object-actions {{
            position: absolute;
            right: -8px;
            top: -8px;
            z-index: 30;
            display: none;
            width: 24px;
            height: 24px;
            border: 1px solid #e3e8f0;
            border-radius: 50%;
            background: white;
            color: #667085;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(23,32,51,.12);
        }}
        .canvas-object:hover .canvas-object-actions {{ display: flex; }}
        @media (max-width: 900px) {{
            #canvas {{ grid-template-columns: 1fr; }}
        }}
    </style>
    </head>
    <body>
        <div id="canvas">{cards_html}</div>
        <div id="canvasMenu" class="canvas-menu">
            <button type="button" data-object="text">＋ Add text box</button>
            <button type="button" data-object="rectangle">▢ Add rectangle</button>
            <button type="button" data-object="circle">○ Add circle</button>
        </div>
        <script>
        const canvas = document.getElementById("canvas");
        const canvasMenu = document.getElementById("canvasMenu");
        const storageKey = "procurement-whiteboard-objects";
        let draggedCard = null;
        let menuPosition = {{ x: 30, y: 30 }};

        function saveObjects() {{
            const objects = [...canvas.querySelectorAll(".canvas-object")].map(element => {{
                const type = element.dataset.objectType;
                return {{
                    id: element.dataset.objectId,
                    type,
                    text: type === "text" ? element.querySelector(".canvas-text").innerText : "",
                    shape: element.dataset.shape || "",
                    left: element.style.left,
                    top: element.style.top
                }};
            }});
            localStorage.setItem(storageKey, JSON.stringify(objects));
        }}

        function addObject(type, left, top, text, shape, persist = true) {{
            if (type === "text" && text === undefined) {{
                text = prompt("Enter text");
                if (text === null) return;
            }}

            const element = document.createElement("div");
            element.className = "canvas-object";
            element.dataset.objectType = type;
            element.dataset.objectId = "object-" + Date.now();
            element.style.left = left + "px";
            element.style.top = top + "px";

            if (type === "text") {{
                const textBox = document.createElement("div");
                textBox.className = "canvas-text";
                textBox.contentEditable = "true";
                textBox.spellcheck = true;
                textBox.innerText = text || "New text";
                textBox.addEventListener("input", saveObjects);
                element.appendChild(textBox);
            }} else {{
                element.dataset.shape = shape || "rectangle";
                const shapeBox = document.createElement("div");
                shapeBox.className = "canvas-shape " + (shape === "circle" ? "circle" : "");
                element.appendChild(shapeBox);
            }}

            const remove = document.createElement("button");
            remove.className = "canvas-object-actions";
            remove.innerText = "×";
            remove.title = "Remove";
            remove.addEventListener("click", event => {{
                event.stopPropagation();
                element.remove();
                saveObjects();
            }});
            element.appendChild(remove);
            makeObjectDraggable(element);
            canvas.appendChild(element);
            if (persist) saveObjects();
        }}

        function makeObjectDraggable(element) {{
            let dragging = false;
            let offsetX = 0;
            let offsetY = 0;
            element.addEventListener("pointerdown", event => {{
                if (event.target.closest(".canvas-object-actions")) return;
                dragging = true;
                const rect = element.getBoundingClientRect();
                offsetX = event.clientX - rect.left;
                offsetY = event.clientY - rect.top;
                element.setPointerCapture(event.pointerId);
                element.style.zIndex = "100";
            }});
            element.addEventListener("pointermove", event => {{
                if (!dragging) return;
                const bounds = canvas.getBoundingClientRect();
                const left = Math.max(0, Math.min(bounds.width - element.offsetWidth, event.clientX - bounds.left - offsetX));
                const top = Math.max(0, Math.min(bounds.height - element.offsetHeight, event.clientY - bounds.top - offsetY));
                element.style.left = left + "px";
                element.style.top = top + "px";
            }});
            element.addEventListener("pointerup", event => {{
                if (!dragging) return;
                dragging = false;
                element.style.zIndex = "20";
                try {{ element.releasePointerCapture(event.pointerId); }} catch (_) {{}}
                saveObjects();
            }});
        }}

        function openMenu(event) {{
            menuPosition = {{
                x: Math.min(canvas.clientWidth - 186, event.offsetX),
                y: Math.min(canvas.clientHeight - 150, event.offsetY)
            }};
            canvasMenu.style.left = menuPosition.x + "px";
            canvasMenu.style.top = menuPosition.y + "px";
            canvasMenu.classList.add("open");
        }}

        canvas.addEventListener("dblclick", event => {{
            if (event.target.closest(".canvas-card") || event.target.closest(".canvas-object")) return;
            openMenu(event);
        }});

        canvasMenu.querySelectorAll("button").forEach(button => {{
            button.addEventListener("click", () => {{
                const type = button.dataset.object;
                if (type === "text") addObject("text", menuPosition.x, menuPosition.y);
                if (type === "rectangle") addObject("shape", menuPosition.x, menuPosition.y, "", "rectangle");
                if (type === "circle") addObject("shape", menuPosition.x, menuPosition.y, "", "circle");
                canvasMenu.classList.remove("open");
            }});
        }});

        document.addEventListener("pointerdown", event => {{
            if (!canvasMenu.contains(event.target)) canvasMenu.classList.remove("open");
        }});

        try {{
            const storedObjects = JSON.parse(localStorage.getItem(storageKey) || "[]");
            storedObjects.forEach(object => {{
                addObject(
                    object.type,
                    parseFloat(object.left) || 30,
                    parseFloat(object.top) || 30,
                    object.text,
                    object.shape,
                    false
                );
            }});
        }} catch (_) {{}}

        canvas.querySelectorAll(".canvas-card").forEach(card => {{
            card.addEventListener("dragstart", event => {{
                draggedCard = card;
                card.classList.add("dragging");
                event.dataTransfer.effectAllowed = "move";
            }});
            card.addEventListener("dragend", () => {{
                card.classList.remove("dragging");
                canvas.querySelectorAll(".canvas-card").forEach(item => item.classList.remove("drag-over"));
                draggedCard = null;
            }});
            card.addEventListener("dragover", event => {{
                event.preventDefault();
                if (draggedCard && draggedCard !== card) {{
                    card.classList.add("drag-over");
                }}
            }});
            card.addEventListener("dragleave", () => card.classList.remove("drag-over"));
            card.addEventListener("drop", event => {{
                event.preventDefault();
                if (!draggedCard || draggedCard === card) return;
                const bounds = card.getBoundingClientRect();
                const before = event.clientX < bounds.left + bounds.width / 2;
                canvas.insertBefore(draggedCard, before ? card : card.nextSibling);
                card.classList.remove("drag-over");
            }});
        }});
        </script>
    </body>
    </html>
    """

    components.html(html, height=520, scrolling=False)


def render_canvas():
    has_content = bool(
        st.session_state.canvas_items
        or st.session_state.canvas_objects
    )
    if not has_content:
        st.markdown(
            """
            <div style="padding:38px 18px 18px;text-align:center;color:#667085;">
                <div style="width:44px;height:44px;margin:0 auto 10px;border-radius:12px;
                            display:grid;place-items:center;background:#edf2ff;color:#315efb;
                            border:1px solid #dbe5ff;font-size:18px;">◆</div>
                <div style="color:#344054;font-size:15px;font-weight:800;margin-bottom:5px;">
                    Build your analysis
                </div>
                <div style="max-width:360px;margin:auto;font-size:12px;line-height:1.45;">
                    Select a generated dataset, choose a visual, and add it here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for index in range(0, max(
        len(st.session_state.canvas_items),
        len(st.session_state.canvas_objects),
    ), 2):
        left_card = (
            st.session_state.canvas_items[index]
            if index < len(st.session_state.canvas_items)
            else None
        )
        right_card = (
            st.session_state.canvas_items[index + 1]
            if index + 1 < len(st.session_state.canvas_items)
            else None
        )
        object_card = (
            st.session_state.canvas_objects[index]
            if index < len(st.session_state.canvas_objects)
            else None
        )

        columns = st.columns(2)
        for column, card in zip(columns, [left_card, right_card]):
            if card is None:
                continue
            with column:
                with st.container(border=True):
                    st.caption(card["title"])
                    render_native_chart(card["df"], card["viz"], height=260)

        if object_card:
            with st.container(border=True):
                for text_object in st.session_state.canvas_objects:
                    if text_object["type"] == "text":
                        text_value = st.text_area(
                            "Text",
                            value=text_object["text"],
                            key=f"canvas_text_{text_object['id']}",
                            label_visibility="collapsed",
                        )
                        text_object["text"] = text_value
                    elif text_object["type"] == "rectangle":
                        st.markdown(
                            '<div style="height:110px;border:2px solid #315efb;'
                            'border-radius:8px;background:#edf2ff;"></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="width:110px;height:110px;margin:auto;'
                            'border:2px solid #315efb;border-radius:50%;'
                            'background:#edf2ff;"></div>',
                            unsafe_allow_html=True,
                        )


# Streamlit columns own the three-panel layout. Avoid opening HTML divs in one
# st.markdown call and closing them in another: Streamlit sanitizes each block
# independently, which previously rendered empty fixed-height bordered boxes.
st.markdown(
    """
    <div class="decision-header">
        <div class="decision-title">DecisionIT</div>
        <div class="decision-underline"></div>
        <div class="decision-subtitle">Your Self-Serve Analytics Platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, center, right = st.columns([0.22, 0.50, 0.28], gap="small")

with left:
    with st.container(border=True):
        st.markdown('<div class="pane-title"><span>Datasets</span></div>', unsafe_allow_html=True)

        datasets = st.session_state.generated_datasets
        if not datasets:
            st.info("Generated datasets will appear here.")
        else:
            for dataset in reversed(datasets):
                active = dataset["id"] == st.session_state.active_dataset_id
                card_class = "dataset-card active" if active else "dataset-card"
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="dataset-title">{escape(dataset["title"])}</div>
                        <div class="dataset-meta">{dataset["rows"]:,} rows · {dataset["columns"]} columns</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Open",
                    key=f"open_dataset_{dataset['id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_dataset_id = dataset["id"]
                    st.session_state.center_view_mode = "dataset"
                    st.rerun()

            active_dataset = next(
                (
                    item for item in datasets
                    if item["id"] == st.session_state.active_dataset_id
                ),
                datasets[-1],
            )
            st.session_state.active_dataset_id = active_dataset["id"]

            with st.expander("Dataset preview", expanded=False):
                st.dataframe(active_dataset["df"].head(100), use_container_width=True)

with center:
    if st.session_state.center_view_mode == "dataset" and st.session_state.active_dataset_id:
        active_dataset = next(
            (
                item for item in st.session_state.generated_datasets
                if item["id"] == st.session_state.active_dataset_id
            ),
            None,
        )
        if active_dataset:
            st.markdown(
                f'<div class="pane-title"><span>{escape(active_dataset["title"])}</span>'
                f'<span>{active_dataset["rows"]:,} rows</span></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                active_dataset["df"],
                use_container_width=True,
                height=560,
            )
    elif st.session_state.center_view_mode == "whiteboard" and st.session_state.whiteboard_items:
        st.markdown('<div class="pane-title"><span>Whiteboard</span></div>', unsafe_allow_html=True)

        def render_whiteboard_item(index: int, item: dict):
            width = item.get("width", "Full")
            height = item.get("height", 280)
            with st.container(border=True):
                title_col, size_col = st.columns([0.68, 0.32])
                with title_col:
                    st.markdown(f"**{escape(item['title'])}**")
                with size_col:
                    with st.popover("ⓘ", help="Visual adjustments"):
                        item["width"] = st.selectbox(
                            "Width",
                            ["Half", "Full"],
                            index=0 if width == "Half" else 1,
                            key=f"whiteboard_width_{item['id']}",
                        )
                        item["height"] = st.slider(
                            "Visual height",
                            min_value=220,
                            max_value=600,
                            value=height,
                            step=20,
                            key=f"whiteboard_height_{item['id']}",
                        )
                        if st.button(
                            "Remove visual",
                            key=f"remove_whiteboard_{index}",
                            use_container_width=True,
                        ):
                            st.session_state.whiteboard_items.pop(index)
                            if not st.session_state.whiteboard_items:
                                st.session_state.center_view_mode = "empty"
                            st.rerun()
                render_native_chart(
                    item["df"],
                    item["viz"],
                    height=item["height"],
                )

        index = 0
        while index < len(st.session_state.whiteboard_items):
            item = st.session_state.whiteboard_items[index]
            if item.get("width", "Full") == "Half":
                next_index = index + 1
                if (
                    next_index < len(st.session_state.whiteboard_items)
                    and st.session_state.whiteboard_items[next_index].get("width", "Full") == "Half"
                ):
                    columns = st.columns(2)
                    for column, (item_index, paired_item) in zip(
                        columns,
                        [
                            (index, item),
                            (
                                next_index,
                                st.session_state.whiteboard_items[next_index],
                            ),
                        ],
                    ):
                        with column:
                            render_whiteboard_item(item_index, paired_item)
                    index += 2
                else:
                    columns = st.columns([0.5, 0.5])
                    with columns[0]:
                        render_whiteboard_item(index, item)
                    index += 1
            else:
                render_whiteboard_item(index, item)
                index += 1
    elif st.session_state.active_dataset_id:
        active_dataset = next(
            (
                item for item in st.session_state.generated_datasets
                if item["id"] == st.session_state.active_dataset_id
            ),
            None,
        )
        if active_dataset:
            st.markdown(
                f'<div class="pane-title"><span>{escape(active_dataset["title"])}</span>'
                f'<span>{active_dataset["rows"]:,} rows</span></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                active_dataset["df"],
                use_container_width=True,
                height=560,
            )
    else:
        st.markdown(
            """
            <div style="height:calc(100vh - 142px);display:flex;flex-direction:column;
                        align-items:center;justify-content:flex-start;padding-top:26px;
                        text-align:center;color:#667085;">
                <div style="width:44px;height:44px;margin-bottom:10px;border-radius:12px;
                            display:grid;place-items:center;background:#edf2ff;color:#315efb;
                            border:1px solid #dbe5ff;font-size:18px;">◆</div>
                <div style="color:#344054;font-size:15px;font-weight:800;margin-bottom:5px;">
                    Build your analysis
                </div>
                <div style="max-width:360px;font-size:12px;line-height:1.45;">
                    Select a generated dataset, choose a visual, and add it here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="pane-title"><span>Conversation</span><span>Live</span></div>', unsafe_allow_html=True)
    with st.container(height=650, border=True):
        with st.container(height=430, border=False):
            st.markdown('<div class="chat-message-padding-top"></div>', unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user">{escape(message["content"])}</div>',
                        unsafe_allow_html=True,
                    )
                elif message.get("declined"):
                    st.markdown(
                        f'<div class="chat-assistant"><div class="chat-status">Guardrail</div>'
                        f'<div class="chat-message-summary">{escape(message["content"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    status = "Enterprise context" if message.get("context_on") else "Ungrounded schema"
                    summary = escape(message.get("summary", ""))
                    consideration_items = message.get("plan", {}).get("grounding_considerations", [])
                    grounding = ""
                    if consideration_items:
                        grounding_items = "".join(
                            f"<li>{escape(str(item))}</li>" for item in consideration_items[:4]
                        )
                        grounding = (
                            '<div class="grounding-mini"><strong>Grounding</strong>'
                            f"<ul>{grounding_items}</ul></div>"
                        )
                    st.markdown(
                        f'<div class="chat-assistant">'
                        f'<div class="chat-status">{status}</div>'
                        f'<div class="chat-message-summary">{summary}</div>'
                        f"{grounding}</div>",
                        unsafe_allow_html=True,
                    )
                    if message.get("df") is not None and not message["df"].empty:
                        render_native_chart(message["df"], message.get("viz", {}), height=230)
                        if st.button(
                            "Add to whiteboard",
                            key=f"add_message_visual_{message.get('id', len(st.session_state.messages))}",
                            use_container_width=True,
                        ):
                            st.session_state.whiteboard_items.append({
                                "id": f"whiteboard_{len(st.session_state.whiteboard_items) + 1}",
                                "title": message["plan"].get("metric_name", "Generated visual"),
                                "df": message["df"].copy(),
                                "viz": message["viz"].copy(),
                                "width": "Full",
                                "height": 280,
                            })
                            st.session_state.center_view_mode = "whiteboard"
                            st.rerun()
                    with st.expander("SQL & details"):
                        st.code(message.get("sql", ""), language="sql")
                        df = message.get("df")
                        if df is not None:
                            st.dataframe(df, use_container_width=True)
            st.markdown('<div class="chat-message-padding-bottom"></div>', unsafe_allow_html=True)

        if st.session_state.pending_grounding is not None:
            pending = st.session_state.pending_grounding
            plan = pending["bundle"]["plan"]
            st.markdown(
                '<div class="chat-assistant">'
                '<div class="chat-status">Grounding verification</div>'
                f'<strong>{escape(plan["metric_name"])}</strong>'
                f'<div class="chat-message-summary">{escape(plan["business_definition"])}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            confirm = c1.button("Confirm", type="primary", use_container_width=True)
            cancel = c2.button("Cancel", use_container_width=True)
            if confirm:
                exec_res = engine.generate_and_execute_sql(
                    pending["question"], pending["bundle"]
                )
                if exec_res["success"]:
                    dataset_id = f"dataset_{len(st.session_state.generated_datasets) + 1}"
                    title = plan["metric_name"]
                    st.session_state.generated_datasets.append({
                        "id": dataset_id,
                        "title": title,
                        "df": exec_res["df"].copy(),
                        "rows": len(exec_res["df"]),
                        "columns": len(exec_res["df"].columns),
                        "sql": exec_res["sql"],
                        "viz": exec_res["viz"],
                        "cot": exec_res["chain_of_thought"],
                    })
                    st.session_state.active_dataset_id = dataset_id
                    st.session_state.messages.append({
                        "id": len(st.session_state.messages),
                        "role": "assistant",
                        "context_on": pending["bundle"]["context_on"],
                        "summary": exec_res["viz"]["executive_summary"],
                        "df": exec_res["df"],
                        "viz": exec_res["viz"],
                        "sql": exec_res["sql"],
                        "plan": plan,
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
            if cancel:
                st.session_state.pending_grounding = None
                st.rerun()

        toggles = st.columns(2)
        with toggles[0]:
            semantic_context_on = st.toggle(
                "Semantic layer",
                value=st.session_state.semantic_context_on,
                key="conversation_semantic_layer",
            )
            st.session_state.semantic_context_on = semantic_context_on
        with toggles[1]:
            hitl_grounding_on = st.toggle(
                "Confirm grounding",
                value=st.session_state.hitl_grounding_on,
                key="conversation_hitl_grounding",
                help="Pause before SQL execution to review the grounded metric and table plan.",
            )
            st.session_state.hitl_grounding_on = hitl_grounding_on

        question = st.chat_input(
            "Ask about KPIs, tables, dimensions, or filters...",
            key="workspace_chat",
        )

if question:
    prior_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    bundle = engine.create_grounding_plan(
        question,
        context_on=st.session_state.semantic_context_on,
        chat_history=prior_history,
    )

    if "semantic_exploration" in bundle:
        exploration = bundle["semantic_exploration"]
        dataset_id = f"dataset_{len(st.session_state.generated_datasets) + 1}"
        df = exploration["df"].copy()
        viz = {"chart_type": "table", "x_col": "", "y_col": "", "chart_title": bundle["plan"]["metric_name"]}
        st.session_state.generated_datasets.append({
            "id": dataset_id,
            "title": bundle["plan"]["metric_name"],
            "df": df,
            "rows": len(df),
            "columns": len(df.columns),
            "sql": exploration["sql"],
            "viz": viz,
            "cot": [],
        })
        st.session_state.active_dataset_id = dataset_id
        st.session_state.messages.append({
            "id": len(st.session_state.messages),
            "role": "assistant",
            "context_on": st.session_state.semantic_context_on,
            "summary": exploration["summary"],
            "df": df,
            "viz": viz,
            "sql": exploration["sql"],
            "plan": bundle["plan"],
        })
        st.rerun()

    plan = bundle["plan"]
    if not plan.get("is_answerable", True):
        st.session_state.messages.append({
            "id": len(st.session_state.messages),
            "role": "assistant",
            "declined": True,
            "content": f"Out of scope: {plan.get('decline_reason', 'This question is outside the lakehouse.')}",
        })
        st.rerun()

    if st.session_state.hitl_grounding_on:
        st.session_state.pending_grounding = {"question": question, "bundle": bundle}
    else:
        exec_res = engine.generate_and_execute_sql(question, bundle)
        if exec_res["success"]:
            dataset_id = f"dataset_{len(st.session_state.generated_datasets) + 1}"
            st.session_state.generated_datasets.append({
                "id": dataset_id,
                "title": plan["metric_name"],
                "df": exec_res["df"].copy(),
                "rows": len(exec_res["df"]),
                "columns": len(exec_res["df"].columns),
                "sql": exec_res["sql"],
                "viz": exec_res["viz"],
                "cot": exec_res["chain_of_thought"],
            })
            st.session_state.active_dataset_id = dataset_id
            st.session_state.messages.append({
                "id": len(st.session_state.messages),
                "role": "assistant",
                "context_on": bundle["context_on"],
                "summary": exec_res["viz"]["executive_summary"],
                "df": exec_res["df"],
                "viz": exec_res["viz"],
                "sql": exec_res["sql"],
                "plan": plan,
            })
        else:
            st.session_state.messages.append({
                "id": len(st.session_state.messages),
                "role": "assistant",
                "declined": True,
                "content": f"SQL execution failed after retries: `{exec_res['error']}`",
            })
    st.rerun()
