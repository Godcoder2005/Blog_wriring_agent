from __future__ import annotations
import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List, Optional

import pandas as pd
import streamlit as st

# Import compiled LangGraph workflow
from backend import workflow


# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def safe_slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9 _-]+", "", title.strip()).lower()
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Stream if supported, else fallback to invoke."""
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Autonomous Blog Writer",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ Autonomous Blog Writing Agent")
st.caption("LangGraph • HITL • Memory • Multi-Agent System")


# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "logs" not in st.session_state:
    st.session_state.logs = []


# ---------------------------------------------------
# Sidebar Input
# ---------------------------------------------------
with st.sidebar:
    st.header("Generate Blog")

    topic = st.text_area(
        "Topic",
        placeholder="e.g. State of Multimodal LLMs in 2026",
        height=120,
    )

    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.divider()

    if st.button("🗑 Clear Logs", use_container_width=True):
        st.session_state.logs = []
        st.rerun()


# ---------------------------------------------------
# Run Workflow
# ---------------------------------------------------
if run_btn:

    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "approved": False,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "feedback": "",
    }

    status = st.status("Running AI agents...", expanded=True)

    logs: List[str] = []
    current_state: Dict[str, Any] = {}

    for kind, payload in try_stream(workflow, inputs):

        if kind == "updates":
            logs.append(json.dumps(payload, default=str)[:800])

        elif kind == "final":
            st.session_state.result = payload
            logs.append("FINAL STATE RECEIVED")

    st.session_state.logs.extend(logs)
    status.update(label="Done", state="complete")


# ---------------------------------------------------
# Tabs Layout
# ---------------------------------------------------
tab_plan, tab_evidence, tab_blog, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Blog", "🖼 Images", "🧾 Logs"]
)

result = st.session_state.result


# ---------------------------------------------------
# Plan Tab
# ---------------------------------------------------
with tab_plan:
    st.subheader("Generated Plan")

    if not result or not result.get("plan"):
        st.info("No plan available.")
    else:
        plan = result["plan"]
        if hasattr(plan, "model_dump"):
            plan = plan.model_dump()

        st.write("**Title:**", plan.get("blog_title"))
        st.write("**Audience:**", plan.get("audience"))
        st.write("**Tone:**", plan.get("tone"))

        tasks = plan.get("tasks", [])
        if tasks:
            df = pd.DataFrame(
                [
                    {
                        "ID": t["id"],
                        "Title": t["title"],
                        "Words": t["target_words"],
                        "Code": t["requires_code"],
                        "Research": t["requires_research"],
                    }
                    for t in tasks
                ]
            )
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------
# Evidence Tab
# ---------------------------------------------------
with tab_evidence:
    st.subheader("Evidence Sources")

    if not result or not result.get("evidence"):
        st.info("No evidence collected.")
    else:
        rows = []
        for e in result["evidence"]:
            if hasattr(e, "model_dump"):
                e = e.model_dump()
            rows.append(
                {
                    "Title": e.get("title"),
                    "URL": e.get("url"),
                    "Date": e.get("published_at"),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------
# Blog Tab
# ---------------------------------------------------
with tab_blog:
    st.subheader("Final Blog")

    if not result or not result.get("final"):
        st.warning("No blog generated.")
    else:
        blog_md = result["final"]
        st.markdown(blog_md)

        plan = result.get("plan")
        if hasattr(plan, "blog_title"):
            filename = safe_slug(plan.blog_title) + ".md"
        else:
            filename = "blog.md"

        st.download_button(
            "Download Markdown",
            blog_md.encode("utf-8"),
            filename,
            "text/markdown",
            use_container_width=True,
        )


# ---------------------------------------------------
# Images Tab
# ---------------------------------------------------
with tab_images:
    st.subheader("Generated Images")

    img_dir = Path("images")

    if not img_dir.exists():
        st.info("No images generated.")
    else:
        files = list(img_dir.glob("*"))
        if not files:
            st.info("Images folder empty.")
        else:
            for f in files:
                st.image(str(f), caption=f.name, use_container_width=True)


# ---------------------------------------------------
# Logs Tab
# ---------------------------------------------------
with tab_logs:
    st.subheader("Execution Logs")

    if not st.session_state.logs:
        st.info("No logs yet.")
    else:
        st.text_area(
            "Logs",
            value="\n\n".join(st.session_state.logs[-100:]),
            height=500,
        )
