"""Phase 6: single-page Streamlit demo UI. Paste text, hit submit, see it
annotated claim-by-claim with verdicts and a headline hallucination score.

Calls the FastAPI backend over HTTP (two small services, not one tangled app).
Run the backend first: uvicorn api.main:app --reload
Then: streamlit run ui/app.py
"""
import difflib
import html
import re

import requests
import streamlit as st

API_URL = "http://localhost:8000/verify"

VERDICT_COLORS = {
    "SUPPORTED": "#1e7d32",  # green
    "CONTRADICTED": "#c62828",  # red
    "UNVERIFIABLE": "#9e9e9e",  # gray
}
VERDICT_BG = {
    "SUPPORTED": "#e8f5e9",
    "CONTRADICTED": "#ffebee",
    "UNVERIFIABLE": "#f5f5f5",
}
# When multiple claims map to the same sentence, the most alarming verdict wins visually.
VERDICT_PRIORITY = {"CONTRADICTED": 0, "UNVERIFIABLE": 1, "SUPPORTED": 2}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def annotate_text(text: str, verdicts: list[dict]) -> str:
    sentences = split_sentences(text)
    rendered = []
    for sentence in sentences:
        best_verdict, best_score = None, 0.0
        for v in verdicts:
            score = difflib.SequenceMatcher(None, sentence.lower(), v["claim_text"].lower()).ratio()
            if score > best_score:
                best_score, best_verdict = score, v
        if best_verdict and best_score >= 0.35:
            verdict = best_verdict["verdict"]
            color = VERDICT_COLORS[verdict]
            bg = VERDICT_BG[verdict]
            tooltip = html.escape(f"{verdict}: {best_verdict['reasoning']}")
            rendered.append(
                f'<span title="{tooltip}" style="background-color:{bg};color:#111111;'
                f'border-bottom:2px solid {color};padding:1px 2px;border-radius:2px;">'
                f'{html.escape(sentence)}</span>'
            )
        else:
            rendered.append(html.escape(sentence))
    return " ".join(rendered)


st.set_page_config(page_title="MythCheck", page_icon="🏛️")
st.title("MythCheck")
st.caption("Paste AI-generated text about Greek/Norse/Egyptian mythology and see which claims hold up.")

text = st.text_area("Text to verify", height=200, placeholder="Paste a paragraph about mythology here...")

if st.button("Verify", type="primary") and text.strip():
    with st.spinner("Extracting and checking claims..."):
        try:
            response = requests.post(API_URL, json={"text": text}, timeout=120)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}. Is it running? ({e})")
            st.stop()

    summary = result["summary"]
    verdicts = result["verdicts"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Supported", f"{summary['supported_pct']}%")
    col2.metric("Contradicted", f"{summary['contradicted_pct']}%")
    col3.metric("Unverifiable", f"{summary['unverifiable_pct']}%")

    st.markdown("---")
    st.markdown(annotate_text(text, verdicts), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"Claims ({summary['total_claims']})")
    for v in verdicts:
        color = VERDICT_COLORS[v["verdict"]]
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:4px 10px;margin:6px 0;">'
            f'<b style="color:{color}">{v["verdict"]}</b> — {html.escape(v["claim_text"])}<br>'
            f'<small>{html.escape(v["reasoning"])}</small></div>',
            unsafe_allow_html=True,
        )
