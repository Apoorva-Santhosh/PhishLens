"""Streamlit interface: python -m streamlit run dashboard/app.py."""

import sys
from html import escape
from pathlib import Path

# Streamlit executes this file as a script; keep repository-root imports portable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard import client
from dashboard.visuals import COLORS, feature_chart, highlighted_text, risk_chart
from phishlens.config import MAX_EMAIL_LENGTH, MAX_URL_LENGTH
from phishlens.schemas import PredictionResponse


def show_result(result: PredictionResponse | None, kind: str) -> None:
    center, detail = st.columns([1.15, 1], gap="large")
    with center:
        st.caption("RISK ASSESSMENT")
        st.plotly_chart(
            risk_chart(result),
            theme=None,
            config={"displayModeBar": False},
            key=f"{kind}_risk",
            width="stretch",
        )
        if result:
            label = result.label.title()
            st.markdown(
                f'<div class="verdict" style="color:{COLORS[result.label]}">{label}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Calibrated probability {result.probability:.1%} · Model {result.model_version}"
            )
            if result.fallback_reason:
                st.info("Online metadata was unavailable. This result uses the offline model.")
            if result.mode == "online":
                st.warning("Experimental online pilot: detection quality is not validated.")
        else:
            st.markdown('<div class="verdict muted">Ready to analyze</div>', unsafe_allow_html=True)
            st.caption("Submit a URL or email to see its risk and the signals behind it.")
    with detail:
        st.caption("INFLUENTIAL SIGNALS")
        if result and result.top_features:
            st.plotly_chart(
                feature_chart(result),
                theme=None,
                config={"displayModeBar": False},
                key=f"{kind}_features",
                width="stretch",
            )
            st.caption(
                "Cyan lowers risk; red raises risk. Contributions are not risk-score points."
            )
            for feature in sorted(
                result.top_features, key=lambda item: abs(item.contribution), reverse=True
            )[:3]:
                direction = (
                    "Raises"
                    if feature.contribution > 0
                    else ("Lowers" if feature.contribution < 0 else "Does not change")
                )
                st.markdown(
                    f'<div class="signal"><span>{escape(feature.name.replace("_", " "))}'
                    f"</span><small>{direction} risk · {escape(feature.description)}</small></div>",
                    unsafe_allow_html=True,
                )
        elif result:
            st.info("No influential vocabulary terms were found in this input.")
        else:
            st.markdown(
                '<div class="empty-signals">No signals yet<br>'
                "<small>Model explanations will appear here after analysis.</small></div>",
                unsafe_allow_html=True,
            )
    if result and kind == "email":
        st.caption("EMAIL EVIDENCE · RED RAISES RISK / CYAN LOWERS RISK")
        st.markdown(highlighted_text(result), unsafe_allow_html=True)
        st.caption("Cleaned text; overlapping phrases use the strongest contribution.")


def analysis_tab(kind: str, available: bool) -> None:
    inputs, output = st.columns([0.78, 2], gap="large")
    with inputs:
        st.caption("ANALYSIS INPUT")
        with st.form(f"{kind}_form", border=False):
            mode = "offline"
            if kind == "url":
                value = st.text_input(
                    "URL", placeholder="https://example.com/about", max_chars=MAX_URL_LENGTH
                )
                mode = st.radio("Analysis mode", ["Offline", "Online"], horizontal=True).lower()
                st.caption("Offline: URL structure only. Online: adds public DNS/WHOIS metadata.")
                if mode == "online":
                    st.caption(
                        "Online analysis may use an experimental model or fall back offline."
                    )
            else:
                value = st.text_area(
                    "Email text",
                    placeholder="Paste the email to inspect…",
                    height=240,
                    max_chars=MAX_EMAIL_LENGTH,
                )
                st.caption("Links are not opened. Highlighting uses cleaned email text.")
            submitted = st.form_submit_button(
                "Analyze URL" if kind == "url" else "Analyze email",
                type="primary",
                disabled=not available,
                width="stretch",
            )
        if submitted:
            # Clear stale evidence before validating or submitting a new analysis.
            st.session_state.pop(f"{kind}_result", None)
            if not value.strip():
                st.warning("Enter a URL first." if kind == "url" else "Paste an email first.")
            else:
                try:
                    with st.spinner("Analyzing signals…"):
                        st.session_state[f"{kind}_result"] = client.analyze(kind, value, mode)
                except client.APIError as exc:
                    st.error(str(exc))
        if st.button("Clear analysis", key=f"clear_{kind}"):
            st.session_state.pop(f"{kind}_result", None)
        st.caption("Results describe model estimates. Review the evidence before acting.")
    with output:
        with st.container(border=True):
            show_result(st.session_state.get(f"{kind}_result"), kind)


def main() -> None:
    st.set_page_config(page_title="PhishLens", page_icon="◈", layout="wide")
    st.markdown(
        "<style>" + Path(__file__).with_name("style.css").read_text() + "</style>",
        unsafe_allow_html=True,
    )
    health = None
    try:
        health = client.health()
    except client.APIError:
        pass
    status = (
        "API offline"
        if health is None
        else ("API connected" if health.status == "ok" else "API degraded")
    )
    connection = "connected" if health and health.status == "ok" else "disconnected"
    top, refresh = st.columns([10, 1])
    with top:
        st.markdown(
            '<div class="identity"><span class="brand">◈ PhishLens</span>'
            f'<span class="connection {connection}">'
            f"● {status}</span></div>",
            unsafe_allow_html=True,
        )
    with refresh:
        if st.button("Refresh", help="Check API availability"):
            st.rerun()
    if health is None:
        st.info("API unavailable. Start it with `make api`, then refresh.")
    url_tab, email_tab = st.tabs(["URL Analysis", "Email Analysis"])
    for kind, tab, model in [("url", url_tab, "url_offline"), ("email", email_tab, "email")]:
        with tab:
            available = bool(health and health.models.get(model) and health.models[model].available)
            if health and not available:
                st.warning(
                    f"The {kind} model is unavailable. Load its trained artifact in the API."
                )
            analysis_tab(kind, available)


if __name__ == "__main__":
    main()
