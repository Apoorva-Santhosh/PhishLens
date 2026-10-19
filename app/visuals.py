"""Charts and escaped highlight markup; never render submitted HTML directly."""

from heapq import heappop, heappush
from html import escape

import plotly.graph_objects as go

from phishlens.schemas import PredictionResponse

CYAN = "#4ecbdc"
MUTED = "#8092ab"
COLORS = {"legitimate": "#53c89b", "suspicious": "#eab65c", "phishing": "#f27483"}


def risk_chart(result: PredictionResponse | None) -> go.Figure:
    """A segmented radial meter inspired by the reference's central instrument."""
    score = result.risk_score if result else 0
    color = COLORS[result.label] if result else CYAN
    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=[0.24] * 60,
            theta=[i * 6 for i in range(60)],
            width=[4] * 60,
            base=[0.77] * 60,
            marker_color=[
                color if result and i < round(score * 0.6) else "#25374e" for i in range(60)
            ],
            marker_line_width=0,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[1.10] * 121,
            theta=[i * 3 for i in range(121)],
            mode="lines",
            line={"color": "#2b4862", "width": 1},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=350,
        margin={"l": 22, "r": 22, "t": 18, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": False, "range": [0, 1.16]},
            "angularaxis": {"visible": False, "rotation": 90, "direction": "clockwise"},
        },
        annotations=[
            {
                "text": str(score) if result else "—",
                "x": 0.5,
                "y": 0.54,
                "showarrow": False,
                "font": {"size": 62, "color": "#e0e8f3"},
            },
            {
                "text": "RISK / 100" if result else "AWAITING INPUT",
                "x": 0.5,
                "y": 0.37,
                "showarrow": False,
                "font": {"size": 11, "color": MUTED},
            },
        ],
    )
    return fig


def feature_chart(result: PredictionResponse) -> go.Figure:
    """Compare signed model contributions while keeping the strongest signals readable."""
    features = sorted(result.top_features, key=lambda item: abs(item.contribution))[-8:]
    fig = go.Figure(
        go.Bar(
            x=[item.contribution for item in features],
            y=[item.name.replace("_", " ") for item in features],
            orientation="h",
            marker_color=["#f27483" if item.contribution > 0 else CYAN for item in features],
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": MUTED, "size": 11, "family": "monospace"},
        margin={"l": 5, "r": 10, "t": 5, "b": 30},
        xaxis={
            "gridcolor": "#26354a",
            "zerolinecolor": "#53647c",
            "title": "Model contribution",
        },
        yaxis={"automargin": True},
    )
    return fig


def highlighted_text(result: PredictionResponse) -> str:
    """Resolve overlapping term spans into disjoint segments without duplicate text."""
    text = result.cleaned_text or ""
    spans = [span for span in result.highlights if 0 <= span.start < span.end <= len(text)]
    bounds = sorted({0, len(text), *(span.start for span in spans), *(s.end for s in spans)})
    parts = []
    ordered = sorted(enumerate(spans), key=lambda item: item[1].start)
    active = []
    cursor = 0
    for start, end in zip(bounds, bounds[1:]):
        segment = escape(text[start:end])
        while cursor < len(ordered) and ordered[cursor][1].start <= start:
            index, span = ordered[cursor]
            heappush(active, (-abs(span.contribution), index, span))
            cursor += 1
        while active and active[0][2].end <= start:
            heappop(active)
        if active:
            strongest = active[0][2]
            style = {
                "raises_risk": "risk-up",
                "lowers_risk": "risk-down",
                "neutral": "risk-neutral",
            }[strongest.direction]
            segment = f'<mark class="{style}">{segment}</mark>'
        parts.append(segment)
    return '<div class="email-preview">' + "".join(parts) + "</div>"
