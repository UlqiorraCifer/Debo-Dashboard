"""
dashboard.py -- single-screen DEBO results dashboard (no scrolling).

Reads results.csv (made by simulation.py) and shows the four evaluation
metrics from Chapter III as charts with the numbers printed on them.

Run locally:   python -m streamlit run dashboard.py
The charts are sized from the browser window height, so the page fits any screen.

Needs: streamlit, pandas, numpy, plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = "results.csv"

STRATEGIES = ["DEBO", "Always-On", "Fixed Duty-Cycle", "Greedy Threshold"]
SHORT = {"DEBO": "DEBO", "Always-On": "Always-On",
         "Fixed Duty-Cycle": "Fixed DC", "Greedy Threshold": "Greedy"}
COLORS = {"DEBO": "#16a34a", "Always-On": "#ef4444",
          "Fixed Duty-Cycle": "#3b82f6", "Greedy Threshold": "#f59e0b"}
STAGES = ["vegetative", "reproductive", "ripening"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="DEBO WSN Scheduling Dashboard",
                   layout="wide", initial_sidebar_state="collapsed")

# Chart heights (px).  Size "S" is tuned for a laptop browser window
# (about 650 px of page height) and never scrolls there.  The "Screen size"
# buttons in the top bar enlarge everything for bigger monitors.
SIZES = {"S": 1.0, "M": 1.25, "L": 1.5, "XL": 1.9}
H_MAIN, H_MINI, H_BOT = 295, 145, 205

# ------------------------------------------------------------------- CSS
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer, #MainMenu,
      [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
      .block-container {padding: 0.5rem 1.1rem 0.2rem 1.1rem !important; max-width: 100% !important;}
      [data-testid="stVerticalBlock"] {gap: 0.3rem !important;}
      [data-testid="stHorizontalBlock"] {gap: 0.6rem !important;}
      [data-testid="stWidgetLabel"] p, label p {font-size: 0.72rem !important; margin: 0 !important; opacity: .75;}
      [data-testid="stButtonGroup"] button, [data-testid="stPills"] button {
          padding: 0.05rem 0.55rem !important; min-height: 1.65rem !important; font-size: 0.74rem !important;}
      [data-testid="stSlider"] {padding-top: 0 !important;}
      /* chart heights follow the window height -> always one screen */
      .app-title {font-size: 1.05rem; font-weight: 700; line-height: 1.15; margin: 0;}
      .app-sub {font-size: 0.7rem; opacity: .65; margin: 0;}
      /* page background */
.stApp {background: radial-gradient(circle at 15% 0%, #16264a 0%, #0b1020 60%);}

/* every chart becomes a card */
[class*="st-key-soc"], [class*="st-key-mini_"], [class*="st-key-bot_"] {
    background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
    border: 1px solid rgba(255,255,255,.10);
    border-top: 3px solid var(--accent, #64748b);
    border-radius: 12px;
    padding: 4px 8px 0 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,.35);
    transition: transform .15s ease, box-shadow .15s ease;
}
[class*="st-key-soc"]:hover, [class*="st-key-mini_"]:hover, [class*="st-key-bot_"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,.5);
}

/* one accent color per metric */
.st-key-soc    {--accent: #16a34a;}
.st-key-mini_1 {--accent: #ef4444;}
.st-key-mini_2 {--accent: #f59e0b;}
.st-key-mini_3 {--accent: #22c55e;}
.st-key-mini_4 {--accent: #3b82f6;}
.st-key-bot_1  {--accent: #a855f7;}
.st-key-bot_2  {--accent: #f59e0b;}
.st-key-bot_3  {--accent: #06b6d4;}

/* livelier title */
.app-title {
    font-size: 1.25rem !important;
    background: linear-gradient(90deg, #4ade80, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "overcast" not in df.columns:
        df["overcast"] = 0
    df["date"] = df["timestamp"].dt.normalize()
    df["month"] = df["timestamp"].dt.month.astype("int8")
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    # penalty from Chapter III:  W(t) * (1 - x_i(t))
    df["pen"] = (df["penalty_weight"] * (1 - df["active"])).astype("float32")
    df["strategy"] = df["strategy"].astype("category")
    df["growth_stage"] = df["growth_stage"].astype("category")
    return df


def compute_metrics(f: pd.DataFrame, order: list) -> pd.DataFrame:
    g = f.groupby("strategy", observed=True)
    m = pd.DataFrame({
        "blackout": g["blackout"].sum(),
        "penalty": g["pen"].sum(),
        "dcr": f[f["growth_stage"] == "reproductive"]
        .groupby("strategy", observed=True)["active"].mean() * 100,
        "soc_oc": f[f["overcast"] == 1]
        .groupby("strategy", observed=True)["soc"].mean() * 100,
    })
    return m.reindex(order)


def improvement(m, base, col, lower_better):
    b, d = m.loc[base, col], m.loc["DEBO", col]
    if pd.isna(b) or pd.isna(d) or b == 0:
        return np.nan
    return ((b - d) if lower_better else (d - b)) / b * 100


# --------------------------------------------------------------- figures
def base_layout(fig, height, title=None, legend=False, margin=None):
    fig.update_layout(
        height=height,
        margin=margin or dict(l=8, r=8, t=26 if title else 6, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
        showlegend=legend,
        title=dict(text=title, x=0.01, y=0.97, xanchor="left", yanchor="top",
                   font=dict(size=12)) if title else None,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02,
                    yanchor="bottom", font=dict(size=10)),
        bargap=0.25,
    )
    return fig


def fmt(v, kind):
    if pd.isna(v):
        return "n/a"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{v:.1f}%"
    return f"{v:.1f}"


def metric_bar(m, col, title, kind):
    vals = m[col]
    top = np.nanmax(vals.to_numpy(dtype=float)) if vals.notna().any() else 1
    top = top if top > 0 else 1
    fig = go.Figure(go.Bar(
        x=[SHORT[s] for s in m.index],
        y=vals.fillna(0),
        text=[fmt(v, kind) for v in vals],
        textposition="outside", cliponaxis=False,
        marker_color=[COLORS[s] for s in m.index],
        marker_line=dict(color=["#ffffff" if s == "DEBO" else "rgba(0,0,0,0)" for s in m.index],
                         width=2),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(visible=False, range=[0, top * 1.3])
    fig.update_xaxes(tickfont=dict(size=9), showgrid=False)
    return base_layout(fig, H_MINI, title, margin=dict(l=6, r=6, t=24, b=4))


def soc_figure(f, order):
    daily = (f.groupby(["date", "strategy"], observed=True)["soc"]
             .mean().unstack("strategy") * 100)
    fig = go.Figure()

    # growth-stage ribbon + faint tint (all three stages, every segment labeled)
    stage_style = {"vegetative": ("#22c55e", "Vegetative"),
                   "reproductive": ("#f59e0b", "Reproductive"),
                   "ripening": ("#94a3b8", "Ripening")}

    stage = (f.drop_duplicates("date").set_index("date")["growth_stage"]
             .sort_index().astype(str))
    run_id = (stage != stage.shift()).cumsum()
    for _, run in stage.groupby(run_id):
        name = run.iloc[0]
        if name not in stage_style:
            continue
        color, label = stage_style[name]
        x0 = run.index[0]
        x1 = run.index[-1] + pd.Timedelta(days=1)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.07, line_width=0)
        fig.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1,
                      y0=0.92, y1=1.0, fillcolor=color, opacity=0.9, line_width=0)
        if (x1 - x0).days >= 20:
            fig.add_annotation(x=x0 + (x1 - x0) / 2, xref="x", y=0.96, yref="paper",
                               text=label, showarrow=False,
                               font=dict(size=9, color="white"))

    # SoC lines
    for s in order:
        if s in daily.columns:
            fig.add_trace(go.Scatter(
                x=daily.index, y=daily[s].round(1), name=SHORT[s], mode="lines",
                line=dict(color=COLORS[s], width=3 if s == "DEBO" else 1.8),
                hovertemplate=SHORT[s] + ": %{y:.1f}%<extra></extra>"))

    fig.update_yaxes(range=[0, 118], tickvals=[0, 25, 50, 75, 100], ticksuffix="%",
                     gridcolor="rgba(128,128,128,.2)")
    fig.update_xaxes(showgrid=False, tickformat="%b")
    fig.update_layout(hovermode="x unified")
    return base_layout(fig, H_MAIN,
                       "🔋 Battery state of charge (daily mean)",
                       legend=True, margin=dict(l=8, r=8, t=30, b=6))


def improvement_figure(m, order):
    metrics = [("blackout", "Blackout", True), ("penalty", "Penalty P", True),
               ("dcr", "Repro. DCR", False), ("soc_oc", "Overcast SoC", False)]
    baselines = [s for s in order if s != "DEBO"]
    fig = go.Figure()
    if "DEBO" not in order or not baselines:
        fig.add_annotation(text="Select DEBO and at least one baseline<br>to see the improvement",
                           showarrow=False, font=dict(size=11), opacity=.7)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return base_layout(fig, H_BOT, "📈 DEBO improvement (%)")
    for b in baselines:
        vals = [improvement(m, b, c, lb) for c, _, lb in metrics]
        fig.add_trace(go.Bar(
            name=SHORT[b], y=[lab for _, lab, _ in metrics],
            x=[0 if pd.isna(v) else v for v in vals], orientation="h",
            text=["n/a" if pd.isna(v) else f"{v:+.1f}%" for v in vals],
            textposition="outside", cliponaxis=False, marker_color=COLORS[b],
            textfont=dict(size=9),
            hovertemplate=SHORT[b] + ": %{text}<extra></extra>"))
       fig.update_xaxes(visible=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    return base_layout(fig, H_BOT, "📈 DEBO improvement (%)",
                       legend=True, margin=dict(l=6, r=44, t=26, b=4))


def stage_figure(f, order):
    pen = (f.groupby(["growth_stage", "strategy"], observed=True)["pen"]
           .sum().unstack("strategy").reindex(STAGES))
    fig = go.Figure()
    for s in order:
        if s in pen.columns:
            vals = pen[s].fillna(0)
            fig.add_trace(go.Bar(
                name=SHORT[s], x=[x.capitalize() for x in STAGES], y=vals,
                text=[f"{v:,.0f}" for v in vals], textposition="outside",
                cliponaxis=False, marker_color=COLORS[s],
                textfont=dict(size=8),
                hovertemplate=SHORT[s] + ": %{text}<extra></extra>"))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_layout(barmode="group", bargap=0.18)
    return base_layout(fig, H_BOT, "🌱 Weighted penalty by growth stage (lower is better)",
                       margin=dict(l=6, r=6, t=26, b=4))


def hour_figure(f, order):
    binned = f.assign(hbin=(f["hour"] // 2) * 2)
    by_hour = (binned.groupby(["hbin", "strategy"], observed=True)["active"]
               .mean().unstack("strategy") * 100)
    fig = go.Figure()
    for s in order:
        if s in by_hour.columns:
            fig.add_trace(go.Scatter(
                x=by_hour.index, y=by_hour[s].round(1), mode="lines", name=SHORT[s],
                line=dict(color=COLORS[s], width=2.6 if s == "DEBO" else 1.6),
                hovertemplate=SHORT[s] + ": %{y:.0f}%<extra></extra>"))
    fig.update_yaxes(range=[0, 105], ticksuffix="%", gridcolor="rgba(128,128,128,.2)")
    fig.update_xaxes(title=None, showgrid=False,
                     ticktext=[f"{h:02d}h" for h in range(0, 24, 4)],
                     tickvals=list(range(0, 24, 4)))
    fig.update_layout(hovermode="x unified")
    return base_layout(fig, H_BOT, "🕐 Share of hours transmitting, by time of day (2-hour bins)",
                       margin=dict(l=6, r=6, t=26, b=6))


def show(fig):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:                          # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, config=cfg)


# ------------------------------------------------------------------ page
try:
    data = load_data(RESULTS_PATH)
except FileNotFoundError:
    st.error(f"Could not find {RESULTS_PATH}. Run simulation.py first.")
    st.stop()

available = [s for s in STRATEGIES if s in set(data["strategy"].astype(str))]

c_title, c_strat, c_node, c_date, c_size = st.columns([1.35, 2.5, 3.0, 2.0, 1.15],
                                                      vertical_alignment="bottom")
with c_title:
    st.markdown('<p class="app-title">DEBO · Rice-field WSN</p>'
                '<p class="app-sub">San Ildefonso · 2024</p>',
                unsafe_allow_html=True)

with c_strat:
    if hasattr(st, "pills"):
        picked = st.pills("Strategy (click to show / hide)", available,
                          selection_mode="multi", default=available,
                          format_func=lambda s: SHORT[s], key="strat")
    else:
        picked = st.multiselect("Strategy", available, default=available,
                                format_func=lambda s: SHORT[s], key="strat")
order = [s for s in available if s in (picked or available)]

nodes = sorted(data["node_id"].unique().tolist())
with c_node:
    if hasattr(st, "pills"):
        node = st.pills("Node", ["All"] + [str(n) for n in nodes],
                        selection_mode="single", default="All", key="node")
    else:
        node = st.selectbox("Node", ["All"] + [str(n) for n in nodes], key="node")
node = node or "All"

with c_date:
    m1, m2 = st.select_slider("Months", options=MONTHS, value=("Jan", "Dec"), key="months")

from streamlit_js_eval import streamlit_js_eval

win_h = streamlit_js_eval(js_expressions="window.parent.innerHeight", key="win_h")

if win_h:
    SC = min(max((win_h - 150) / 400, 0.8), 2.2)
else:
    SC = 1.0



H_MAIN, H_MINI, H_BOT = int(295 * SC), int(145 * SC), int(205 * SC)

lo, hi = MONTHS.index(m1) + 1, MONTHS.index(m2) + 1
mask = data["strategy"].isin(order) & data["month"].between(lo, hi)
if node != "All":
    mask &= data["node_id"] == int(node)
f = data[mask]

if f.empty:
    st.warning("No data for this selection.")
    st.stop()

metrics = compute_metrics(f, order)

left, right = st.columns([1.7, 1.3])
with left:
    with st.container(key="soc"):
        show(soc_figure(f, order))
with right:
    with st.container():
        r1a, r1b = st.columns(2)
        r2a, r2b = st.columns(2)
        with r1a, st.container(key="mini_1"):
            show(metric_bar(metrics, "blackout", "Blackout hours ↓", "int"))
        with r1b, st.container(key="mini_2"):
            show(metric_bar(metrics, "penalty", "Weighted penalty P ↓", "int"))
        with r2a, st.container(key="mini_3"):
            show(metric_bar(metrics, "dcr", "Reproductive continuity ↑", "pct"))
        with r2b, st.container(key="mini_4"):
            show(metric_bar(metrics, "soc_oc", "SoC in overcast ↑", "pct"))

b1, b2, b3 = st.columns([1.1, 1.15, 1.15])
with b1, st.container(key="bot_1"):
    show(improvement_figure(metrics, order))
with b2, st.container(key="bot_2"):
    show(stage_figure(f, order))
with b3, st.container(key="bot_3"):
    show(hour_figure(f, order))
