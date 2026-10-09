"""
dashboard.py

Web dashboard for the DEBO (Dynamic Energy-Budgeted Operation) thesis
results. Reads results.csv (see generate_mock_results.py for the expected
shape) and displays the four evaluation metrics, with filters by strategy
and node.

Run locally with:
    streamlit run dashboard.py

This file does NOT run the MILP solver -- it only visualizes results that
were already computed and saved to results.csv. Swap in the real
simulation's output file (same column structure) once it's ready; no
changes to this file should be needed.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DEBO WSN Scheduling Dashboard", layout="wide")

RESULTS_PATH = "results.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    total_blackout_hours = df["blackout"].sum()
    weighted_penalty = (df["blackout"] * df["penalty_weight"]).sum()
    avg_soc_overcast = df["soc"].mean()  # placeholder: refine once overcast flag exists
    total_steps = len(df)
    continuity_rate = (
        1 - (df["blackout"].sum() / total_steps) if total_steps else 0
    )
    return {
        "Total Blackout Hours": int(total_blackout_hours),
        "Weighted Penalty Score": round(float(weighted_penalty), 2),
        "Avg SoC (overcast proxy)": round(float(avg_soc_overcast), 3),
        "Data Continuity Rate": f"{continuity_rate:.1%}",
    }


def main():
    st.title("DEBO: Energy-Harvesting WSN Scheduling Dashboard")
    st.caption(
        "San Ildefonso, Bulacan rice field deployment — "
        "simulation results viewer"
    )

    try:
        df = load_data(RESULTS_PATH)
    except FileNotFoundError:
        st.error(
            f"Could not find {RESULTS_PATH}. Run generate_mock_results.py "
            "first, or point RESULTS_PATH to your real simulation output."
        )
        return

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    strategies = sorted(df["strategy"].unique())
    selected_strategies = st.sidebar.multiselect(
        "Strategy", strategies, default=strategies
    )

    nodes = sorted(df["node_id"].unique())
    selected_nodes = st.sidebar.multiselect("Node(s)", nodes, default=nodes)

    date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
    date_range = st.sidebar.date_input(
        "Date range", value=(date_min.date(), date_max.date())
    )

    filtered = df[
        df["strategy"].isin(selected_strategies)
        & df["node_id"].isin(selected_nodes)
    ]
    if len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start)
            & (filtered["timestamp"].dt.date <= end)
        ]

    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    # --- Top-line metrics (for DEBO specifically, if present) ---
    st.subheader("Key Metrics")
    metric_cols = st.columns(len(selected_strategies))
    for col, strategy in zip(metric_cols, selected_strategies):
        strat_df = filtered[filtered["strategy"] == strategy]
        metrics = compute_metrics(strat_df)
        with col:
            st.markdown(f"**{strategy}**")
            for label, value in metrics.items():
                st.metric(label, value)

    # --- Strategy comparison chart: blackout hours ---
    st.subheader("Blackout Hours by Strategy")
    blackout_by_strategy = (
        filtered.groupby("strategy")["blackout"].sum().reindex(selected_strategies)
    )
    st.bar_chart(blackout_by_strategy)

    # --- SoC over time ---
    st.subheader("Average State of Charge Over Time")
    soc_over_time = (
        filtered.groupby(["timestamp", "strategy"])["soc"]
        .mean()
        .unstack("strategy")
    )
    st.line_chart(soc_over_time)

    # --- Weighted penalty by growth stage ---
    st.subheader("Weighted Penalty Score by Growth Stage")
    filtered = filtered.copy()
    filtered["weighted_penalty"] = filtered["blackout"] * filtered["penalty_weight"]
    penalty_by_stage = (
        filtered.groupby(["growth_stage", "strategy"])["weighted_penalty"]
        .sum()
        .unstack("strategy")
    )
    st.bar_chart(penalty_by_stage)

    # --- Raw data (optional, for debugging) ---
    with st.expander("View raw filtered data"):
        st.dataframe(filtered)


if __name__ == "__main__":
    main()
