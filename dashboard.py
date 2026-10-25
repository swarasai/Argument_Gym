"""Reusable Streamlit charts using the original saved speech summary keys."""
import pandas as pd
import streamlit as st

FIELDS = {
    "Speaking pace (WPM)": "pace_excluding_pauses_wpm",
    "Fillers per minute": "fillers_per_min",
    "Relative volume (dB)": "avg_volume_vs_baseline_db",
    "Pitch variety (semitones)": "pitch_variation_semitones",
    "Loudness variety (dB)": "loudness_variation_db",
    "Long pauses": "long_pauses",
}


def show_dashboard(records):
    if not records:
        st.info("Complete a session to see progress charts.")
        return
    rows = []
    for item in records:
        summary = item.get("summary", {})
        row = {"Date": item.get("when"), "Category": item.get("category", "Unknown")}
        row.update({label: summary.get(key) for label, key in FIELDS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    choices = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    selected = st.selectbox("Category", choices)
    if selected != "All":
        df = df[df["Category"] == selected]
    if df.empty:
        st.info("No sessions match this filter.")
        return
    st.metric("Recorded sessions", len(df))
    for label in FIELDS:
        st.markdown(f"**{label}**")
        chart = df[["Date", label]].dropna().set_index("Date")
        if chart.empty:
            st.caption("Not enough measured data yet.")
        else:
            st.line_chart(chart)
    st.caption("Filler recognition and pitch estimates are approximate. Higher or lower isn't always better.")
