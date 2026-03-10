"""Visualization tools for pitching mechanics analysis.

Generates plots for:
  - OBP benchmark distributions (box/violin plots)
  - Pitcher vs. benchmark comparison (radar/spider charts)
  - Metric percentile gauges
  - Joint angle time series
"""

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_benchmark_distributions(
    summary_df,
    title: str = "OBP Pitching Biomechanics Benchmarks",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create a horizontal box-like chart showing percentile ranges for key metrics.

    Args:
        summary_df: DataFrame from OBPBenchmarks.summary_table().
        title: Chart title.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for i, row in summary_df.iterrows():
        name = row["display_name"]

        # Draw the P10-P90 range as a thin line
        fig.add_trace(go.Scatter(
            x=[row["p10"], row["p90"]],
            y=[name, name],
            mode="lines",
            line=dict(color="lightgray", width=6),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Draw the P25-P75 (IQR) range as a thicker bar
        fig.add_trace(go.Scatter(
            x=[row["p25"], row["p75"]],
            y=[name, name],
            mode="lines",
            line=dict(color="#4A90D9", width=14),
            showlegend=False,
            hovertemplate=f"P25: {row['p25']:.1f} | P75: {row['p75']:.1f}<extra>{name}</extra>",
        ))

        # Draw the median as a dot
        fig.add_trace(go.Scatter(
            x=[row["p50"]],
            y=[name],
            mode="markers",
            marker=dict(color="white", size=8, line=dict(color="#4A90D9", width=2)),
            showlegend=False,
            hovertemplate=f"Median: {row['p50']:.1f} {row['unit']}<extra>{name}</extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        height=max(400, len(summary_df) * 35),
        margin=dict(l=300),
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def plot_pitcher_comparison(
    comparisons: list[dict],
    title: str = "Pitcher vs. OBP Benchmarks",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create a radar chart comparing pitcher metrics to benchmark medians.

    Args:
        comparisons: Output from OBPBenchmarks.compare_to_benchmarks().
        title: Chart title.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure object.
    """
    # Filter to metrics that have percentile ranks
    valid = [c for c in comparisons if c["percentile_rank"] is not None]
    if not valid:
        raise ValueError("No valid comparisons to plot")

    categories = [c["display_name"] for c in valid]
    pitcher_pcts = [c["percentile_rank"] for c in valid]

    fig = go.Figure()

    # Benchmark median line (always at 50th percentile by definition)
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(categories),
        theta=categories,
        fill="toself",
        fillcolor="rgba(74, 144, 217, 0.1)",
        line=dict(color="rgba(74, 144, 217, 0.3)", dash="dash"),
        name="OBP Median (50th pct)",
    ))

    # Pitcher values
    fig.add_trace(go.Scatterpolar(
        r=pitcher_pcts,
        theta=categories,
        fill="toself",
        fillcolor="rgba(255, 107, 53, 0.2)",
        line=dict(color="#FF6B35", width=2),
        name="Pitcher",
        text=[f"{c['value']:.1f} {c['unit']}" for c in valid],
        hovertemplate="%{theta}<br>Percentile: %{r:.0f}<br>Value: %{text}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75],
                ticktext=["25th", "50th", "75th"],
            )
        ),
        title=title,
        height=600,
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def plot_percentile_gauges(
    comparisons: list[dict],
    max_cols: int = 3,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create gauge charts for each metric showing percentile rank.

    Args:
        comparisons: Output from OBPBenchmarks.compare_to_benchmarks().
        max_cols: Max columns in the grid.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure object.
    """
    valid = [c for c in comparisons if c["percentile_rank"] is not None][:9]
    n = len(valid)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
        horizontal_spacing=0.05,
        vertical_spacing=0.15,
    )

    for i, c in enumerate(valid):
        row = i // cols + 1
        col = i % cols + 1

        pct = c["percentile_rank"]
        color = "#28a745" if pct >= 60 else "#ffc107" if pct >= 40 else "#dc3545"

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                title={"text": c["display_name"], "font": {"size": 11}},
                number={"suffix": "th pct", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 25], "color": "#ffebee"},
                        {"range": [25, 75], "color": "#f5f5f5"},
                        {"range": [75, 100], "color": "#e8f5e9"},
                    ],
                },
            ),
            row=row, col=col,
        )

    fig.update_layout(height=250 * rows, template="plotly_white")

    if output_path:
        fig.write_html(str(output_path))

    return fig
