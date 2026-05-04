"""
Visualization helpers.

Three charts are produced for every run:

    1. Top 20 deals by ROI (horizontal bar chart)
    2. Max bid vs list price (scatter with break-even diagonal)
    3. Expected profit distribution (histogram)
    4. Risk fan chart (when Monte Carlo simulation is enabled)

We use matplotlib only -- no seaborn -- to keep the dependency
footprint minimal. The Agg backend is forced so the script runs in
headless environments (CI, server cron jobs).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_top_roi(df: pd.DataFrame, out_dir: str | os.PathLike,
                 top_n: int = 20) -> Path:
    out_dir = _ensure_dir(out_dir)
    feasible = df[df["feasible"]].copy()
    top = feasible.nlargest(top_n, "expected_roi")[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["property_id"] + " (" + top["city"] + ")",
            (top["expected_roi"] * 100), color="#2c7fb8")
    ax.set_xlabel("Expected ROI (%)")
    ax.set_title(f"Top {top_n} Deals by ROI")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / "top_roi.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_bid_vs_list(df: pd.DataFrame, out_dir: str | os.PathLike) -> Path:
    out_dir = _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(9, 7))
    feasible = df[df["feasible"]]
    infeasible = df[~df["feasible"]]

    ax.scatter(infeasible["list_price"], infeasible["max_bid"],
               c="lightgray", alpha=0.6, label="Infeasible", s=30)
    ax.scatter(feasible["list_price"], feasible["max_bid"],
               c="#d7191c", alpha=0.8, label="Feasible", s=40)

    # 45-degree break-even line: bidding above this means overpaying
    # relative to list. Below means there's room to negotiate.
    lim = max(df["list_price"].max(), df["max_bid"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Bid = List")

    ax.set_xlabel("List Price ($)")
    ax.set_ylabel("Max Bid ($)")
    ax.set_title("Max Bid vs List Price")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / "bid_vs_list.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_profit_distribution(df: pd.DataFrame,
                             out_dir: str | os.PathLike) -> Path:
    out_dir = _ensure_dir(out_dir)
    feasible = df[df["feasible"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(feasible["expected_profit"] / 1000, bins=25,
            color="#31a354", edgecolor="white")
    ax.set_xlabel("Expected Profit ($000s)")
    ax.set_ylabel("Number of Deals")
    ax.set_title("Expected Profit Distribution (Feasible Deals)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / "profit_distribution.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_risk_fan(df: pd.DataFrame, top_n: int = 10,
                  out_dir: str | os.PathLike = "outputs") -> Path:
    out_dir = _ensure_dir(out_dir)
    if "p_feasible" not in df.columns or "profit_p5" not in df.columns:
        raise ValueError("Risk fan chart requires Monte Carlo simulation columns.")

    if "risk_adjusted_score" in df.columns:
        ranking_col = "risk_adjusted_score"
    else:
        ranking_col = "profit_p50"

    candidate = df[df["feasible"]].copy()
    top = candidate.nlargest(top_n, ranking_col)[::-1]

    labels = top["property_id"] + " (" + top["city"] + ")"
    y_positions = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hlines(y=y_positions, xmin=top["profit_p5"], xmax=top["profit_p95"],
              color="#6baed6", alpha=0.8, linewidth=6)
    ax.scatter(top["profit_p50"], y_positions, color="#2171b5", s=80, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Profit ($)")
    ax.set_title(f"Risk Fan Chart: Profit Distribution for Top {len(top)} Deals")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = out_dir / "risk_fan.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_all(df: pd.DataFrame, out_dir: str | os.PathLike) -> list[Path]:
    return [
        plot_top_roi(df, out_dir),
        plot_bid_vs_list(df, out_dir),
        plot_profit_distribution(df, out_dir),
    ]
