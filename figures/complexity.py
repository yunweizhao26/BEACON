"""Manuscript plotting kernels; caller supplies bundle data."""
from __future__ import annotations

import argparse

import shutil

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

METHOD_ORDER = ["BEACON", "GNNLINK", "Inferelator 3.0"]

METHOD_LABELS = {"BEACON": "BEACON", "GNNLINK": "GNNLink", "Inferelator 3.0": "Inferelator 3.0"}

COLORS = {"BEACON": "#222222", "GNNLINK": "#0072B2", "Inferelator 3.0": "#D55E00"}

def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source table: {path}. Run plot_beeline_topology_associations.py first to rebuild it."
        )
    df = pd.read_csv(path)
    required = {
        "dataset_id",
        "method",
        "auroc_pct",
        "auprc_pct",
        "mean_directed_degree",
        "max_finite_directed_shortest_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        raise ValueError(f"No rows for {METHOD_ORDER} in {path}")
    expected_pairs = {(method, dataset_id) for method in METHOD_ORDER for dataset_id in sorted(df["dataset_id"].unique())}
    observed_pairs = {(str(row.method), int(row.dataset_id)) for row in df.itertuples(index=False)}
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(f"Missing method/dataset rows in {path}: {missing_pairs[:12]}")

    for column in ["auroc_pct", "auprc_pct", "mean_directed_degree", "max_finite_directed_shortest_path"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    bins = df[["dataset_id", "mean_directed_degree"]].drop_duplicates().copy()
    bins["density_bin"] = pd.qcut(bins["mean_directed_degree"], q=7, labels=False, duplicates="drop")
    df = df.merge(bins[["dataset_id", "density_bin"]], on="dataset_id", how="left", validate="many_to_one")

    long_frames = []
    metric_specs = [("AUPRC", "auprc_pct"), ("AUROC", "auroc_pct")]
    for metric, score_col in metric_specs:
        part = df.copy()
        part["Metric"] = metric
        part["score"] = part[score_col] / 100.0
        part["Method"] = part["method"].map(METHOD_LABELS)
        part["method_key"] = part["method"]
        part["Dataset_ID"] = part["dataset_id"].astype(int)
        part["avg_directed_degree"] = part["mean_directed_degree"]
        part["max_directed_path_length"] = part["max_finite_directed_shortest_path"]
        long_frames.append(part)

    table = pd.concat(long_frames, ignore_index=True)
    method_rank = {method: rank for rank, method in enumerate(METHOD_ORDER)}
    metric_rank = {"AUPRC": 0, "AUROC": 1}
    table["_method_rank"] = table["method_key"].map(method_rank)
    table["_metric_rank"] = table["Metric"].map(metric_rank)
    table = table.sort_values(["_metric_rank", "_method_rank", "Dataset_ID"]).drop(
        columns=["_metric_rank", "_method_rank"]
    )
    return table.reset_index(drop=True)

def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    density = (
        table.groupby(["Metric", "method_key", "Method", "density_bin"], as_index=False)
        .agg(
            x=("avg_directed_degree", "mean"),
            x_min=("avg_directed_degree", "min"),
            x_max=("avg_directed_degree", "max"),
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            n=("Dataset_ID", "nunique"),
        )
        .sort_values(["Metric", "method_key", "x"])
    )

    depth = (
        table.groupby(["Metric", "method_key", "Method", "max_directed_path_length"], as_index=False)
        .agg(
            x=("max_directed_path_length", "mean"),
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            n=("Dataset_ID", "nunique"),
        )
        .sort_values(["Metric", "method_key", "x"])
    )
    return density, depth

def metric_ylim(metric: str) -> tuple[float, float]:
    if metric == "AUROC":
        return 0.75, 1.01
    return 0.0, 1.01

def plot_panel(
    ax,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    x_col: str,
    title: str,
    xlabel: str,
    show_ylabel: bool,
    mean_x: float,
) -> None:
    metric_table = table[table["Metric"] == metric]
    metric_summary = summary[summary["Metric"] == metric]
    for method_key in METHOD_ORDER:
        color = COLORS[method_key]
        label = METHOD_LABELS[method_key]
        points = metric_table[metric_table["method_key"] == method_key].sort_values(x_col)
        ax.scatter(
            points[x_col],
            points["score"],
            s=10,
            color=color,
            alpha=0.17,
            edgecolors="none",
        )

        line = metric_summary[metric_summary["method_key"] == method_key].sort_values("x")
        x = line["x"].to_numpy(dtype=float)
        y = line["mean_score"].to_numpy(dtype=float)
        sd = line["sd_score"].fillna(0).to_numpy(dtype=float)
        linewidth = 1.8 if method_key == "BEACON" else 1.25
        ax.plot(x, y, color=color, marker="o", linewidth=linewidth, markersize=2.8, label=label)
        ax.fill_between(
            x,
            np.clip(y - sd, 0, 1),
            np.clip(y + sd, 0, 1),
            color=color,
            alpha=0.09,
            linewidth=0,
        )
    ax.axvline(
        mean_x,
        color="#777777",
        linewidth=0.85,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        label="BEELINE mean",
    )
    if title:
        ax.set_title(title, fontsize=9, fontweight="normal", loc="center", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(metric if show_ylabel else "", fontsize=8.5)
    ax.set_ylim(*metric_ylim(metric))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5, width=0.6, length=3)

def make_plot(table: pd.DataFrame, density: pd.DataFrame, depth: pd.DataFrame, output_path: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.7,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.65), sharex="col", constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.95, bottom=0.16, wspace=0.14, hspace=0.15)

    topology = table[["Dataset_ID", "avg_directed_degree", "max_directed_path_length"]].drop_duplicates()
    mean_degree = topology["avg_directed_degree"].mean()
    mean_depth = topology["max_directed_path_length"].mean()

    plot_panel(
        axes[0, 0],
        table,
        density,
        "AUPRC",
        "avg_directed_degree",
        "Mean directed degree",
        "",
        True,
        mean_degree,
    )
    plot_panel(
        axes[0, 1],
        table,
        depth,
        "AUPRC",
        "max_directed_path_length",
        "Maximum directed path length",
        "",
        False,
        mean_depth,
    )
    plot_panel(
        axes[1, 0],
        table,
        density,
        "AUROC",
        "avg_directed_degree",
        "",
        "Mean directed degree (edges per gene)",
        True,
        mean_degree,
    )
    plot_panel(
        axes[1, 1],
        table,
        depth,
        "AUROC",
        "max_directed_path_length",
        "",
        "Maximum directed path length",
        False,
        mean_depth,
    )

    handles, labels = axes[0, 1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handlelength=2.1,
        columnspacing=1.5,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
