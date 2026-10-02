"""Manuscript plotting kernels; caller supplies bundle data."""
from __future__ import annotations

import argparse

import json

import shutil

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib.lines import Line2D

K_VALUES = [10, 50, 100, 200, 500, 1000]

PANEL_ORDER = ["TF+500", "TF+1000"]

NETWORK_ORDER = ["Specific", "LOF/GOF", "Non-specific", "STRING"]

NETWORK_TITLE = {
    "Specific": "Specific ChIP-seq",
    "LOF/GOF": "LOF/GOF perturbation",
    "Non-specific": "Non-specific ChIP-seq",
    "STRING": "STRING associations",
}

PANEL_COLORS = {"TF+500": "#0072B2", "TF+1000": "#D55E00"}

CELL_COLORS = {
    "hESC": "#0072B2",
    "hHEP": "#D55E00",
    "mDC": "#009E73",
    "mESC": "#CC79A7",
    "mHSC E": "#E69F00",
    "mHSC GM": "#56B4E9",
    "mHSC L": "#4D4D4D",
}

PANEL_STYLES = {"TF+500": "-", "TF+1000": (0, (4, 2))}

UNLABELED_COLOR = "#0072B2"

POSITIVE_COLOR = "#D55E00"

def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=3)

def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Write matching raster and vector versions from one rendered figure."""
    fig.savefig(out_path, dpi=360, bbox_inches="tight", facecolor="white")

def panel_from_cell_label(cell_label: str) -> str:
    return "TF+1000" if cell_label.endswith("1000") else "TF+500"

def display_cell_label(cell_label: str) -> str:
    name, panel = cell_label.rsplit(" ", 1)
    return f"{name} ({panel})"

def plot_category_curves(precision: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for panel_index, (ax, network) in enumerate(zip(axes_flat, NETWORK_ORDER)):
        subset = precision[precision["network"] == network]
        summary = (
            subset.groupby(["tf_panel", "k"], observed=False)["precision_percent"]
            .agg(["mean", "std"])
            .reset_index()
        )
        for panel in PANEL_ORDER:
            panel_df = summary[summary["tf_panel"] == panel].sort_values("k")
            if panel_df.empty:
                continue
            x = panel_df["k"].to_numpy(dtype=float)
            mean = panel_df["mean"].to_numpy(dtype=float)
            std = panel_df["std"].fillna(0.0).to_numpy(dtype=float)
            color = PANEL_COLORS[panel]
            ax.plot(x, mean, marker="o", markersize=3.0, linewidth=1.45, color=color, label=panel)
            ax.fill_between(
                x,
                np.clip(mean - std, 0, 100),
                np.clip(mean + std, 0, 100),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(NETWORK_TITLE[network], loc="left", fontweight="normal", pad=4)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xticks([10, 100, 1000], labels=["10", "100", "1,000"])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Retained pairs, K" if panel_index >= 2 else "")
        ax.set_ylabel("Precision@K (%)" if panel_index % 2 == 0 else "")
        finish_axis(ax)

    handles = [
        Line2D([0], [0], color=PANEL_COLORS[p], marker="o", markersize=3, linewidth=1.45, label=p.replace("TF+", "TFs + "))
        for p in PANEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2, frameon=False)
    fig.text(0.995, 0.018, "Lines: mean; bands: ±1 SD across cell types", ha="right", va="bottom", fontsize=6.8, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.15, wspace=0.13, hspace=0.24)
    save_figure(fig, out_path)
    plt.close(fig)

def plot_network_curves(precision: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.20), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for panel_index, (ax, network) in enumerate(zip(axes_flat, NETWORK_ORDER)):
        subset = precision[precision["network"] == network]
        for cell_display, group in subset.groupby("cell_display", sort=False):
            group = group.sort_values("k")
            cell = cell_display.rsplit(" (", 1)[0]
            panel = "TF+1000" if cell_display.endswith("(1000)") else "TF+500"
            ax.plot(
                group["k"],
                group["precision_percent"],
                color=CELL_COLORS[cell],
                linestyle=PANEL_STYLES[panel],
                marker="o",
                markersize=2.1,
                linewidth=0.9,
                alpha=0.88,
            )
        ax.set_title(NETWORK_TITLE[network], loc="left", fontweight="normal", pad=4)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xticks([10, 100, 1000], labels=["10", "100", "1,000"])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Retained pairs, K" if panel_index >= 2 else "")
        ax.set_ylabel("Precision@K (%)" if panel_index % 2 == 0 else "")
        finish_axis(ax)

    cell_handles = [
        Line2D([0], [0], color=color, linewidth=1.6, label=cell)
        for cell, color in CELL_COLORS.items()
    ]
    panel_handles = [
        Line2D([0], [0], color="#333333", linestyle=PANEL_STYLES[p], linewidth=1.25, label=p.replace("TF+", "TFs + "))
        for p in PANEL_ORDER
    ]
    first_legend = fig.legend(
        handles=cell_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.053),
        ncol=7,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.add_artist(first_legend)
    fig.legend(
        handles=panel_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.24, wspace=0.13, hspace=0.25)
    save_figure(fig, out_path)
    plt.close(fig)

def add_distribution_annotation(ax: plt.Axes, labels: pd.Series) -> None:
    pos_n = int(labels.sum())
    neg_n = int(len(labels) - pos_n)
    text = f"Sampled unlabeled n = {neg_n:,}\nReference positive n = {pos_n:,}"
    ax.text(
        0.20,
        0.94,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.8,
        linespacing=1.25,
        color="#444444",
    )

def plot_distribution(ax: plt.Axes, data: pd.DataFrame, title: str | None = None, legend: bool = True) -> None:
    neg = data[data["label"] == 0]["score"]
    pos = data[data["label"] == 1]["score"]
    sns.kdeplot(
        x=neg,
        ax=ax,
        color=UNLABELED_COLOR,
        fill=True,
        alpha=0.10,
        linewidth=1.35,
        label="Sampled unlabeled",
        bw_adjust=0.65,
        clip=(0, 1),
        cut=0,
    )
    sns.kdeplot(
        x=pos,
        ax=ax,
        color=POSITIVE_COLOR,
        fill=True,
        alpha=0.10,
        linewidth=1.35,
        label="Reference positive",
        bw_adjust=0.65,
        clip=(0, 1),
        cut=0,
    )
    neg_mean = float(neg.mean())
    pos_mean = float(pos.mean())
    ax.axvline(
        neg_mean,
        color=UNLABELED_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        alpha=0.9,
    )
    ax.axvline(
        pos_mean,
        color=POSITIVE_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        alpha=0.9,
    )
    ax.text(neg_mean, 0.72, f"mean {neg_mean:.3f}", color=UNLABELED_COLOR, ha="left", va="top", rotation=90, transform=ax.get_xaxis_transform(), fontsize=6.4)
    ax.text(pos_mean, 0.72, f"mean {pos_mean:.3f}", color=POSITIVE_COLOR, ha="right", va="top", rotation=90, transform=ax.get_xaxis_transform(), fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("BEACON score")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title, loc="left", fontweight="normal", pad=4)
    add_distribution_annotation(ax, data["label"])
    finish_axis(ax)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

def plot_all_distribution(predictions: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.15, 2.40))
    plot_distribution(ax, predictions, legend=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.29)
    save_figure(fig, out_path)
    plt.close(fig)

def plot_category_distribution(predictions: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.00), sharex=True)
    for panel_index, (ax, network) in enumerate(zip(axes.ravel(), NETWORK_ORDER)):
        subset = predictions[predictions["network"] == network]
        plot_distribution(ax, subset, title=NETWORK_TITLE[network], legend=False)
        if panel_index < 2:
            ax.set_xlabel("")
        if panel_index % 2:
            ax.set_ylabel("")
    handles = [
        Line2D([0], [0], color=UNLABELED_COLOR, linewidth=1.5, label="Sampled unlabeled"),
        Line2D([0], [0], color=POSITIVE_COLOR, linewidth=1.5, label="Reference positive"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.96, bottom=0.17, wspace=0.15, hspace=0.29)
    save_figure(fig, out_path)
    plt.close(fig)

def write_distribution_summary(predictions: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for network, subset in [("All", predictions), *list(predictions.groupby("network", sort=False))]:
        labels = subset["label"].astype(int)
        neg = subset[labels == 0]["score"]
        pos = subset[labels == 1]["score"]
        rows.append(
            {
                "network": network,
                "sampled_unlabeled_n": int((labels == 0).sum()),
                "reference_positive_n": int((labels == 1).sum()),
                "sampled_unlabeled_mean": float(neg.mean()),
                "reference_positive_mean": float(pos.mean()),
                "fold_separation": float(pos.mean() / neg.mean()) if float(neg.mean()) else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
