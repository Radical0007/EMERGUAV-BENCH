"""Plot manuscript model tradeoffs from the displayed four-decimal results.

Run from any directory with the repository Python environment. The source CSV
and README comparison table must agree; this script never updates results.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("S_proto", "Semantic_Avg", "Acc_safe", "Rate_unsafe", "PACS", "MAD", "Strict_Avg")
# Labels and offsets are visual annotations only: coordinates always come from CSV.
LABELS = {
    "Qwen3-VL-8B": ("Qwen3-VL-8B", (-6, 26), "right"),
    "Qwen3-VL-4B": ("Qwen3-VL-4B", (-18, 22), "right"),
    "Qwen3-VL-2B": ("Qwen3-VL-2B", (14, -9), "left"),
    "Qwen3.5-9B": ("Qwen3.5-9B", (-3, 20), "right"),
    "Qwen3.5-4B": ("Qwen3.5-4B", (12, -23), "left"),
    "Gemma-4-E4B-it": ("Gemma-4-E4B", (-8, 20), "center"),
    "Gemma-4-E2B-it": ("Gemma-4-E2B", (5, 26), "center"),
    "InternVL3.5-8B": ("InternVL3.5-8B", (-4, 24), "center"),
    "GLM-4.6V-Flash": ("GLM-4.6V-Flash", (10, -18), "left"),
    "Qwen2.5-VL-7B": ("Qwen2.5-VL-7B", (-9, -26), "center"),
    "MiniCPM-V-4.5": ("MiniCPM-V-4.5", (12, -17), "left"),
    "LLaVA-NeXT-8B": ("LLaVA-NeXT-8B", (17, 0), "left"),
    "Phi-4-MM": ("Phi-4-MM", (-5, -24), "center"),
}


def read_results(csv_path: Path, readme: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["Model"] for row in rows]
    if len(names) != 13 or len(set(names)) != 13 or set(names) != set(LABELS):
        raise ValueError("Expected the thirteen manuscript models; update labels for a changed model set.")
    table = readme.read_text(encoding="utf-8").split("### 3.1. Model Comparison", 1)[1].split("### 3.2.", 1)[0]
    for row in rows:
        candidates = [line for line in table.splitlines() if line.startswith("| " + row["Model"] + " |")]
        if len(candidates) != 1:
            raise ValueError(f"Missing or duplicate README model: {row['Model']}")
        displayed = re.findall(r"0\.\d{4}", candidates[0])
        expected = [row[metric] for metric in METRICS]
        if displayed != expected:
            raise ValueError(f"CSV/README mismatch for {row['Model']}: {displayed} != {expected}")
        for metric in METRICS:
            row[metric] = float(row[metric])
            if not 0 <= row[metric] <= 1:
                raise ValueError(f"Invalid {metric} for {row['Model']}")
    return rows


def plot(rows: list[dict], output: Path) -> None:
    # Fail explicitly if the requested publication font is not installed.
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelcolor": "#263447", "text.color": "#263447",
        "xtick.color": "#58667B", "ytick.color": "#58667B",
    })
    fig = plt.figure(figsize=(6.6, 4.55), facecolor="white")
    ax = fig.add_axes((0.115, 0.25, 0.845, 0.69))
    ax.set_facecolor("#F7F9FC")
    ax.set_axisbelow(True)
    ax.grid(color="#E2E8F0", linewidth=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#8B99AE")
        ax.spines[side].set_linewidth(0.8)
    cmap = LinearSegmentedColormap.from_list("unsafe_rate", ["#4E909C", "#C7B580", "#BE6976"])
    norm = Normalize(vmin=0, vmax=0.23)
    area_scale = 1250.0  # Matplotlib s is marker area, in points squared.
    points = ax.scatter(
        [row["Semantic_Avg"] for row in rows],
        [row["Acc_safe"] for row in rows],
        s=[area_scale * row["S_proto"] for row in rows],
        c=[row["Rate_unsafe"] for row in rows], cmap=cmap, norm=norm,
        edgecolors="white", linewidths=1.25, alpha=0.96, zorder=3,
    )
    for row in rows:
        label, offset, align = LABELS[row["Model"]]
        ax.annotate(
            label, (row["Semantic_Avg"], row["Acc_safe"]),
            xytext=offset, textcoords="offset points", ha=align, va="center",
            fontsize=8.8, fontweight="bold" if row["Model"] in {
                "Qwen3-VL-8B", "Qwen3-VL-4B", "Qwen3.5-9B", "Qwen2.5-VL-7B"
            } else "normal",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "none", "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "color": "#8491A3", "lw": 0.65, "shrinkA": 2, "shrinkB": 9},
            zorder=4,
        )
    ax.set(xlim=(0.350, 0.512), ylim=(0.475, 0.885),
           xlabel="Semantic average", ylabel="Safe-action accuracy")
    ax.xaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax.tick_params(length=3, width=0.7, pad=5)
    legend_values = (0.25, 0.35, 0.42)
    # Both legends follow the same vertical reading order: symbol, value, meaning.
    lax = fig.add_axes((0.09, 0.035, 0.43, 0.115))
    lax.set(xlim=(0, 1), ylim=(0, 1))
    lax.set_axis_off()
    for x, value in zip((0.18, 0.5, 0.82), legend_values):
        lax.scatter(x, 0.59, s=area_scale * value, facecolor="#BBC6D5",
                    edgecolor="white", linewidth=1, clip_on=False)
        lax.text(x, 0.17, f"{value:.2f}", ha="center", va="center", fontsize=8.5)
    fig.text(0.305, 0.019, r"Bubble area: $S_{\mathrm{proto}}$",
             ha="center", va="center", fontsize=8.8)
    cax = fig.add_axes((0.61, 0.091, 0.32, 0.023))
    colorbar = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=(0, 0.1, 0.2))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5, length=2, pad=6.5)
    fig.text(0.77, 0.019, "Unsafe-action rate (lower is better)",
             ha="center", va="center", fontsize=8.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".jpg"), dpi=600, facecolor="white", pil_kwargs={"quality": 95})
    fig.savefig(output.with_suffix(".pdf"), facecolor="white",
                metadata={"Title": "Emergency UAV model safety tradeoffs", "CreationDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "img/tables/main_leaderboard.csv")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--output", type=Path, default=ROOT / "img/generated/model_safety_tradeoffs")
    args = parser.parse_args()
    rows = read_results(args.results, args.readme)
    plot(rows, args.output)
    print(f"Validated {len(rows)} models against README; wrote {args.output}.jpg and .pdf")


if __name__ == "__main__":
    main()
