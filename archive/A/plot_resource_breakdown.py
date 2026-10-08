from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

# User-provided run_experiment token data (millions)
RUN_EXP_TOKENS_M = {
    1: 3.48, 2: 5.53, 3: 12.27, 4: 18.62,
    5: 16.4, 6: 21.0, 7: 18.49,
}

# Stacking order (bottom → top)
AGENT_ORDER = [
    "orchestrator", "learn_search", "learn_read", "work",
    "internalize", "associate_reason", "associate_random",
    "run_experiment",
]

LABEL = {
    "orchestrator": "Orchestrator",
    "learn_search": "Learn-Search",
    "learn_read": "Learn-Read",
    "work": "Work",
    "internalize": "Internalize",
    "associate_reason": "Assoc.-Reason",
    "associate_random": "Assoc.-Random",
    "run_experiment": "Run-Exp",
}

COLOR = {
    "orchestrator": "#1E90FF",
    "learn_search": "#FF6347",
    "learn_read": "#32CD32",
    "work": "#FFD700",
    "internalize": "#9370DB",
    "associate_reason": "#20B2AA",
    "associate_random": "#FF69B4",
    "run_experiment": "#A9A9A9",
}

LLM_TYPES = [t for t in AGENT_ORDER if t != "run_experiment"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_day(raw: dict) -> dict[int, dict[str, dict[str, float]]]:
    """Aggregate token and time data by day and agent type."""
    out: dict[int, dict[str, dict[str, float]]] = {}
    for dk, dv in raw["by_day"].items():
        dn = dv["day_number"]
        agg: dict[str, dict[str, float]] = defaultdict(
            lambda: {"tokens": 0.0, "time": 0.0}
        )

        for r in dv.get("orchestrator_runs", []):
            agg["orchestrator"]["tokens"] += r["total_tokens"]
            agg["orchestrator"]["time"] += r["llm_time"]

        for r in dv.get("behaviours", []):
            at = r["agent_type"]
            if at != "run_experiment":
                agg[at]["tokens"] += r["total_tokens"]
            agg[at]["time"] += r["llm_time"]

        # Override run_exp tokens with user-provided values
        agg["run_experiment"]["tokens"] = RUN_EXP_TOKENS_M.get(dn, 0) * 1e6
        out[dn] = dict(agg)

    return out


def _stack(ax, days, agg, types, field, divisor, width=0.55):
    """Draw stacked bars and return x positions."""
    x = np.arange(len(days))
    bot = np.zeros(len(days))
    for t in types:
        vals = np.array([
            agg[d].get(t, {"tokens": 0, "time": 0})[field] / divisor
            for d in days
        ])
        kw = dict(color=COLOR[t], edgecolor="white", linewidth=0.5)
        if t == "run_experiment":
            kw["hatch"] = "///"
        ax.bar(x, vals, width, bottom=bot, **kw)
        bot += vals
    return x


def _dress_main(ax, days, ylabel, panel_label):
    """Format a main panel axes."""
    ax.set_xlabel("Day", fontsize=12, weight="bold")
    ax.set_ylabel(ylabel, fontsize=12, weight="bold")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(panel_label, fontsize=13, weight="bold", loc="left")
    ax.grid(axis="y", alpha=0.2, linestyle="--")


def plot():
    raw = load_json(BASE_DIR / "log_analysis.json")
    agg = aggregate_by_day(raw)
    days = sorted(agg.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    # ---- (a) Token Consumption (full) ----
    _stack(ax_a, days, agg, AGENT_ORDER, "tokens", 1e6)
    _dress_main(ax_a, days, "Tokens (×10⁶)", "(a)")

    # ---- (b) Wall-Clock Time (full) ----
    _stack(ax_b, days, agg, AGENT_ORDER, "time", 3600)
    _dress_main(ax_b, days, "Wall-Clock Time (h)", "(b)")

    # ---- (c) Token Consumption (LLM agents only) ----
    _stack(ax_c, days, agg, LLM_TYPES, "tokens", 1e3)
    _dress_main(ax_c, days, "Tokens (×10³)", "(c)")

    # ---- (d) Wall-Clock Time (LLM agents only) ----
    _stack(ax_d, days, agg, LLM_TYPES, "time", 60)
    _dress_main(ax_d, days, "Time (min)", "(d)")

    # ---- Shared Legend ----
    handles = [
        mpatches.Patch(facecolor=COLOR[t], label=LABEL[t],
                       hatch=("///" if t == "run_experiment" else ""))
        for t in AGENT_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8,
               fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=2.5)

    out_pdf = BASE_DIR / "resource_breakdown.pdf"
    out_png = BASE_DIR / "resource_breakdown.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_pdf.name}, {out_png.name}")


if __name__ == "__main__":
    plot()
