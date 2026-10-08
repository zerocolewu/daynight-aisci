from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from sklearn.decomposition import PCA


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


COLORS = {
    "knowledge": "#E7A29C",
    "experience": "#8E4E86",
    "integration": "#176B87",
    "integration_dark": "#0B4257",
    "idea": "#E58B48",
    "gold": "#F2C14E",
    "edge": "#78909C",
    "neutral": "#67747D",
    "paper": "#FFFFFF",
}

MARKERS = {
    "knowledge": "o",
    "experience": "*",
    "integration": "h",
    "idea": "D",
}

HIGHLIGHTS = {
    "mem_104": "PERIODICITY x",
    "mem_127": "MECHANISMS x",
    "mem_138": "REPRESENTATION",
    "mem_153": "DIMENSIONALITY",
}

FLOW_COLORS = {
    "knowledge": {"fill": "#902046", "edge": "#731936", "halo": "#D6A1B3"},
    "experience": {"fill": "#A36C98", "edge": "#84527A", "halo": "#D9C2D4"},
    "integration": {"fill": "#4C67B1", "edge": "#4855A2", "halo": "#BCC5E4"},
    "idea": {"fill": "#F2C66D", "edge": "#D7A649", "halo": "#F9E2AD"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_graph(archive: Path):
    mem_path = archive / "mem.json"
    memories = json.loads(mem_path.read_text(encoding="utf-8"))["memory"]
    by_id = {m["id"]: m for m in memories}

    def get_day(memory):
        return (memory.get("metadata") or {}).get("day", memory.get("day"))

    nodes = {}
    vectors = []
    vector_ids = []
    for memory in memories:
        mem_type = str(memory.get("type", "")).lower()
        day = get_day(memory)
        if mem_type not in MARKERS or not isinstance(day, int) or not 1 <= day <= 7:
            continue
        node = {
            "id": memory["id"],
            "type": mem_type,
            "day": day,
            "abstract": memory.get("abstract", ""),
        }
        nodes[memory["id"]] = node
        embedding = (memory.get("metadata") or {}).get("embedding")
        if embedding:
            embedding_path = archive / embedding
            if embedding_path.exists():
                vector = np.asarray(np.load(embedding_path), dtype=float).reshape(-1)
                if vector.size:
                    vectors.append(vector)
                    vector_ids.append(memory["id"])

    semantic = {}
    if vectors:
        matrix = np.vstack(vectors)
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
        lo = np.percentile(coords, 3, axis=0)
        hi = np.percentile(coords, 97, axis=0)
        scaled = np.clip((coords - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 2 - 1
        semantic = {mem_id: scaled[i] for i, mem_id in enumerate(vector_ids)}

    edges = []
    for memory in memories:
        if memory.get("type") != "integration" or memory.get("id") not in nodes:
            continue
        content = memory.get("content") or {}
        refs = content.get("mentioned_memory_ids", []) if isinstance(content, dict) else []
        target = memory["id"]
        for source in refs:
            if source not in nodes:
                continue
            lag = nodes[target]["day"] - nodes[source]["day"]
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "lag": lag,
                    "source_type": nodes[source]["type"],
                    "higher_order": nodes[source]["type"] == "integration",
                }
            )
    return nodes, edges, semantic


def export_figure(fig, output: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 420
        fig.savefig(output / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def legend_handles():
    handles = []
    for mem_type in ("knowledge", "experience", "integration", "idea"):
        size = 7.5 if mem_type == "integration" else 6
        handles.append(
            Line2D(
                [0],
                [0],
                marker=MARKERS[mem_type],
                linestyle="none",
                color="none",
                markerfacecolor=COLORS[mem_type],
                markeredgecolor="white" if mem_type == "integration" else COLORS[mem_type],
                markeredgewidth=0.7,
                markersize=size,
                label=mem_type,
            )
        )
    return handles


def draw_curve(
    ax,
    p0,
    p1,
    color,
    linewidth,
    alpha,
    arch=0.0,
    zorder=1,
):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    control1 = (x0 + dx * 0.32, y0 + arch)
    control2 = (x0 + dx * 0.68, y1 + arch)
    path = MplPath(
        [p0, control1, control2, p1],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )


def integration_sizes(nodes, edges):
    incoming = Counter(edge["target"] for edge in edges)
    return {
        node_id: 75 + 10 * incoming[node_id]
        for node_id, node in nodes.items()
        if node["type"] == "integration"
    }


def candidate_temporal(nodes, edges, semantic, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.95, bottom=0.12)
    ax.set_facecolor("white")
    ax.axis("off")

    rng = np.random.default_rng(23)
    pos = {}
    for node_id, node in nodes.items():
        sem = semantic.get(node_id, np.zeros(2))
        x = node["day"] + 0.19 * sem[0] + rng.uniform(-0.045, 0.045)
        y = 1.45 * sem[1] + 0.13 * {
            "knowledge": 0.6,
            "experience": 0.1,
            "integration": -0.15,
            "idea": -0.55,
        }[node["type"]]
        pos[node_id] = (x, y)

    for day in range(1, 8):
        ax.axvline(day, color="#DDE3E6", lw=0.55, alpha=0.65, zorder=0)
        ax.text(day, -1.92, f"D{day}", ha="center", va="center", fontsize=7.2, color="#525E65")

    for edge in edges:
        p0, p1 = pos[edge["source"]], pos[edge["target"]]
        lag = max(edge["lag"], 0)
        if edge["higher_order"]:
            color = COLORS["integration_dark"]
            lw, alpha, zorder = 1.15, 0.42, 2
        elif lag > 0:
            color = COLORS["edge"]
            lw, alpha, zorder = 0.48 + 0.05 * lag, 0.18, 1
        else:
            color = "#AEBBC1"
            lw, alpha, zorder = 0.32, 0.09, 1
        sign = -1 if int(edge["source"].split("_")[-1]) % 2 else 1
        arch = sign * (0.05 + 0.035 * lag)
        draw_curve(ax, p0, p1, color, lw, alpha, arch=arch, zorder=zorder)

    sizes = integration_sizes(nodes, edges)
    for mem_type in ("knowledge", "idea", "experience"):
        ids = [node_id for node_id, node in nodes.items() if node["type"] == mem_type]
        xy = np.array([pos[node_id] for node_id in ids])
        if not len(xy):
            continue
        size = {"knowledge": 10, "idea": 22, "experience": 58}[mem_type]
        alpha = {"knowledge": 0.42, "idea": 0.66, "experience": 0.84}[mem_type]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=size,
            marker=MARKERS[mem_type],
            c=COLORS[mem_type],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    integration_ids = [
        node_id for node_id, node in nodes.items() if node["type"] == "integration"
    ]
    int_xy = np.array([pos[node_id] for node_id in integration_ids])
    int_sizes = np.array([sizes[node_id] for node_id in integration_ids])
    ax.scatter(
        int_xy[:, 0],
        int_xy[:, 1],
        s=int_sizes * 1.85,
        marker="o",
        c=COLORS["integration"],
        alpha=0.10,
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        int_xy[:, 0],
        int_xy[:, 1],
        s=int_sizes,
        marker=MARKERS["integration"],
        c=COLORS["integration"],
        edgecolors="white",
        linewidths=0.85,
        zorder=5,
    )

    y_arrow = 1.75
    ax.add_patch(
        FancyArrowPatch(
            (1.05, y_arrow),
            (6.95, y_arrow),
            connectionstyle="arc3,rad=-0.12",
            arrowstyle="Simple,tail_width=2.4,head_width=10,head_length=13",
            color=COLORS["gold"],
            alpha=0.72,
            linewidth=0,
            zorder=0,
        )
    )
    ax.text(3.95, 1.86, "REPLAY", ha="center", color="#9B761B", fontsize=7.2, fontweight="bold")

    offsets = {
        "mem_104": (-0.05, -0.36, "right"),
        "mem_127": (0.05, 0.36, "left"),
        "mem_138": (0.06, -0.38, "left"),
        "mem_153": (-0.06, 0.36, "right"),
    }
    for mem_id, label in HIGHLIGHTS.items():
        if mem_id not in pos:
            continue
        x, y = pos[mem_id]
        dx, dy, align = offsets[mem_id]
        ax.text(
            x + dx,
            y + dy,
            label,
            ha=align,
            va="center",
            fontsize=6.1,
            fontweight="bold",
            color=COLORS["integration_dark"],
            zorder=7,
        )
        ax.plot([x, x + dx * 0.7], [y, y + dy * 0.68], color=COLORS["integration_dark"], lw=0.55)

    ax.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        columnspacing=1.6,
        handletextpad=0.35,
        fontsize=6.3,
    )
    ax.text(0.01, 0.98, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.set_xlim(0.55, 7.45)
    ax.set_ylim(-2.05, 2.05)
    export_figure(fig, output, "candidate_1_temporal_atlas")


def radial_positions(nodes, semantic):
    day_angles = {
        day: angle
        for day, angle in zip(range(1, 8), np.linspace(math.radians(220), math.radians(-40), 7))
    }
    grouped = defaultdict(list)
    for node_id, node in nodes.items():
        grouped[(node["day"], node["type"])].append(node_id)

    pos = {}
    for (day, mem_type), ids in grouped.items():
        ids = sorted(ids, key=lambda item: int(item.split("_")[-1]))
        base = day_angles[day]
        for index, node_id in enumerate(ids):
            sem = semantic.get(node_id, np.zeros(2))
            centered = index - (len(ids) - 1) / 2
            if mem_type == "integration":
                radius = 0.86 + 0.08 * sem[0]
                angle = base + centered * 0.075
            else:
                radius = {
                    "knowledge": 1.83,
                    "experience": 1.48,
                    "idea": 1.60,
                }[mem_type]
                radius += 0.15 * sem[0]
                angle = base + centered * 0.045 + 0.08 * sem[1]
            pos[node_id] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos, day_angles


def candidate_radial(nodes, edges, semantic, output):
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    ax.set_aspect("equal")
    ax.axis("off")

    pos, day_angles = radial_positions(nodes, semantic)
    sizes = integration_sizes(nodes, edges)

    for edge in edges:
        p0, p1 = pos[edge["source"]], pos[edge["target"]]
        if edge["higher_order"]:
            color, lw, alpha, zorder = COLORS["integration_dark"], 1.1, 0.44, 3
        elif edge["lag"] > 0:
            color, lw, alpha, zorder = COLORS["edge"], 0.48, 0.18, 1
        else:
            color, lw, alpha, zorder = "#B7C1C6", 0.30, 0.08, 1
        draw_curve(ax, p0, p1, color, lw, alpha, arch=0.0, zorder=zorder)

    for mem_type in ("knowledge", "idea", "experience"):
        ids = [node_id for node_id, node in nodes.items() if node["type"] == mem_type]
        xy = np.array([pos[node_id] for node_id in ids])
        size = {"knowledge": 10, "idea": 22, "experience": 58}[mem_type]
        alpha = {"knowledge": 0.40, "idea": 0.67, "experience": 0.86}[mem_type]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=size,
            marker=MARKERS[mem_type],
            c=COLORS[mem_type],
            alpha=alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )

    integration_ids = [
        node_id for node_id, node in nodes.items() if node["type"] == "integration"
    ]
    xy = np.array([pos[node_id] for node_id in integration_ids])
    ss = np.array([sizes[node_id] for node_id in integration_ids])
    ax.scatter(xy[:, 0], xy[:, 1], s=ss * 2.0, c=COLORS["integration"], alpha=0.10, edgecolors="none")
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=ss,
        marker=MARKERS["integration"],
        c=COLORS["integration"],
        edgecolors="white",
        linewidths=0.85,
        zorder=6,
    )

    for day, angle in day_angles.items():
        radius = 2.15
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        ax.text(x, y, f"D{day}", ha="center", va="center", fontsize=7.1, color="#535F65")

    theta = np.linspace(math.radians(225), math.radians(-35), 120)
    radius = 2.36
    arc_xy = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    ax.plot(arc_xy[:, 0], arc_xy[:, 1], color=COLORS["gold"], lw=5.0, alpha=0.55, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            arc_xy[-8],
            arc_xy[-1],
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.0,
            color=COLORS["gold"],
            zorder=8,
        )
    )
    ax.text(0, 0.02, "INTEGRATE", ha="center", va="center", fontsize=8.0, fontweight="bold", color=COLORS["integration_dark"])

    label_positions = {
        "mem_104": (-0.88, 0.48, "right"),
        "mem_127": (1.18, 0.78, "left"),
        "mem_138": (1.48, -0.16, "left"),
        "mem_153": (1.18, -1.22, "left"),
    }
    for mem_id, label in HIGHLIGHTS.items():
        if mem_id not in pos:
            continue
        x, y = pos[mem_id]
        tx, ty, align = label_positions[mem_id]
        ax.text(
            tx,
            ty,
            label,
            ha=align,
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=COLORS["integration_dark"],
            zorder=8,
        )
        line_end_x = tx - 0.06 if align == "left" else tx + 0.06
        ax.plot([x, line_end_x], [y, ty], color=COLORS["integration_dark"], lw=0.55)

    ax.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        columnspacing=1.4,
        handletextpad=0.3,
        fontsize=6.2,
    )
    ax.text(0.01, 0.99, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.set_xlim(-2.62, 2.62)
    ax.set_ylim(-2.48, 2.58)
    export_figure(fig, output, "candidate_2_radial_constellation")


def candidate_alluvial(nodes, edges, semantic, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.13)
    ax.axis("off")

    type_y = {
        "knowledge": 1.45,
        "experience": 0.63,
        "integration": -0.15,
        "idea": -1.08,
    }
    group_counts = Counter()
    for edge in edges:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        group_counts[(source["day"], source["type"], target["day"])] += 1

    for day in range(1, 8):
        ax.axvline(day, color="#E1E6E8", lw=0.55, alpha=0.75, zorder=0)
        ax.text(day, -1.66, f"D{day}", ha="center", fontsize=7.2, color="#000000")

    for mem_type, y in type_y.items():
        palette = FLOW_COLORS[mem_type]
        if mem_type != "integration":
            ax.plot([0.85, 7.15], [y, y], color=palette["fill"], lw=8.5, alpha=0.055, zorder=0)
        ax.text(
            0.72,
            y,
            mem_type.upper(),
            ha="right",
            va="center",
            fontsize=6.1,
            fontweight="bold" if mem_type == "integration" else "normal",
            color=palette["edge"],
        )

    for (source_day, source_type, target_day), count in sorted(group_counts.items()):
        p0 = (source_day, type_y[source_type])
        p1 = (target_day, type_y["integration"])
        lag = max(target_day - source_day, 0)
        direction = np.sign(type_y[source_type] - type_y["integration"])
        draw_curve(
            ax,
            p0,
            p1,
            FLOW_COLORS[source_type]["fill"],
            0.55 + 0.45 * count,
            0.13 if lag == 0 else 0.25,
            arch=0.08 * direction,
            zorder=1,
        )

    daily_integration_count = Counter(
        node["day"] for node in nodes.values() if node["type"] == "integration"
    )
    palette = FLOW_COLORS["integration"]
    for day in range(1, 8):
        count = daily_integration_count[day]
        size = 26 + 15 * count
        ax.scatter(
            [day],
            [type_y["integration"]],
            s=size * 1.8,
            marker="h",
            c=palette["halo"],
            alpha=0.42,
            edgecolors="none",
            zorder=4,
        )
        ax.scatter(
            [day],
            [type_y["integration"]],
            s=size,
            marker="h",
            c=palette["fill"],
            edgecolors=palette["edge"],
            linewidths=0.8,
            zorder=5,
        )

    for mem_type in ("knowledge", "experience", "idea"):
        palette = FLOW_COLORS[mem_type]
        for day in range(1, 8):
            count = sum(
                1
                for node in nodes.values()
                if node["type"] == mem_type and node["day"] == day
            )
            if count == 0:
                continue
            size = 14 + 8 * count
            marker = {"knowledge": "o", "experience": "*", "idea": "D"}[mem_type]
            ax.scatter(
                [day],
                [type_y[mem_type]],
                s=size * 1.55,
                marker=marker,
                c=palette["halo"],
                alpha=0.34,
                edgecolors="none",
                zorder=3,
            )
            ax.scatter(
                [day],
                [type_y[mem_type]],
                s=size,
                marker=marker,
                c=palette["fill"],
                edgecolors=palette["edge"],
                linewidths=0.6,
                alpha=0.98,
                zorder=4,
            )

    higher = Counter()
    for edge in edges:
        if not edge["higher_order"]:
            continue
        source_day = nodes[edge["source"]]["day"]
        target_day = nodes[edge["target"]]["day"]
        if source_day < target_day:
            higher[(source_day, target_day)] += 1
    for (source_day, target_day), count in sorted(higher.items()):
        y = type_y["integration"]
        draw_curve(
            ax,
            (source_day, y - 0.03),
            (target_day, y + 0.03),
            FLOW_COLORS["integration"]["edge"],
            0.72 + 0.24 * count,
            0.38,
            arch=-0.12 - 0.025 * (target_day - source_day),
            zorder=3,
        )

    ax.text(0.01, 0.98, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.set_xlim(0.48, 7.58)
    ax.set_ylim(-1.78, 1.82)
    export_figure(fig, output, "candidate_3_braided_flow_clean")


def make_comparison(output):
    stems = [
        ("1  temporal provenance atlas", "candidate_1_temporal_atlas.png"),
        ("2  radial integration constellation", "candidate_2_radial_constellation.png"),
        ("3  braided consolidation flow", "candidate_3_braided_flow_clean.png"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 13.2))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.02, hspace=0.12)
    for ax, (title, filename) in zip(axes, stems):
        image = plt.imread(output / filename)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=3)
    fig.savefig(output / "candidate_comparison.png", dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "candidate_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nodes, edges, semantic = load_graph(args.archive)
    candidate_temporal(nodes, edges, semantic, args.output)
    candidate_radial(nodes, edges, semantic, args.output)
    candidate_alluvial(nodes, edges, semantic, args.output)
    make_comparison(args.output)
    print(f"nodes={len(nodes)} edges={len(edges)} output={args.output}")


if __name__ == "__main__":
    main()
