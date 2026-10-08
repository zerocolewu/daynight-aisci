from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE


# Relative-path only: everything is resolved from this script directory.
BASE_DIR = Path(__file__).resolve().parent
MEM_PATH = BASE_DIR / "mem.json"

COLORS = {
    "knowledge": "red",
    "experience": "purple",
    "integration": "blue",
    "idea": "orange",
}

MARKERS = {
    "knowledge": "o",
    "experience": "*",
    "integration": "^",
    "idea": "D",
}

POINT_SIZE = {
    "knowledge": 72,
    "experience": 120,
    "integration": 72,
    "idea": 72,
}


def load_embeddings(mem_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with mem_path.open("r", encoding="utf-8") as f:
        memories = json.load(f)["memory"]

    target_types = {"knowledge", "experience", "integration", "idea"}
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    mem_ids: list[str] = []

    for mem in memories:
        mem_id = mem.get("id", "")
        if not (isinstance(mem_id, str) and mem_id.startswith("mem_")):
            continue

        try:
            mem_num = int(mem_id.split("_")[1])
        except Exception:
            continue
        if mem_num < 3:
            continue

        mem_type = str(mem.get("type", "")).strip().lower()
        if mem_type not in target_types:
            continue

        emb_rel = (mem.get("metadata") or {}).get("embedding")
        if not emb_rel:
            continue

        emb_path = BASE_DIR / emb_rel
        if not emb_path.exists():
            continue

        vec = np.asarray(np.load(emb_path)).reshape(-1)
        if vec.size == 0:
            continue

        vectors.append(vec)
        labels.append(mem_type)
        mem_ids.append(mem_id)

    if not vectors:
        raise RuntimeError("No usable embeddings found.")

    return np.vstack(vectors), labels, mem_ids


def compute_tsne(vectors: np.ndarray) -> np.ndarray:
    n = len(vectors)
    if n < 3:
        return np.zeros((n, 2), dtype=float)

    perplexity = min(30, max(2, n // 3))
    perplexity = min(perplexity, n - 1)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    return tsne.fit_transform(vectors)


def add_cov_ellipse(
    ax: plt.Axes, points: np.ndarray, color: str, n_std: float = 2.0, alpha: float = 0.18
) -> None:
    if len(points) < 3:
        return

    cov = np.cov(points, rowvar=False)
    if np.linalg.det(cov) <= 1e-12:
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    center = points.mean(axis=0)

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        lw=1.4,
        alpha=alpha,
    )
    ax.add_patch(ellipse)


def plot_scatter(
    z: np.ndarray,
    labels: list[str],
    include_types: Iterable[str],
    out_path: Path,
) -> None:
    include_types = list(include_types)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)

    for mem_type in include_types:
        idx = [i for i, lb in enumerate(labels) if lb == mem_type]
        if not idx:
            continue

        pts = z[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=COLORS[mem_type],
            marker=MARKERS[mem_type],
            s=POINT_SIZE[mem_type],
            alpha=0.58,
            edgecolors=COLORS[mem_type],
            linewidths=0.8,
            label=f"{mem_type} (n={len(idx)})",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_with_ellipses(
    z: np.ndarray,
    labels: list[str],
    include_types: Iterable[str],
    out_path: Path,
) -> None:
    include_types = list(include_types)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)

    for mem_type in include_types:
        idx = [i for i, lb in enumerate(labels) if lb == mem_type]
        if not idx:
            continue

        pts = z[idx]
        add_cov_ellipse(ax, pts, COLORS[mem_type], n_std=2.0, alpha=0.18)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=COLORS[mem_type],
            marker=MARKERS[mem_type],
            s=POINT_SIZE[mem_type],
            alpha=0.54,
            edgecolors=COLORS[mem_type],
            linewidths=0.8,
            label=f"{mem_type} (n={len(idx)})",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_with_memid_text(
    z: np.ndarray,
    labels: list[str],
    mem_ids: list[str],
    include_types: Iterable[str],
    out_path: Path,
) -> None:
    include_types = list(include_types)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)

    for mem_type in include_types:
        idx = [i for i, lb in enumerate(labels) if lb == mem_type]
        if not idx:
            continue

        pts = z[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=COLORS[mem_type],
            marker=MARKERS[mem_type],
            s=POINT_SIZE[mem_type],
            alpha=0.22,
            edgecolors=COLORS[mem_type],
            linewidths=0.7,
            label=f"{mem_type} (n={len(idx)})",
        )
        for i in idx:
            mem_num = mem_ids[i].split("_")[-1]
            ax.text(
                z[i, 0],
                z[i, 1],
                mem_num,
                color=COLORS[mem_type],
                fontsize=5.4,
                ha="center",
                va="center",
                alpha=0.92,
            )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    vectors, labels, mem_ids = load_embeddings(MEM_PATH)
    z = compute_tsne(vectors)

    # 1) no-ellipse scatter for four types
    out_scatter = BASE_DIR / "memory_tsne_4types_styled.pdf"
    plot_scatter(
        z=z,
        labels=labels,
        include_types=["knowledge", "experience", "integration", "idea"],
        out_path=out_scatter,
    )

    # 2) ellipse plot for four types
    out_full = BASE_DIR / "memory_tsne_4types_styled_with_ellipses.pdf"
    plot_with_ellipses(
        z=z,
        labels=labels,
        include_types=["knowledge", "experience", "integration", "idea"],
        out_path=out_full,
    )

    # 3) ellipse plot without knowledge
    out_no_knowledge = BASE_DIR / "memory_tsne_4types_styled_with_ellipses_no_knowledge.pdf"
    plot_with_ellipses(
        z=z,
        labels=labels,
        include_types=["experience", "integration", "idea"],
        out_path=out_no_knowledge,
    )

    # 4) scatter with mem_id numbers shown on markers
    out_memid = BASE_DIR / "memory_tsne_4types_styled_with_memid.pdf"
    plot_with_memid_text(
        z=z,
        labels=labels,
        mem_ids=mem_ids,
        include_types=["knowledge", "experience", "integration", "idea"],
        out_path=out_memid,
    )

    print(f"saved: {out_scatter.relative_to(BASE_DIR)}")
    print(f"saved: {out_full.relative_to(BASE_DIR)}")
    print(f"saved: {out_no_knowledge.relative_to(BASE_DIR)}")
    print(f"saved: {out_memid.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
