from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def load_memory_stats(mem_path: Path) -> dict[int, dict[str, int]]:
    """统计每天每种类型记忆的数量"""
    with mem_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    day_stats = defaultdict(lambda: defaultdict(int))

    for mem in data.get("memory", []):
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
        if mem_type not in ["knowledge", "experience", "integration", "idea"]:
            continue

        metadata = mem.get("metadata", {})
        day = metadata.get("day")

        if day is not None:
            day_stats[int(day)][mem_type] += 1

    return dict(day_stats)


def load_rmse_data(json_path: Path) -> dict[int, tuple[float, float | None]]:
    """从 rmse_by_day.json 读取每天 (rmse, std)；该 JSON 是单一真相源（含模型/口径/纠错说明）。"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rmse_data: dict[int, tuple[float, float | None]] = {}
    for day_str, info in data["days"].items():
        rmse_data[int(day_str)] = (float(info["rmse"]), info.get("std"))
    return rmse_data


def plot_rmse_memory():
    """绘制 RMSE（含误差棒）和累积记忆数量随天数变化的图表。

    RMSE 数据从同目录 rmse_by_day.json 读取（每日头条模型 mean±std，已从 src/ 结果文件核实，
    纠错记录与口径说明见该 JSON 各天的 note 字段）。
    """
    rmse_data = load_rmse_data(BASE_DIR / "rmse_by_day.json")

    # 加载记忆统计
    mem_path = BASE_DIR / "mem.json"
    day_stats = load_memory_stats(mem_path)

    # 准备数据
    days = sorted(set(rmse_data.keys()) & set(day_stats.keys()))
    days = [d for d in days if d > 0]  # 排除 day 0

    rmse_values = [rmse_data[d][0] for d in days]
    # 单划分/单种子天无 std → 误差棒长度 0（不显示）
    rmse_stds = [rmse_data[d][1] or 0 for d in days]

    # 计算每天的增量
    knowledge_daily = [day_stats[d].get("knowledge", 0) for d in days]
    experience_daily = [day_stats[d].get("experience", 0) for d in days]
    integration_daily = [day_stats[d].get("integration", 0) for d in days]
    idea_daily = [day_stats[d].get("idea", 0) for d in days]
    total_daily = [sum(day_stats[d].values()) for d in days]

    # 计算累积和
    knowledge_cumsum = np.cumsum(knowledge_daily)
    experience_cumsum = np.cumsum(experience_daily)
    integration_cumsum = np.cumsum(integration_daily)
    idea_cumsum = np.cumsum(idea_daily)
    total_cumsum = np.cumsum(total_daily)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=180)

    # 左 Y 轴 - RMSE（Run A 红，与 composite panel A 一致）
    color_rmse = "#C0392B"
    ax1.set_xlabel("Day", fontsize=13, weight="bold")
    ax1.set_ylabel("Test RMSE (K)", color=color_rmse, fontsize=13, weight="bold")
    ax1.plot(days, rmse_values, color=color_rmse, marker="o",
             linewidth=2.5, markersize=10,
             label="Test RMSE (mean±SD)", zorder=10)
    # 误差棒 = fold/seed-std（单种子天长度 0）
    ax1.errorbar(days, rmse_values, yerr=rmse_stds, fmt="none",
                 ecolor=color_rmse, elinewidth=1.4, capsize=5,
                 capthick=1.4, zorder=9)
    ax1.tick_params(axis="y", labelcolor=color_rmse, labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)

    # RMSE 轴从 15 K 起（与 composite panel A 一致）
    ax1.set_ylim(15, 52)

    ax1.grid(axis="both", alpha=0.2, linestyle="--")

    # 右 Y 轴 - 累积记忆数量 (蓝绿黄色系)
    ax2 = ax1.twinx()
    # 左轴(RMSE)绘制在右轴(记忆)之上，与 composite panel A 一致
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax2.set_ylabel("Cumulative Memory Count", fontsize=13, weight="bold")

    colors = {
        "knowledge": "#1E90FF",      # 道奇蓝
        "experience": "#32CD32",     # 酸橙绿
        "integration": "#20B2AA",    # 浅海绿
        "idea": "#FFD700",           # 金色
        "total": "#000080",          # 深蓝色
    }

    ax2.plot(days, knowledge_cumsum, color=colors["knowledge"],
             marker="s", linewidth=2, markersize=7, label="Knowledge", alpha=0.8)
    ax2.plot(days, experience_cumsum, color=colors["experience"],
             marker="^", linewidth=2, markersize=7, label="Experience", alpha=0.8)
    ax2.plot(days, integration_cumsum, color=colors["integration"],
             marker="D", linewidth=2, markersize=7, label="Integration", alpha=0.8)
    ax2.plot(days, idea_cumsum, color=colors["idea"],
             marker="v", linewidth=2, markersize=7, label="Idea", alpha=0.8)
    ax2.plot(days, total_cumsum, color=colors["total"],
             marker="p", linewidth=3, markersize=9, label="Total", alpha=0.9)

    ax2.tick_params(axis="y", labelsize=11)

    max_memory = max(total_cumsum)
    ax2.set_ylim(0, max_memory * 1.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.95, fontsize=11, ncol=2)

    # Read the manuscript run label from metadata, including for linked archives.
    with (BASE_DIR / "rmse_by_day.json").open("r", encoding="utf-8") as stream:
        run_label = json.load(stream)["run"]
    suffix = f" (Run {run_label})"
    plt.title(f"Test RMSE and Cumulative Memory Growth by Day{suffix}",
              fontsize=15, weight="bold", pad=15)
    fig.tight_layout()

    out_path = BASE_DIR / "rmse_memory_by_day.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.relative_to(BASE_DIR)}")
    print(f"Saved: {out_path.with_suffix('.svg').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    plot_rmse_memory()
