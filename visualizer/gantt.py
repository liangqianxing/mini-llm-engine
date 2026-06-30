"""
调度器 Gantt 图可视化。

展示每条序列在时间轴上的状态（waiting / prefill / decoding / finished），
直观对比 Naive Batching 和 Continuous Batching 的 GPU 利用率差异。

运行方式：
    python -m visualizer.gantt
    python -m visualizer.gantt --num-requests 10 --max-tokens 30
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TimelineEvent:
    """记录一个序列在一个时间片内的状态。"""
    seq_id: int
    step: int
    phase: str       # "prefill" / "decoding" / "waiting" / "finished"
    tokens: int = 0  # 该步的 token 数


def simulate_continuous_batching_timeline(
    num_requests: int,
    prompt_len: int,
    output_lengths: List[int],
    max_concurrent: int,
) -> Tuple[List[TimelineEvent], int]:
    """
    模拟 Continuous Batching 的调度时间线。

    Returns:
        (events, total_steps)
    """
    events: List[TimelineEvent] = []
    waiting = list(range(num_requests))  # 等待队列
    running: Dict[int, int] = {}         # seq_id → 已生成 tokens
    finished = set()

    step = 0
    while waiting or running:
        step += 1

        # 记录 waiting 状态
        for req_id in waiting:
            events.append(TimelineEvent(seq_id=req_id, step=step, phase="waiting"))

        # 拉入新请求
        while waiting and len(running) < max_concurrent:
            req_id = waiting.pop(0)
            running[req_id] = 0
            events.append(TimelineEvent(seq_id=req_id, step=step, phase="prefill", tokens=prompt_len))

        # decode 一步
        to_finish = []
        for req_id in running:
            if req_id not in [e.seq_id for e in events if e.step == step]:
                events.append(TimelineEvent(seq_id=req_id, step=step, phase="decoding", tokens=1))
            running[req_id] += 1
            if running[req_id] >= output_lengths[req_id]:
                to_finish.append(req_id)

        for req_id in to_finish:
            del running[req_id]
            finished.add(req_id)

        if step > 500:  # 防止无限循环
            break

    return events, step


def simulate_naive_batching_timeline(
    num_requests: int,
    prompt_len: int,
    output_lengths: List[int],
    batch_size: int,
) -> Tuple[List[TimelineEvent], int]:
    """
    模拟 Naive Batching 的调度时间线。
    每个 batch 中的序列必须等最长的序列完成才能开始下一批。
    """
    events: List[TimelineEvent] = []
    step = 0

    for batch_start in range(0, num_requests, batch_size):
        batch = list(range(batch_start, min(batch_start + batch_size, num_requests)))
        max_len = max(output_lengths[i] for i in batch)

        # 记录 waiting（这批开始前的序列等待）
        for future_id in range(batch_start + batch_size, num_requests):
            for s in range(step, step + max_len + 1):
                events.append(TimelineEvent(seq_id=future_id, step=s, phase="waiting"))

        # prefill
        step += 1
        for req_id in batch:
            events.append(TimelineEvent(seq_id=req_id, step=step, phase="prefill", tokens=prompt_len))

        # decode（等最长序列完成）
        for t in range(max_len):
            step += 1
            for req_id in batch:
                if t < output_lengths[req_id]:
                    events.append(TimelineEvent(seq_id=req_id, step=step, phase="decoding", tokens=1))
                else:
                    # 序列已完成，但 batch 还未结束 → GPU 槽位空闲（浪费！）
                    events.append(TimelineEvent(seq_id=req_id, step=step, phase="finished"))

    return events, step


def plot_gantt(
    num_requests: int = 8,
    max_tokens: int = 30,
    batch_size: int = 4,
    max_concurrent: int = 4,
    prompt_len: int = 10,
    eos_prob: float = 0.08,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> None:
    """绘制 Continuous Batching vs Naive Batching 的 Gantt 图对比。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[Error] Install matplotlib: pip install matplotlib")
        return

    rng = random.Random(seed)
    output_lengths = []
    for _ in range(num_requests):
        for l in range(1, max_tokens + 1):
            if rng.random() < eos_prob or l == max_tokens:
                output_lengths.append(l)
                break

    print(f"Output lengths: {output_lengths}")
    print(f"Average: {sum(output_lengths)/len(output_lengths):.1f} tokens")

    # 模拟两种策略
    cb_events, cb_steps = simulate_continuous_batching_timeline(
        num_requests, prompt_len, output_lengths, max_concurrent
    )
    naive_events, naive_steps = simulate_naive_batching_timeline(
        num_requests, prompt_len, output_lengths, batch_size
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Scheduling Timeline: Continuous Batching vs Naive Batching",
                 fontsize=14, fontweight="bold")

    phase_colors = {
        "waiting":  "#ecf0f1",
        "prefill":  "#f39c12",
        "decoding": "#2ecc71",
        "finished": "#bdc3c7",
    }

    def draw_gantt(ax, events, total_steps, title):
        for event in events:
            color = phase_colors.get(event.phase, "#95a5a6")
            ax.barh(
                y=event.seq_id,
                width=1,
                left=event.step - 1,
                height=0.6,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.9,
            )

        ax.set_yticks(range(num_requests))
        ax.set_yticklabels([f"Req {i}" for i in range(num_requests)], fontsize=9)
        ax.set_xlabel("Decode Step", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(0, total_steps + 1)
        ax.set_ylim(-0.5, num_requests - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(total_steps + 0.5, num_requests / 2, f"{total_steps}\nsteps",
                ha="left", va="center", fontsize=9, color="gray")

        # GPU 利用率（每步有多少序列在 decode/prefill）
        step_util = {}
        for e in events:
            if e.phase in ("decoding", "prefill"):
                step_util[e.step] = step_util.get(e.step, 0) + 1
        if step_util:
            avg_util = sum(step_util.values()) / (total_steps * num_requests)
            ax.text(0.02, 0.97, f"GPU Util: {avg_util:.0%}",
                    transform=ax.transAxes, va="top", fontsize=10,
                    color="#2c3e50", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    draw_gantt(axes[0], cb_events, cb_steps,
               f"✅ Continuous Batching  (max_concurrent={max_concurrent})")
    draw_gantt(axes[1], naive_events, naive_steps,
               f"⚠️  Naive Batching  (batch_size={batch_size})")

    # 图例
    legend_patches = [
        mpatches.Patch(color=phase_colors["prefill"], label="Prefill (processing prompt)"),
        mpatches.Patch(color=phase_colors["decoding"], label="Decoding (generating token)"),
        mpatches.Patch(color=phase_colors["waiting"], label="Waiting (in queue)"),
        mpatches.Patch(color=phase_colors["finished"], label="Finished (idle slot — wasted)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path is None:
        save_path = str(Path(__file__).parent.parent / "results" / "scheduling_gantt.png")
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Gantt] Saved → {save_path}")
    print(f"\nKey insight:")
    print(f"  Naive Batching  : {naive_steps} steps  (short seqs wait for long ones)")
    print(f"  Cont. Batching  : {cb_steps} steps  (new requests fill gaps immediately)")
    print(f"  Step reduction  : {(1 - cb_steps/naive_steps)*100:.0f}%")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Scheduling Gantt Chart")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--eos-prob", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_gantt(
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        eos_prob=args.eos_prob,
        seed=args.seed,
        save_path=args.save,
    )
