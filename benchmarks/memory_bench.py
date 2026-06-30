"""
KV Cache 内存利用率 Benchmark。

对比：
  - 静态预分配（Static Allocation）：每条序列在开始时就分配 max_tokens 个槽
  - Paged KV Cache（动态分配）    ：按需分配 block，序列结束立即回收

核心指标：
  - 内存利用率（Memory Utilization）：实际使用 / 分配的 token 槽
  - 内存碎片率（Fragmentation）     ：1 - 利用率
  - 最大并发序列数（Max Concurrency）：相同内存下能同时运行的序列数

运行方式：
    python -m benchmarks.memory_bench
    python -m benchmarks.memory_bench --num-requests 100 --max-tokens 256
"""

import argparse
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import BlockAllocator
from engine.sequence import Sequence, SamplingParams, SequenceStatus


def simulate_output_lengths(
    num_requests: int,
    max_tokens: int,
    eos_prob: float,
    seed: int = 42,
) -> List[int]:
    """模拟输出长度的几何分布（等同于每步 eos_prob 概率的停止）。"""
    rng = random.Random(seed)
    lengths = []
    for _ in range(num_requests):
        for l in range(1, max_tokens + 1):
            if rng.random() < eos_prob or l == max_tokens:
                lengths.append(l)
                break
    return lengths


# ──────────────────────────────────────────────────────────────────────────────
# 静态预分配模拟
# ──────────────────────────────────────────────────────────────────────────────

def simulate_static_allocation(
    output_lengths: List[int],
    max_tokens: int,
    num_memory_slots: int,
    batch_size: int,
) -> Dict:
    """
    模拟静态分配策略：每条序列预分配 max_tokens 个 token 槽。

    由于不知道实际输出长度，必须按最坏情况预分配。
    """
    total_allocated = 0
    total_used = 0
    fragmentation_list = []

    for batch_start in range(0, len(output_lengths), batch_size):
        batch = output_lengths[batch_start: batch_start + batch_size]

        # 每条序列预分配 max_tokens 个槽
        allocated_this_batch = len(batch) * max_tokens
        used_this_batch = sum(batch)  # 实际使用的 token 数

        total_allocated += allocated_this_batch
        total_used += used_this_batch

        frag = 1.0 - used_this_batch / allocated_this_batch if allocated_this_batch > 0 else 0
        fragmentation_list.append(frag)

    avg_fragmentation = sum(fragmentation_list) / len(fragmentation_list) if fragmentation_list else 0
    utilization = total_used / total_allocated if total_allocated > 0 else 0

    # 计算能同时放多少条序列（不超过内存限制）
    max_concurrent = num_memory_slots // max_tokens

    return {
        "strategy": "Static Allocation",
        "total_allocated_tokens": total_allocated,
        "total_used_tokens": total_used,
        "utilization": utilization,
        "avg_fragmentation": avg_fragmentation,
        "max_concurrent_seqs": max_concurrent,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Paged KV Cache 模拟
# ──────────────────────────────────────────────────────────────────────────────

def simulate_paged_allocation(
    output_lengths: List[int],
    max_tokens: int,
    num_kv_blocks: int,
    block_size: int,
    max_concurrent_seqs: int,
) -> Dict:
    """
    模拟 Paged KV Cache：按块动态分配，序列结束立即回收。
    """
    allocator = BlockAllocator(num_kv_blocks, block_size)

    # 跟踪每条序列当前持有的块数
    seq_blocks: Dict[int, int] = {}

    total_allocated_tokens = 0
    total_used_tokens = 0
    peak_util = 0.0
    util_history = []

    # 模拟 Continuous Batching：维护 running / waiting 队列
    waiting = list(range(len(output_lengths)))  # 请求 ID
    running: Dict[int, int] = {}  # req_id → 已生成 token 数

    # 每条序列的 prompt 长度（固定为 20 tokens 做简化）
    prompt_len = 20

    completed = 0

    while waiting or running:
        # 尝试从 waiting 拉入新请求
        while waiting and len(running) < max_concurrent_seqs:
            req_id = waiting.pop(0)
            # 分配 prompt 所需的块
            needed = (prompt_len + block_size - 1) // block_size
            if not allocator.can_allocate(needed):
                waiting.insert(0, req_id)
                break
            for _ in range(needed):
                allocator.allocate()
            seq_blocks[req_id] = needed
            running[req_id] = 0
            total_allocated_tokens += needed * block_size

        if not running:
            break

        # 模拟一步 decode：每条序列生成一个 token
        to_finish = []
        for req_id, generated in list(running.items()):
            running[req_id] = generated + 1
            total_used_tokens += 1

            # 检查是否需要新块
            current_tokens = prompt_len + running[req_id]
            blocks_needed = (current_tokens + block_size - 1) // block_size
            if blocks_needed > seq_blocks[req_id]:
                if allocator.can_allocate(1):
                    allocator.allocate()
                    seq_blocks[req_id] += 1
                    total_allocated_tokens += block_size

            # 检查是否完成
            if running[req_id] >= output_lengths[req_id]:
                to_finish.append(req_id)

        # 释放完成序列的块
        for req_id in to_finish:
            blocks = seq_blocks.pop(req_id)
            for _ in range(blocks):
                from engine.block import PhysicalBlock
                # 直接模拟释放（减少空闲块计数）
                allocator._free_blocks.append(PhysicalBlock(block_id=-1))
            running.pop(req_id)
            completed += 1

        # 记录利用率
        util = allocator.utilization
        util_history.append(util)
        peak_util = max(peak_util, util)

    avg_util = sum(util_history) / len(util_history) if util_history else 0
    # paged 利用率 = 实际使用 / 实际分配（block 对齐后）
    utilization = total_used_tokens / total_allocated_tokens if total_allocated_tokens > 0 else 0

    return {
        "strategy": "Paged KV Cache",
        "total_allocated_tokens": total_allocated_tokens,
        "total_used_tokens": total_used_tokens,
        "utilization": utilization,
        "avg_fragmentation": 1.0 - utilization,
        "max_concurrent_seqs": max_concurrent_seqs,
        "peak_kv_utilization": peak_util,
        "avg_kv_utilization": avg_util,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 主 Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_memory_benchmark(args) -> None:
    print("=" * 60)
    print("  Mini LLM Engine — KV Cache Memory Benchmark")
    print("=" * 60)
    print(f"  Requests         : {args.num_requests}")
    print(f"  Max tokens       : {args.max_tokens}")
    print(f"  Block size       : {args.block_size}")
    print(f"  KV blocks        : {args.num_kv_blocks}")
    print(f"  Memory slots     : {args.num_kv_blocks * args.block_size}")
    print(f"  EOS probability  : {args.eos_prob:.0%}  (avg len ≈ {1/args.eos_prob:.0f} tokens)")
    print("=" * 60)

    # 生成输出长度分布
    output_lengths = simulate_output_lengths(
        args.num_requests, args.max_tokens, args.eos_prob, args.seed
    )
    avg_len = sum(output_lengths) / len(output_lengths)
    print(f"\n  Output length distribution: avg={avg_len:.1f}, "
          f"min={min(output_lengths)}, max={max(output_lengths)}")

    num_memory_slots = args.num_kv_blocks * args.block_size

    # 静态分配
    static_results = simulate_static_allocation(
        output_lengths=output_lengths,
        max_tokens=args.max_tokens,
        num_memory_slots=num_memory_slots,
        batch_size=8,
    )

    # Paged 分配
    paged_results = simulate_paged_allocation(
        output_lengths=output_lengths,
        max_tokens=args.max_tokens,
        num_kv_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        max_concurrent_seqs=min(64, args.num_kv_blocks * args.block_size // avg_len),
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("  MEMORY EFFICIENCY COMPARISON")
    print("=" * 60)

    metrics = [
        ("Utilization", "utilization", "{:.1%}"),
        ("Fragmentation", "avg_fragmentation", "{:.1%}"),
        ("Max Concurrent Seqs", "max_concurrent_seqs", "{:d}"),
    ]

    for label, key, fmt in metrics:
        sv = static_results[key]
        pv = paged_results[key]
        sv_str = fmt.format(int(sv) if isinstance(sv, float) and key == "max_concurrent_seqs" else sv)
        pv_str = fmt.format(int(pv) if isinstance(pv, float) and key == "max_concurrent_seqs" else pv)
        print(f"  {label:<25} Static: {sv_str:>8}   Paged: {pv_str:>8}")

    concurrency_gain = paged_results["max_concurrent_seqs"] / max(static_results["max_concurrent_seqs"], 1)
    util_improvement = paged_results["utilization"] / max(static_results["utilization"], 1e-9)
    print(f"\n  🧠 Memory utilization: {util_improvement:.2f}x improvement")
    print(f"  🔀 Concurrent capacity: {concurrency_gain:.2f}x improvement")
    print("=" * 60)

    # 可视化
    if not args.no_plot:
        try:
            _plot_memory_results(static_results, paged_results, output_lengths, args)
        except ImportError:
            print("\n[Info] Install matplotlib: pip install matplotlib")


def _plot_memory_results(static, paged, output_lengths, args):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"KV Cache Memory Analysis — Static vs Paged\n"
        f"({args.num_requests} requests, max_tokens={args.max_tokens}, "
        f"block_size={args.block_size})",
        fontsize=12, fontweight="bold"
    )

    colors = ["#e74c3c", "#2ecc71"]

    # 图 1：内存利用率
    ax1 = axes[0]
    vals = [static["utilization"], paged["utilization"]]
    bars = ax1.bar(["Static\nAllocation", "Paged\nKV Cache"], vals,
                   color=colors, width=0.5, edgecolor="white")
    ax1.set_title("Memory Utilization ↑", fontsize=11)
    ax1.set_ylabel("Utilized / Allocated")
    ax1.set_ylim(0, 1.2)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", va="bottom", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 图 2：碎片率
    ax2 = axes[1]
    frags = [static["avg_fragmentation"], paged["avg_fragmentation"]]
    bars2 = ax2.bar(["Static\nAllocation", "Paged\nKV Cache"], frags,
                    color=colors, width=0.5, edgecolor="white")
    ax2.set_title("Memory Fragmentation ↓", fontsize=11)
    ax2.set_ylabel("Fragmented / Allocated")
    ax2.set_ylim(0, 1.2)
    for bar, val in zip(bars2, frags):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", va="bottom", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 图 3：输出长度分布
    ax3 = axes[2]
    ax3.hist(output_lengths, bins=20, color="#3498db", alpha=0.8, edgecolor="white")
    ax3.axvline(sum(output_lengths) / len(output_lengths), color="#e74c3c",
                linestyle="--", linewidth=2, label=f"mean={sum(output_lengths)/len(output_lengths):.0f}")
    ax3.set_title("Output Length Distribution", fontsize=11)
    ax3.set_xlabel("Output tokens per request")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "results" / "memory_comparison.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {out_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="KV Cache Memory Utilization Benchmark"
    )
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=256)
    parser.add_argument("--eos-prob", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_memory_benchmark(parse_args())
