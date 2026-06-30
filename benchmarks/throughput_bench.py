"""
吞吐量对比 Benchmark：Continuous Batching vs Naive Batching

Naive Batching（静态批处理）：
  - 将所有请求分成固定大小的 batch
  - 每个 batch 内所有序列跑完才能开始下一个 batch
  - 短序列等长序列 → GPU 利用率低

Continuous Batching（连续批处理）：
  - 每步结束后立即检查完成的序列
  - 有空位就拉入等待队列中的新请求
  - GPU 始终保持高利用率

运行方式：
    python -m benchmarks.throughput_bench
    python -m benchmarks.throughput_bench --num-requests 100 --max-tokens 128
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

# 确保包路径正确
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import LLMEngine
from engine.sequence import SamplingParams


# ──────────────────────────────────────────────────────────────────────────────
# Naive Batching 实现（用于对比）
# ──────────────────────────────────────────────────────────────────────────────

def run_naive_batching(
    prompts: List[str],
    max_tokens: int,
    batch_size: int,
    decode_time_per_step: float,
    eos_probability: float,
    seed: int,
) -> Tuple[float, int, float]:
    """
    模拟 Naive Batching：将 prompts 按 batch_size 切分，顺序处理每个 batch。

    Returns:
        (total_time, total_output_tokens, throughput)
    """
    from engine.model_runner import MockModelRunner
    from engine.sequence import Sequence, SamplingParams

    runner = MockModelRunner(
        decode_time_per_step=decode_time_per_step,
        eos_probability=eos_probability,
        seed=seed,
    )

    total_output_tokens = 0
    start = time.monotonic()

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start: batch_start + batch_size]
        # 模拟一批序列：prefill 一次，decode 直到全部 done 或 max_tokens
        seqs = [
            Sequence(
                seq_id=batch_start + i,
                prompt_token_ids=[ord(c) % 50257 for c in p[:20]],
                block_size=16,
                sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    eos_token_id=runner.eos_token_id,
                ),
            )
            for i, p in enumerate(batch)
        ]

        # prefill 一次（所有序列一起）
        from engine.sequence import SequenceStatus
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING
        prefill_tokens = runner.step(prefill_seqs=seqs, decode_seqs=[])
        for seq in seqs:
            seq.append_token(prefill_tokens.get(seq.seq_id, 1))

        # decode 直到所有序列结束（等最长的那条 — naive batching 的核心缺陷）
        active = list(seqs)
        while active:
            new_tokens = runner.step(prefill_seqs=[], decode_seqs=active)
            still_active = []
            for seq in active:
                seq.append_token(new_tokens.get(seq.seq_id, 1))
                if not seq.should_stop():
                    still_active.append(seq)
            active = still_active

        total_output_tokens += sum(s.num_output_tokens for s in seqs)

    total_time = time.monotonic() - start
    throughput = total_output_tokens / total_time if total_time > 0 else 0
    return total_time, total_output_tokens, throughput


# ──────────────────────────────────────────────────────────────────────────────
# Continuous Batching 实现
# ──────────────────────────────────────────────────────────────────────────────

def run_continuous_batching(
    prompts: List[str],
    max_tokens: int,
    num_kv_blocks: int,
    max_num_seqs: int,
    decode_time_per_step: float,
    eos_probability: float,
    seed: int,
) -> Tuple[float, int, float]:
    """
    使用 LLMEngine（Continuous Batching）处理所有 prompts。

    Returns:
        (total_time, total_output_tokens, throughput)
    """
    engine = LLMEngine.from_config(
        num_kv_blocks=num_kv_blocks,
        block_size=16,
        max_num_seqs=max_num_seqs,
        use_real_model=False,
        decode_time_per_step=decode_time_per_step,
        eos_probability=eos_probability,
        seed=seed,
    )

    results = engine.generate(
        prompts=prompts,
        max_tokens=max_tokens,
        verbose=False,
    )

    total_output_tokens = sum(len(r.output_token_ids) for r in results)
    total_time = engine.stats.total_time
    throughput = total_output_tokens / total_time if total_time > 0 else 0
    return total_time, total_output_tokens, throughput


# ──────────────────────────────────────────────────────────────────────────────
# 主 Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args) -> None:
    """运行完整对比 benchmark，打印结果并生成可视化图表。"""

    import random
    rng = random.Random(args.seed)

    # 生成模拟 prompts（随机长度，更真实）
    sample_prompts = [
        "The future of artificial intelligence is",
        "In a world where machines can think,",
        "The history of computing began with",
        "Scientists have discovered that",
        "The most important skill in the 21st century is",
        "Once upon a time in a land far away,",
        "The key to understanding large language models is",
        "Climate change will affect",
    ]
    prompts = [rng.choice(sample_prompts) for _ in range(args.num_requests)]

    print("=" * 60)
    print("  Mini LLM Engine — Throughput Benchmark")
    print("=" * 60)
    print(f"  Requests         : {args.num_requests}")
    print(f"  Max tokens/req   : {args.max_tokens}")
    print(f"  EOS probability  : {args.eos_prob:.0%}  (avg len ≈ {1/args.eos_prob:.0f} tokens)")
    print(f"  Decode latency   : {args.decode_ms}ms/step (simulated)")
    print(f"  Naive batch size : {args.batch_size}")
    print(f"  CB max seqs      : {args.max_seqs}")
    print(f"  KV blocks        : {args.num_kv_blocks}")
    print("=" * 60)

    # ── Naive Batching ─────────────────────────────────────────────────────
    print("\n[1/2] Running Naive Batching ...")
    naive_time, naive_tokens, naive_tput = run_naive_batching(
        prompts=prompts,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        decode_time_per_step=args.decode_ms / 1000,
        eos_probability=args.eos_prob,
        seed=args.seed,
    )
    print(f"      Time: {naive_time:.2f}s  |  Tokens: {naive_tokens}  |  Throughput: {naive_tput:.1f} tok/s")

    # ── Continuous Batching ────────────────────────────────────────────────
    print("\n[2/2] Running Continuous Batching ...")
    cb_time, cb_tokens, cb_tput = run_continuous_batching(
        prompts=prompts,
        max_tokens=args.max_tokens,
        num_kv_blocks=args.num_kv_blocks,
        max_num_seqs=args.max_seqs,
        decode_time_per_step=args.decode_ms / 1000,
        eos_probability=args.eos_prob,
        seed=args.seed,
    )
    print(f"      Time: {cb_time:.2f}s  |  Tokens: {cb_tokens}  |  Throughput: {cb_tput:.1f} tok/s")

    # ── 结果对比 ──────────────────────────────────────────────────────────
    speedup = cb_tput / max(naive_tput, 1e-9)
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Strategy':<25} {'Time':>8} {'Tokens':>8} {'Throughput':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*12}")
    print(f"  {'Naive Batching':<25} {naive_time:>7.2f}s {naive_tokens:>8} {naive_tput:>10.1f}/s")
    print(f"  {'Continuous Batching':<25} {cb_time:>7.2f}s {cb_tokens:>8} {cb_tput:>10.1f}/s")
    print(f"  {'':25} {'':8} {'':8}")
    print(f"  🚀 Speedup: {speedup:.2f}x  ({(speedup-1)*100:+.0f}% throughput improvement)")
    print("=" * 60)

    # ── 可视化 ────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            _plot_results(
                naive_tput=naive_tput,
                cb_tput=cb_tput,
                naive_time=naive_time,
                cb_time=cb_time,
                speedup=speedup,
                args=args,
            )
        except ImportError:
            print("\n[Info] Install matplotlib to enable plots: pip install matplotlib")


def _plot_results(naive_tput, cb_tput, naive_time, cb_time, speedup, args):
    """生成对比柱状图。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Continuous Batching vs Naive Batching\n"
        f"({args.num_requests} requests, max_tokens={args.max_tokens}, "
        f"decode_latency={args.decode_ms}ms)",
        fontsize=13, fontweight="bold"
    )

    colors = ["#e74c3c", "#2ecc71"]
    labels = ["Naive\nBatching", "Continuous\nBatching"]

    # 图 1：吞吐量
    ax1 = axes[0]
    bars = ax1.bar(labels, [naive_tput, cb_tput], color=colors, width=0.5,
                   edgecolor="white", linewidth=1.5)
    ax1.set_title("Throughput (tokens/sec) ↑", fontsize=12, pad=10)
    ax1.set_ylabel("Output tokens / second")
    for bar, val in zip(bars, [naive_tput, cb_tput]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim(0, max(naive_tput, cb_tput) * 1.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.98, 0.97, f"🚀 {speedup:.2f}x faster",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=11, color="#2ecc71", fontweight="bold")

    # 图 2：总耗时
    ax2 = axes[1]
    bars2 = ax2.bar(labels, [naive_time, cb_time], color=colors, width=0.5,
                    edgecolor="white", linewidth=1.5)
    ax2.set_title("Total Time (seconds) ↓", fontsize=12, pad=10)
    ax2.set_ylabel("Seconds")
    for bar, val in zip(bars2, [naive_time, cb_time]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}s", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylim(0, max(naive_time, cb_time) * 1.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "results" / "throughput_comparison.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {out_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark: Continuous Batching vs Naive Batching"
    )
    parser.add_argument("--num-requests", type=int, default=50,
                        help="Number of requests to process (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max output tokens per request (default: 64)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Naive batch size (default: 8)")
    parser.add_argument("--max-seqs", type=int, default=32,
                        help="CB max concurrent sequences (default: 32)")
    parser.add_argument("--num-kv-blocks", type=int, default=512,
                        help="Number of KV cache blocks (default: 512)")
    parser.add_argument("--decode-ms", type=float, default=2.0,
                        help="Simulated decode latency per step in ms (default: 2.0)")
    parser.add_argument("--eos-prob", type=float, default=0.05,
                        help="EOS token probability per step (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib visualization")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
