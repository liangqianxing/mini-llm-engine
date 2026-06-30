"""
Prefix Cache Benchmark。

对比：有/无 Prefix Caching 时的内存利用率和吞吐量差异。

场景：
  - 所有请求共享相同的 system prompt（常见于 API 服务）
  - 测量 cache hit rate、内存节省比例、吞吐量提升

运行：
    python -m benchmarks.prefix_cache_bench
    python -m benchmarks.prefix_cache_bench --num-requests 50 --shared-prefix-len 64
"""

import argparse
import sys
import time
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import LLMEngine
from engine.sequence import SamplingParams


def run_without_prefix_cache(
    prompts_token_ids: List[List[int]],
    max_tokens: int,
    num_kv_blocks: int,
    block_size: int,
    decode_time: float,
    seed: int,
) -> Dict:
    engine = LLMEngine.from_config(
        num_kv_blocks=num_kv_blocks,
        block_size=block_size,
        max_num_seqs=32,
        prefix_caching=False,
        use_real_model=False,
        decode_time_per_step=decode_time,
        prefill_time_per_token=0.00005,
        eos_probability=0.1,
        seed=seed,
    )

    params = SamplingParams(max_tokens=max_tokens)
    for token_ids in prompts_token_ids:
        engine.add_request(
            prompt="",
            prompt_token_ids=token_ids,
            sampling_params=params,
        )

    start = time.monotonic()
    while engine.scheduler.has_unfinished_seqs:
        engine.step()
    total_time = time.monotonic() - start

    total_tokens = engine.stats.total_output_tokens
    return {
        "strategy": "No Prefix Cache",
        "total_time_s": round(total_time, 3),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / max(total_time, 1e-9), 1),
        "peak_kv_util": engine.stats.peak_kv_utilization,
        "prefix_cache_hit_rate": 0.0,
        "prefix_cache_hits": 0,
    }


def run_with_prefix_cache(
    prompts_token_ids: List[List[int]],
    max_tokens: int,
    num_kv_blocks: int,
    block_size: int,
    decode_time: float,
    seed: int,
) -> Dict:
    engine = LLMEngine.from_config(
        num_kv_blocks=num_kv_blocks,
        block_size=block_size,
        max_num_seqs=32,
        prefix_caching=True,
        max_prefix_cached_blocks=64,
        use_real_model=False,
        decode_time_per_step=decode_time,
        prefill_time_per_token=0.00005,
        eos_probability=0.1,
        seed=seed,
    )

    params = SamplingParams(max_tokens=max_tokens)
    for token_ids in prompts_token_ids:
        engine.add_request(
            prompt="",
            prompt_token_ids=token_ids,
            sampling_params=params,
        )

    start = time.monotonic()
    while engine.scheduler.has_unfinished_seqs:
        engine.step()
    total_time = time.monotonic() - start

    total_tokens = engine.stats.total_output_tokens
    pc = engine.kv_cache_manager.prefix_cache
    pc_stats = pc.stats() if pc else {}

    return {
        "strategy": "Prefix Cache",
        "total_time_s": round(total_time, 3),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / max(total_time, 1e-9), 1),
        "peak_kv_util": engine.stats.peak_kv_utilization,
        "prefix_cache_hit_rate": round(pc_stats.get("hit_rate", 0), 3),
        "prefix_cache_hits": pc_stats.get("num_hits", 0),
        "prefix_cache_misses": pc_stats.get("num_misses", 0),
    }


def build_prompts(
    num_requests: int,
    shared_prefix_len: int,
    unique_suffix_len: int,
    vocab_size: int = 50257,
    seed: int = 42,
) -> List[List[int]]:
    """
    构造 prompt：所有请求共享相同前缀 + 独有的后缀。

    Args:
        shared_prefix_len:  所有请求共享的 token 数（模拟 system prompt）
        unique_suffix_len:  每个请求独有的 token 数（模拟 user query）
    """
    rng = random.Random(seed)
    shared_prefix = [rng.randint(0, vocab_size - 1) for _ in range(shared_prefix_len)]
    prompts = []
    for _ in range(num_requests):
        suffix = [rng.randint(0, vocab_size - 1) for _ in range(unique_suffix_len)]
        prompts.append(shared_prefix + suffix)
    return prompts


def run_benchmark(args) -> None:
    print("=" * 60)
    print("  Mini LLM Engine — Prefix Cache Benchmark")
    print("=" * 60)
    print(f"  Requests          : {args.num_requests}")
    print(f"  Shared prefix len : {args.shared_prefix_len} tokens (system prompt)")
    print(f"  Unique suffix len : {args.unique_suffix_len} tokens (user query)")
    print(f"  Max output tokens : {args.max_tokens}")
    print(f"  KV blocks         : {args.num_kv_blocks}  (block_size={args.block_size})")
    print("=" * 60)

    prompts = build_prompts(
        num_requests=args.num_requests,
        shared_prefix_len=args.shared_prefix_len,
        unique_suffix_len=args.unique_suffix_len,
        seed=args.seed,
    )

    print("\n[1/2] Running without Prefix Cache ...")
    no_cache = run_without_prefix_cache(
        prompts_token_ids=prompts,
        max_tokens=args.max_tokens,
        num_kv_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        decode_time=args.decode_ms / 1000,
        seed=args.seed,
    )
    print(f"      Time: {no_cache['total_time_s']:.3f}s  |  "
          f"Throughput: {no_cache['throughput_tok_s']:.1f} tok/s  |  "
          f"KV util: {no_cache['peak_kv_util']:.1%}")

    print("\n[2/2] Running with Prefix Cache ...")
    with_cache = run_with_prefix_cache(
        prompts_token_ids=prompts,
        max_tokens=args.max_tokens,
        num_kv_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        decode_time=args.decode_ms / 1000,
        seed=args.seed,
    )
    print(f"      Time: {with_cache['total_time_s']:.3f}s  |  "
          f"Throughput: {with_cache['throughput_tok_s']:.1f} tok/s  |  "
          f"KV util: {with_cache['peak_kv_util']:.1%}  |  "
          f"Hit rate: {with_cache['prefix_cache_hit_rate']:.1%}")

    # 结果汇总
    speedup = with_cache["throughput_tok_s"] / max(no_cache["throughput_tok_s"], 1e-9)
    prefix_ratio = args.shared_prefix_len / (args.shared_prefix_len + args.unique_suffix_len)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Strategy':<25} {'Time':>8} {'Throughput':>12} {'Peak KV':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*10}")
    print(f"  {'No Prefix Cache':<25} {no_cache['total_time_s']:>7.3f}s "
          f"{no_cache['throughput_tok_s']:>10.1f}/s "
          f"{no_cache['peak_kv_util']:>9.1%}")
    print(f"  {'Prefix Cache':<25} {with_cache['total_time_s']:>7.3f}s "
          f"{with_cache['throughput_tok_s']:>10.1f}/s "
          f"{with_cache['peak_kv_util']:>9.1%}")
    print(f"\n  🎯 Cache hit rate    : {with_cache['prefix_cache_hit_rate']:.1%}")
    print(f"  💾 Prefix ratio      : {prefix_ratio:.1%} of each prompt is shared")
    print(f"  🚀 Throughput speedup: {speedup:.2f}x")
    print(f"\n  Key insight: with {args.num_requests} requests sharing the same {args.shared_prefix_len}-token")
    print(f"  system prompt, Prefix Cache avoids recomputing the same KV cache")
    print(f"  {args.num_requests - 1} times, saving ~{prefix_ratio:.0%} of prefill compute.")
    print("=" * 60)

    if not args.no_plot:
        try:
            _plot_results(no_cache, with_cache, speedup, args)
        except ImportError:
            print("\n[Info] Install matplotlib: pip install matplotlib")


def _plot_results(no_cache, with_cache, speedup, args):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        f"Prefix Cache Benchmark\n"
        f"({args.num_requests} requests, shared_prefix={args.shared_prefix_len} tokens, "
        f"unique_suffix={args.unique_suffix_len} tokens)",
        fontsize=12, fontweight="bold"
    )

    colors = ["#e74c3c", "#2ecc71"]
    labels = ["No Prefix Cache", "Prefix Cache"]

    # 图 1：吞吐量
    ax1 = axes[0]
    vals = [no_cache["throughput_tok_s"], with_cache["throughput_tok_s"]]
    bars = ax1.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    ax1.set_title("Throughput ↑", fontsize=11)
    ax1.set_ylabel("Output tokens / second")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    ax1.text(0.97, 0.97, f"🚀 {speedup:.2f}x",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=11, color="#2ecc71", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 图 2：KV cache 利用率（峰值）
    ax2 = axes[1]
    utils = [no_cache["peak_kv_util"], with_cache["peak_kv_util"]]
    bars2 = ax2.bar(labels, utils, color=colors, width=0.5, edgecolor="white")
    ax2.set_title("Peak KV Cache Utilization", fontsize=11)
    ax2.set_ylabel("Utilization")
    ax2.set_ylim(0, 1.2)
    for bar, val in zip(bars2, utils):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.1%}", ha="center", va="bottom", fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 图 3：prompt 构成（共享 vs 独有）
    ax3 = axes[2]
    sizes = [args.shared_prefix_len, args.unique_suffix_len]
    labels3 = [
        f"Shared prefix\n({args.shared_prefix_len} tokens)",
        f"Unique suffix\n({args.unique_suffix_len} tokens)",
    ]
    wedge_colors = ["#3498db", "#e67e22"]
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels3, colors=wedge_colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 9}
    )
    ax3.set_title("Prompt Composition", fontsize=11)

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "results" / "prefix_cache_comparison.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {out_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Prefix Cache Benchmark")
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--shared-prefix-len", type=int, default=48,
                        help="Shared system prompt length in tokens (default: 48)")
    parser.add_argument("--unique-suffix-len", type=int, default=16,
                        help="Unique user query length in tokens (default: 16)")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--num-kv-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--decode-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
