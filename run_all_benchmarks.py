"""
一键运行所有 Benchmark 并生成完整报告。

运行：
    python run_all_benchmarks.py           # 完整运行（含图表）
    python run_all_benchmarks.py --no-plot # 不生成图表（CI 环境）
    python run_all_benchmarks.py --fast    # 快速模式（减少请求数）
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def section(title: str) -> None:
    print(f"\n{'━' * 65}")
    print(f"  {title}")
    print(f"{'━' * 65}")


def run_all(args) -> None:
    N = 20 if args.fast else 50          # 请求数
    T = 30 if args.fast else 64          # max tokens
    no_plot = args.no_plot

    print("=" * 65)
    print("  Mini LLM Inference Engine — Full Benchmark Suite")
    print("=" * 65)
    print(f"  Mode    : {'fast' if args.fast else 'full'}")
    print(f"  Requests: {N}   Max tokens: {T}")
    print(f"  Plots   : {'disabled' if no_plot else 'enabled'}")

    results = {}
    t_total = time.monotonic()

    # ── 1. Throughput: Continuous vs Naive Batching ───────────────────────
    section("1 / 5  Throughput: Continuous Batching vs Naive Batching")
    from benchmarks.throughput_bench import (
        run_naive_batching, run_continuous_batching,
    )

    naive_time, naive_tokens, naive_tput = run_naive_batching(
        prompts=["test prompt " + str(i) for i in range(N)],
        max_tokens=T, batch_size=8,
        decode_time_per_step=0.002, eos_probability=0.05, seed=42,
    )
    cb_time, cb_tokens, cb_tput = run_continuous_batching(
        prompts=["test prompt " + str(i) for i in range(N)],
        max_tokens=T, num_kv_blocks=512, max_num_seqs=32,
        decode_time_per_step=0.002, eos_probability=0.05, seed=42,
    )
    speedup_cb = cb_tput / max(naive_tput, 1e-9)
    print(f"  Naive Batching    : {naive_tput:.1f} tok/s  ({naive_time:.2f}s)")
    print(f"  Continuous Batch  : {cb_tput:.1f} tok/s  ({cb_time:.2f}s)")
    print(f"  🚀 Speedup        : {speedup_cb:.2f}x")
    results["throughput_speedup"] = round(speedup_cb, 2)

    # ── 2. Memory: Paged vs Static ────────────────────────────────────────
    section("2 / 5  Memory: Paged KV Cache vs Static Allocation")
    from benchmarks.memory_bench import (
        simulate_output_lengths, simulate_static_allocation,
        simulate_paged_allocation,
    )
    output_lengths = simulate_output_lengths(200, 128, 0.05, 42)
    static_r = simulate_static_allocation(output_lengths, 128, 256 * 16, 8)
    paged_r  = simulate_paged_allocation(output_lengths, 128, 256, 16, 64)
    util_imp = paged_r["utilization"] / max(static_r["utilization"], 1e-9)
    print(f"  Static Allocation : util={static_r['utilization']:.1%}  frag={static_r['avg_fragmentation']:.1%}")
    print(f"  Paged KV Cache    : util={paged_r['utilization']:.1%}  frag={paged_r['avg_fragmentation']:.1%}")
    print(f"  🧠 Memory improvement: {util_imp:.2f}x")
    results["memory_util_improvement"] = round(util_imp, 2)

    # ── 3. Prefix Cache ───────────────────────────────────────────────────
    section("3 / 5  Prefix Cache: Shared System Prompt")
    from benchmarks.prefix_cache_bench import (
        build_prompts, run_without_prefix_cache, run_with_prefix_cache
    )
    M = N
    prompts = build_prompts(M, shared_prefix_len=48, unique_suffix_len=16, seed=42)
    no_cache_r  = run_without_prefix_cache(prompts, T, 256, 16, 0.001, 42)
    with_cache_r = run_with_prefix_cache(prompts, T, 256, 16, 0.001, 42)
    speedup_pc = with_cache_r["throughput_tok_s"] / max(no_cache_r["throughput_tok_s"], 1e-9)
    print(f"  No Prefix Cache   : {no_cache_r['throughput_tok_s']:.1f} tok/s")
    print(f"  Prefix Cache      : {with_cache_r['throughput_tok_s']:.1f} tok/s  "
          f"hit_rate={with_cache_r['prefix_cache_hit_rate']:.1%}")
    print(f"  🚀 Speedup        : {speedup_pc:.2f}x")
    results["prefix_cache_speedup"] = round(speedup_pc, 2)
    results["prefix_cache_hit_rate"] = with_cache_r["prefix_cache_hit_rate"]

    # ── 4. Speculative Decoding ───────────────────────────────────────────
    section("4 / 5  Speculative Decoding (K=4, acceptance_rate=0.7)")
    from engine.speculative import benchmark_speculative
    spec_r = benchmark_speculative(
        num_requests=N, max_tokens=T,
        K=4, acceptance_rate=0.7,
        draft_decode_ms=0.5, target_decode_ms=2.0,
        seed=42,
    )
    print(f"  Standard Decoding : {spec_r['standard']['throughput_tok_s']:.1f} tok/s")
    print(f"  Speculative (K=4) : {spec_r['speculative']['throughput_tok_s']:.1f} tok/s  "
          f"avg_accepted={spec_r['speculative']['avg_accepted_per_step']:.1f}/step")
    print(f"  🚀 Speedup        : {spec_r['speedup']:.2f}x")
    results["speculative_speedup"] = spec_r["speedup"]
    results["speculative_avg_accepted"] = spec_r["speculative"]["avg_accepted_per_step"]

    # ── 5. Chunked Prefill TTFT 对比 ──────────────────────────────────────
    section("5 / 5  Chunked Prefill: TTFT Comparison")
    from engine import LLMEngine
    from engine.sequence import SamplingParams

    # 标准 prefill：长 prompt 一次性处理 → 其他请求的 TTFT 受影响
    long_prompts = ["x" * 50 for _ in range(5)]     # 长 prompt（模拟）
    short_prompts = ["y" * 5 for _ in range(5)]      # 短 prompt（同时存在）
    all_prompts = long_prompts + short_prompts

    def avg_ttft(engine_cfg):
        eng = LLMEngine.from_config(**engine_cfg)
        results_local = eng.generate(all_prompts, max_tokens=15, verbose=False)
        ttfts = [r.ttft for r in results_local if r.ttft > 0]
        return sum(ttfts) / len(ttfts) if ttfts else 0.0

    base_cfg = dict(
        num_kv_blocks=128, block_size=4, max_num_seqs=10,
        chunked_prefill=False,
        eos_probability=0.15, decode_time_per_step=0.001,
        prefill_time_per_token=0.0002, seed=42,
    )
    chunked_cfg = {**base_cfg, "chunked_prefill": True, "max_prefill_tokens_per_step": 8}

    ttft_standard = avg_ttft(base_cfg)
    ttft_chunked  = avg_ttft(chunked_cfg)
    ttft_ratio    = ttft_standard / max(ttft_chunked, 1e-9)

    print(f"  Standard Prefill  : avg TTFT = {ttft_standard * 1000:.1f}ms")
    print(f"  Chunked Prefill   : avg TTFT = {ttft_chunked * 1000:.1f}ms")
    print(f"  📉 TTFT reduction : {(1 - ttft_chunked / max(ttft_standard, 1e-9)) * 100:.0f}%")
    results["chunked_prefill_ttft_ratio"] = round(ttft_ratio, 2)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    section("FULL BENCHMARK SUMMARY")
    print(f"  {'Benchmark':<35} {'Result':>15}")
    print(f"  {'-'*35} {'-'*15}")
    print(f"  {'Continuous Batching speedup':<35} {results['throughput_speedup']:>14.2f}x")
    print(f"  {'Memory util improvement':<35} {results['memory_util_improvement']:>14.2f}x")
    print(f"  {'Prefix Cache speedup':<35} {results['prefix_cache_speedup']:>14.2f}x")
    print(f"  {'Prefix Cache hit rate':<35} {results['prefix_cache_hit_rate']:>13.1%}")
    print(f"  {'Speculative Decoding speedup':<35} {results['speculative_speedup']:>14.2f}x")
    print(f"  {'Speculative avg accepted/step':<35} {results['speculative_avg_accepted']:>14.1f}")
    print(f"  {'Chunked Prefill TTFT ratio':<35} {results['chunked_prefill_ttft_ratio']:>14.2f}x")

    total_elapsed = time.monotonic() - t_total
    print(f"\n  Total benchmark time: {total_elapsed:.1f}s")
    print("=" * 65)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable matplotlib plots (for CI)")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce request counts for faster run")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
