"""
基础用法示例：使用 MockModelRunner 演示引擎的完整生成流程。

运行：
    python examples/basic_usage.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import LLMEngine
from engine.sequence import SamplingParams


def main():
    print("=" * 55)
    print("  Mini LLM Engine — Basic Usage Demo")
    print("=" * 55)

    # ── 初始化引擎 ─────────────────────────────────────────────────────────
    engine = LLMEngine.from_config(
        num_kv_blocks=128,      # 共 128 个物理块（128 × 16 = 2048 token slots）
        block_size=16,          # 每块 16 个 token
        max_num_seqs=32,        # 最多同时处理 32 条序列
        use_real_model=False,   # 使用 Mock 模型（无 GPU 要求）
        decode_time_per_step=0.001,  # 1ms/step，加快 demo 速度
        eos_probability=0.1,    # 平均生成 ~10 个 token 后结束
        seed=42,
    )

    print(f"\nEngine initialized:")
    print(f"  KV blocks   : {engine.kv_cache_manager.allocator.num_blocks}")
    print(f"  Block size  : {engine.kv_cache_manager.block_size}")
    print(f"  Token slots : {engine.kv_cache_manager.allocator.num_blocks * engine.kv_cache_manager.block_size}")

    # ── 提交请求并生成 ────────────────────────────────────────────────────
    prompts = [
        "The future of artificial intelligence is",
        "In a world where machines can think,",
        "The history of computing began with",
        "Scientists have discovered that",
        "The key to understanding transformers is",
    ]

    print(f"\nSubmitting {len(prompts)} prompts ...\n")

    results = engine.generate(
        prompts=prompts,
        max_tokens=30,
        verbose=True,
    )

    # ── 打印结果 ──────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Generation Results")
    print("─" * 55)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n[{i}] Prompt   : {prompt!r}")
        print(f"    Output   : {result.output_text}")
        print(f"    Tokens   : {len(result.output_token_ids)}")
        print(f"    Latency  : {result.latency:.3f}s")
        print(f"    TTFT     : {result.ttft:.3f}s")
        print(f"    Speed    : {result.throughput:.1f} tok/s")

    # ── 引擎统计 ──────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Engine Statistics")
    print("─" * 55)
    stats = engine.stats
    print(f"  Total requests  : {stats.total_requests}")
    print(f"  Total tokens    : {stats.total_output_tokens}")
    print(f"  Total time      : {stats.total_time:.3f}s")
    print(f"  Throughput      : {stats.throughput:.1f} tok/s")
    print(f"  Total steps     : {stats.num_steps}")
    print(f"  Peak KV util    : {stats.peak_kv_utilization:.1%}")


if __name__ == "__main__":
    main()
