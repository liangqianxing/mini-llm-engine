"""
GPT-2 真实模型示例（需要安装 torch 和 transformers）。

运行：
    pip install torch transformers
    python examples/gpt2_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("请先安装依赖：pip install torch transformers")
        return

    from engine import LLMEngine
    from engine.sequence import SamplingParams

    print("=" * 55)
    print("  Mini LLM Engine — GPT-2 Real Model Demo")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    engine = LLMEngine.from_config(
        num_kv_blocks=64,
        block_size=16,
        max_num_seqs=4,
        use_real_model=True,
        model_name="gpt2",
        device=device,
    )

    prompts = [
        "The capital of France is",
        "Artificial intelligence will",
        "In the beginning,",
    ]

    print(f"\nGenerating with GPT-2 ...\n")
    results = engine.generate(
        prompts=prompts,
        max_tokens=30,
        sampling_params=SamplingParams(max_tokens=30, temperature=0.7),
        verbose=True,
    )

    print("\n" + "─" * 55)
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt : {prompt!r}")
        print(f"Output : {result.output_text!r}")
        print(f"Tokens : {len(result.output_token_ids)}  Latency: {result.latency:.2f}s")


if __name__ == "__main__":
    main()
