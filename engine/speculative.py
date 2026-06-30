"""
Speculative Decoding（投机解码）。

核心思路
────────
标准解码：每步调用 target model 生成 1 个 token（memory-bound，GPU 利用率低）。

投机解码：
  Step 1 (Draft)  : 用小的 draft model 连续生成 K 个候选 token（成本低）
  Step 2 (Verify) : 将 K 个候选 token 送给 target model 并行验证
                    （利用 prefill 的并行计算能力，成本约等于 1 个 decode step）
  Step 3 (Accept) : 从左到右检查每个 token：
                    - target 同意 → accept（计入有效输出）
                    - target 拒绝 → 截断，使用 target 的 token

有效加速比：
  - 理想情况（draft 全部被接受）：K 个 token / 1 个 verify step = K倍加速
  - 实际（接受率 α ∈ [0,1]）：expected_tokens = (1 - α^(K+1)) / (1 - α)

最佳实践：
  - K = 4~8（经验值）
  - draft model 应比 target model 小 5-10 倍（如 GPT-2 small vs GPT-2 large）

Ref: "Fast Inference from Transformers via Speculative Decoding"
     Leviathan et al., ICML 2023
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .model_runner import BaseModelRunner, MockModelRunner
from .sequence import Sequence, SequenceStatus


@dataclass
class SpeculativeStep:
    """单步投机解码的结果。"""
    seq_id: int
    draft_tokens: List[int]        # draft model 生成的 K 个候选 token
    accepted_tokens: List[int]     # 被 target model 接受的 token
    num_accepted: int              # 接受数量（0~K）
    target_token: int              # target 在截断点给出的 token


class SpeculativeDecoder:
    """
    投机解码器。

    将 draft model 和 target model 组合，实现 speculative decoding。

    Args:
        draft_runner:  草稿模型 runner（快速、小模型）
        target_runner: 目标模型 runner（慢速、大模型）
        num_speculative_tokens: 每步草稿的 token 数（K）
        acceptance_rate:        模拟的接受率（仅 Mock 模式有效）
        seed:                   随机种子
    """

    def __init__(
        self,
        draft_runner: BaseModelRunner,
        target_runner: BaseModelRunner,
        num_speculative_tokens: int = 4,
        acceptance_rate: float = 0.7,
        seed: Optional[int] = 42,
    ) -> None:
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.K = num_speculative_tokens
        self.acceptance_rate = acceptance_rate
        self._rng = random.Random(seed)

        # 统计
        self.total_draft_tokens: int = 0
        self.total_accepted_tokens: int = 0
        self.total_verify_steps: int = 0

    @property
    def effective_acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def speedup_ratio(self) -> float:
        """
        理论加速比：accepted tokens per verify step。
        标准 decode 每 step 产出 1 token；speculative 每 step 平均产出更多。
        """
        if self.total_verify_steps == 0:
            return 1.0
        return self.total_accepted_tokens / self.total_verify_steps

    def step(
        self,
        seqs: List[Sequence],
    ) -> Dict[int, List[int]]:
        """
        对一批序列执行一步投机解码。

        Returns:
            Dict[seq_id → new_tokens_list]
            （每条序列本步接受的 token 列表，长度 ∈ [1, K+1]）
        """
        if not seqs:
            return {}

        results: Dict[int, List[int]] = {}

        # ── Phase 1: Draft ────────────────────────────────────────────────
        # draft model 连续生成 K 个 token（独立运行，不影响 target）
        # 为简化实现，我们对每条序列独立处理
        draft_tokens_per_seq: Dict[int, List[int]] = {}

        for _ in range(self.K):
            # 每次 draft step 处理所有序列
            new_draft = self.draft_runner.step(
                prefill_seqs=[],
                decode_seqs=seqs,
            )
            for seq in seqs:
                draft_tok = new_draft.get(seq.seq_id, 0)
                if seq.seq_id not in draft_tokens_per_seq:
                    draft_tokens_per_seq[seq.seq_id] = []
                draft_tokens_per_seq[seq.seq_id].append(draft_tok)
            self.total_draft_tokens += len(seqs)

        # ── Phase 2: Verify ───────────────────────────────────────────────
        # target model 对 K 个 draft token 做并行验证（1 次 prefill-like forward）
        # 在 Mock 模式下，我们模拟接受率
        verify_tokens = self.target_runner.step(
            prefill_seqs=[],
            decode_seqs=seqs,
        )
        self.total_verify_steps += 1

        # ── Phase 3: Accept/Reject ────────────────────────────────────────
        for seq in seqs:
            draft_toks = draft_tokens_per_seq.get(seq.seq_id, [])
            target_tok = verify_tokens.get(seq.seq_id, 0)

            accepted: List[int] = []
            for draft_tok in draft_toks:
                # 模拟接受/拒绝：target 同意 draft token 的概率 = acceptance_rate
                if self._rng.random() < self.acceptance_rate:
                    accepted.append(draft_tok)
                else:
                    # 拒绝：截断，使用 target 的 token
                    accepted.append(target_tok)
                    break
            else:
                # 所有 draft token 都被接受，额外获得 target token
                accepted.append(target_tok)

            self.total_accepted_tokens += len(accepted)
            results[seq.seq_id] = accepted

        return results

    def stats(self) -> Dict:
        return {
            "K": self.K,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_verify_steps": self.total_verify_steps,
            "effective_acceptance_rate": round(self.effective_acceptance_rate, 3),
            "avg_tokens_per_step": round(self.speedup_ratio, 2),
            "theoretical_max_speedup": self.K + 1,
        }

    def __repr__(self) -> str:
        return (
            f"SpeculativeDecoder("
            f"K={self.K}, "
            f"accept={self.effective_acceptance_rate:.1%}, "
            f"speedup={self.speedup_ratio:.2f}x)"
        )


def benchmark_speculative(
    num_requests: int = 30,
    max_tokens: int = 50,
    K: int = 4,
    acceptance_rate: float = 0.7,
    draft_decode_ms: float = 0.5,   # draft model 更快
    target_decode_ms: float = 2.0,  # target model
    eos_prob: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    对比标准解码 vs 投机解码的吞吐量。

    Returns:
        比较结果字典
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.sequence import Sequence, SamplingParams, SequenceStatus

    prompts = list(range(num_requests))  # 用 ID 代替真实 prompt

    # ── 标准解码（只用 target model）──────────────────────────────────────
    target_runner = MockModelRunner(
        decode_time_per_step=target_decode_ms / 1000,
        eos_probability=eos_prob,
        seed=seed,
    )

    rng = random.Random(seed)

    def make_seqs(runner_seed):
        r = random.Random(runner_seed)
        seqs = []
        for i in range(num_requests):
            s = Sequence(
                seq_id=i,
                prompt_token_ids=[r.randint(0, 100) for _ in range(10)],
                block_size=16,
                sampling_params=SamplingParams(max_tokens=max_tokens, eos_token_id=50256),
            )
            s.status = SequenceStatus.RUNNING
            s.num_prefilled_tokens = s.prompt_len
            seqs.append(s)
        return seqs

    # 标准解码
    standard_seqs = make_seqs(seed)
    t0 = time.monotonic()
    total_standard_tokens = 0
    for _ in range(max_tokens):
        active = [s for s in standard_seqs if not s.is_finished]
        if not active:
            break
        new_toks = target_runner.step(prefill_seqs=[], decode_seqs=active)
        for s in active:
            tok = new_toks.get(s.seq_id, 0)
            s.append_token(tok)
            total_standard_tokens += 1
            if s.should_stop():
                s.mark_finished()
    standard_time = time.monotonic() - t0
    standard_throughput = total_standard_tokens / standard_time

    # ── 投机解码 ──────────────────────────────────────────────────────────
    draft_runner = MockModelRunner(
        decode_time_per_step=draft_decode_ms / 1000,
        eos_probability=eos_prob,
        seed=seed + 1,
    )
    spec_target_runner = MockModelRunner(
        decode_time_per_step=target_decode_ms / 1000,
        eos_probability=eos_prob,
        seed=seed,
    )
    spec_decoder = SpeculativeDecoder(
        draft_runner=draft_runner,
        target_runner=spec_target_runner,
        num_speculative_tokens=K,
        acceptance_rate=acceptance_rate,
        seed=seed,
    )

    spec_seqs = make_seqs(seed + 10)
    t1 = time.monotonic()
    total_spec_tokens = 0
    spec_steps = 0
    for _ in range(max_tokens):
        active = [s for s in spec_seqs if not s.is_finished]
        if not active:
            break
        new_tok_map = spec_decoder.step(active)
        spec_steps += 1
        for s in active:
            toks = new_tok_map.get(s.seq_id, [0])
            for tok in toks:
                if s.should_stop():
                    break
                s.append_token(tok)
                total_spec_tokens += 1
            if s.should_stop():
                s.mark_finished()
    spec_time = time.monotonic() - t1
    spec_throughput = total_spec_tokens / spec_time if spec_time > 0 else 0

    speedup = spec_throughput / standard_throughput if standard_throughput > 0 else 1.0

    return {
        "standard": {
            "throughput_tok_s": round(standard_throughput, 1),
            "total_tokens": total_standard_tokens,
            "total_time_s": round(standard_time, 3),
        },
        "speculative": {
            "K": K,
            "acceptance_rate": acceptance_rate,
            "throughput_tok_s": round(spec_throughput, 1),
            "total_tokens": total_spec_tokens,
            "total_time_s": round(spec_time, 3),
            "avg_accepted_per_step": round(spec_decoder.speedup_ratio, 2),
        },
        "speedup": round(speedup, 2),
        "stats": spec_decoder.stats(),
    }
