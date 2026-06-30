"""
LLM Engine：调度器 + 模型运行器的顶层集成。

对外暴露简洁的 generate() 接口，内部协调：
  - Scheduler（连续批处理调度）
  - KVCacheManager（物理块分配）
  - ModelRunner（实际推理）

使用方式：
    engine = LLMEngine.from_config(
        num_kv_blocks=512,
        block_size=16,
        use_real_model=False,   # True 时使用 GPT-2
    )
    results = engine.generate(
        prompts=["Hello, my name is", "The future of AI is"],
        max_tokens=50,
    )
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .kv_cache import KVCacheManager
from .model_runner import BaseModelRunner, MockModelRunner, GPT2ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, SequenceGroup, SequenceStatus


@dataclass
class RequestOutput:
    """单个请求的输出结果。"""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    output_text: str
    latency: float          # 端到端延迟（秒）
    ttft: float             # Time-to-first-token（秒）
    throughput: float       # output tokens/sec（该请求）

    def __repr__(self) -> str:
        short_text = self.output_text[:80] + "..." if len(self.output_text) > 80 else self.output_text
        return (
            f"Output(req={self.request_id}, "
            f"out_tokens={len(self.output_token_ids)}, "
            f"latency={self.latency:.2f}s, "
            f"text='{short_text}')"
        )


@dataclass
class EngineStats:
    """引擎整体运行统计。"""
    total_requests: int = 0
    total_output_tokens: int = 0
    total_time: float = 0.0          # 从第一个请求到最后一个请求完成
    num_steps: int = 0
    peak_kv_utilization: float = 0.0

    @property
    def throughput(self) -> float:
        """系统吞吐量：output tokens/sec。"""
        if self.total_time == 0:
            return 0.0
        return self.total_output_tokens / self.total_time

    @property
    def avg_latency(self) -> float:
        return self.total_time / max(self.total_requests, 1)

    def __repr__(self) -> str:
        return (
            f"EngineStats(\n"
            f"  requests      = {self.total_requests}\n"
            f"  total_tokens  = {self.total_output_tokens}\n"
            f"  throughput    = {self.throughput:.1f} tok/s\n"
            f"  total_time    = {self.total_time:.2f}s\n"
            f"  num_steps     = {self.num_steps}\n"
            f"  peak_kv_util  = {self.peak_kv_utilization:.1%}\n"
            f")"
        )


class LLMEngine:
    """
    LLM 推理引擎。

    Args:
        scheduler:        调度器实例
        model_runner:     模型执行器实例
        tokenizer:        分词器（可选，用于文本输入输出）
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: BaseModelRunner,
        tokenizer=None,
    ) -> None:
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self._next_seq_id: int = 0
        self._next_req_id: int = 0
        self.stats = EngineStats()

        # 存储所有请求的输出
        self._outputs: Dict[str, RequestOutput] = {}
        # 跟踪序列到请求的映射
        self._seq_to_req: Dict[int, str] = {}
        # 跟踪每个请求的原始 prompt 文本
        self._req_prompts: Dict[str, str] = {}

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        num_kv_blocks: int = 256,
        block_size: int = 16,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 4096,
        use_real_model: bool = False,
        model_name: str = "gpt2",
        device: str = "cpu",
        # MockModelRunner 参数
        decode_time_per_step: float = 0.002,
        eos_probability: float = 0.05,
        seed: int = 42,
    ) -> "LLMEngine":
        """便捷工厂方法，从配置参数创建引擎。"""
        kv_manager = KVCacheManager(num_kv_blocks, block_size)
        scheduler = Scheduler(kv_manager, max_num_seqs, max_num_batched_tokens)

        if use_real_model:
            runner = GPT2ModelRunner(model_name=model_name, device=device)
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except ImportError:
                tokenizer = None
        else:
            runner = MockModelRunner(
                decode_time_per_step=decode_time_per_step,
                eos_probability=eos_probability,
                seed=seed,
            )
            tokenizer = None

        return cls(scheduler=scheduler, model_runner=runner, tokenizer=tokenizer)

    # ── 请求提交 ─────────────────────────────────────────────────────────────

    def add_request(
        self,
        prompt: str,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        向引擎提交一个生成请求。

        Args:
            prompt:            文本 prompt（如有 tokenizer 会自动编码）
            prompt_token_ids:  直接提供 token id（优先于 prompt 文本）
            sampling_params:   生成参数
            request_id:        自定义请求 ID（None 时自动生成）

        Returns:
            request_id
        """
        if request_id is None:
            request_id = f"req-{self._next_req_id}"
            self._next_req_id += 1

        # token 化
        if prompt_token_ids is None:
            if self.tokenizer is not None:
                prompt_token_ids = self.tokenizer.encode(prompt)
            else:
                # MockMode：把 prompt 的 ASCII 码当 token id（仅演示用）
                prompt_token_ids = [ord(c) % 50257 for c in prompt[:20]]

        # 创建序列
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=prompt_token_ids,
            block_size=self.scheduler.kv_cache.block_size,
            sampling_params=sampling_params or SamplingParams(),
        )

        # 创建请求组并加入调度器
        seq_group = SequenceGroup(
            request_id=request_id,
            sequences=[seq],
        )
        self.scheduler.add_seq_group(seq_group)
        self._seq_to_req[seq_id] = request_id
        self._req_prompts[request_id] = prompt

        return request_id

    # ── 推理主循环 ────────────────────────────────────────────────────────────

    def step(self) -> List[RequestOutput]:
        """
        执行一个推理步骤，返回本步完成的请求输出列表。

        这是引擎的核心循环体，外部调用 generate() 时会持续调用 step()
        直到所有请求完成。
        """
        # 1. 调度
        sched_output = self.scheduler.schedule()
        if sched_output.is_empty:
            return []

        # 2. 模型推理
        new_token_ids = self.model_runner.step(
            sched_output.prefill_seqs,
            sched_output.decode_seqs,
        )

        # 3. 更新状态，获取完成的请求
        finished_groups = self.scheduler.on_step_done(sched_output, new_token_ids)

        # 4. 更新统计
        self.stats.num_steps += 1
        self.stats.peak_kv_utilization = max(
            self.stats.peak_kv_utilization,
            self.scheduler.kv_cache.utilization,
        )

        # 5. 构造输出
        outputs: List[RequestOutput] = []
        for seq_group in finished_groups:
            for seq in seq_group.seqs:
                req_id = self._seq_to_req.get(seq.seq_id, seq_group.request_id)
                output_token_ids = seq.token_ids[seq.prompt_len:]
                latency = seq.latency or 0.0
                ttft = seq.ttft or 0.0
                out_text = self._decode(output_token_ids)

                result = RequestOutput(
                    request_id=req_id,
                    prompt=self._req_prompts.get(req_id, ""),
                    prompt_token_ids=seq.token_ids[:seq.prompt_len],
                    output_token_ids=output_token_ids,
                    output_text=out_text,
                    latency=latency,
                    ttft=ttft,
                    throughput=len(output_token_ids) / max(latency, 1e-9),
                )
                self._outputs[req_id] = result
                outputs.append(result)

                self.stats.total_output_tokens += len(output_token_ids)
                self.stats.total_requests += 1

        return outputs

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        sampling_params: Optional[SamplingParams] = None,
        verbose: bool = False,
    ) -> List[RequestOutput]:
        """
        批量生成接口：提交所有 prompts，运行到全部完成，返回结果。

        Args:
            prompts:         文本 prompt 列表
            max_tokens:      每条请求最多生成的 token 数
            sampling_params: 共用采样参数（None 时使用默认值 + max_tokens）
            verbose:         是否打印进度

        Returns:
            按输入顺序排列的 RequestOutput 列表
        """
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=max_tokens)
        else:
            sampling_params.max_tokens = max_tokens

        # 提交所有请求
        req_ids: List[str] = []
        for prompt in prompts:
            rid = self.add_request(prompt, sampling_params=sampling_params)
            req_ids.append(rid)

        start_time = time.monotonic()

        # 持续推理直到全部完成
        step_count = 0
        while self.scheduler.has_unfinished_seqs:
            completed = self.step()
            step_count += 1
            if verbose and completed:
                for out in completed:
                    print(f"  ✓ {out.request_id} done "
                          f"({len(out.output_token_ids)} tokens, "
                          f"latency={out.latency:.2f}s)")

        self.stats.total_time = time.monotonic() - start_time

        if verbose:
            print(f"\n{'='*50}")
            print(f"[Engine] All {len(prompts)} requests completed")
            print(f"[Engine] {self.stats}")

        # 按输入顺序返回结果
        return [self._outputs[rid] for rid in req_ids if rid in self._outputs]

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    def _decode(self, token_ids: List[int]) -> str:
        """将 token id 列表解码为文本。"""
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # MockMode：返回占位文本
        return f"[{len(token_ids)} tokens generated]"

    @property
    def kv_cache_manager(self) -> KVCacheManager:
        return self.scheduler.kv_cache
