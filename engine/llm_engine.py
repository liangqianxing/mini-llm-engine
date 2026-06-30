"""
LLM Engine：调度器 + 模型运行器的顶层集成。

集成内容：
  - Scheduler（Continuous Batching + Chunked Prefill + 优先级调度）
  - KVCacheManager（Paged KV Cache + Prefix Cache）
  - SwapManager（CPU Swap）
  - MetricsCollector（延迟 / 吞吐 / KV util 统计）
  - ModelRunner（Mock 或 GPT-2）

对外暴露简洁的 generate() 接口。

使用方式：
    engine = LLMEngine.from_config(
        num_kv_blocks=512,
        block_size=16,
        use_real_model=False,
        chunked_prefill=True,
        prefix_caching=True,
        cpu_swap_gb=4.0,
    )
    results = engine.generate(
        prompts=["Hello, my name is", "The future of AI is"],
        max_tokens=50,
    )
    engine.metrics.print_report()
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .kv_cache import KVCacheManager
from .metrics import MetricsCollector
from .model_runner import BaseModelRunner, MockModelRunner, GPT2ModelRunner
from .prefix_cache import PrefixCache
from .scheduler import Scheduler, SchedulerOutput, SchedulerPolicy
from .sequence import SamplingParams, Sequence, SequenceGroup, SequenceStatus
from .swap_manager import SwapManager


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
        short_text = (
            self.output_text[:80] + "..."
            if len(self.output_text) > 80
            else self.output_text
        )
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
    total_time: float = 0.0
    num_steps: int = 0
    peak_kv_utilization: float = 0.0

    @property
    def throughput(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_output_tokens / self.total_time

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
    LLM 推理引擎（完整版）。

    Args:
        scheduler:        调度器实例（含 Chunked Prefill / 优先级等配置）
        model_runner:     模型执行器实例（Mock 或 GPT-2）
        tokenizer:        分词器（可选，文本输入输出时使用）
        collect_metrics:  是否启用 MetricsCollector
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: BaseModelRunner,
        tokenizer=None,
        collect_metrics: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.metrics = MetricsCollector() if collect_metrics else None

        self._next_seq_id: int = 0
        self._next_req_id: int = 0
        self.stats = EngineStats()

        self._outputs: Dict[str, RequestOutput] = {}
        self._seq_to_req: Dict[int, str] = {}
        self._req_prompts: Dict[str, str] = {}
        # 跟踪每个请求的 prompt token 数和到达时间
        self._req_meta: Dict[str, Dict] = {}

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        # KV Cache
        num_kv_blocks: int = 256,
        block_size: int = 16,
        # Scheduler
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 4096,
        chunked_prefill: bool = False,
        max_prefill_tokens_per_step: int = 512,
        policy: SchedulerPolicy = SchedulerPolicy.FCFS,
        # Prefix Cache
        prefix_caching: bool = False,
        max_prefix_cached_blocks: int = 128,
        # CPU Swap
        cpu_swap_gb: float = 0.0,
        # Model
        use_real_model: bool = False,
        model_name: str = "gpt2",
        device: str = "cpu",
        # MockModelRunner 参数
        decode_time_per_step: float = 0.002,
        prefill_time_per_token: float = 0.0001,
        eos_probability: float = 0.05,
        seed: int = 42,
        # Metrics
        collect_metrics: bool = True,
    ) -> "LLMEngine":
        """从配置参数创建引擎。"""

        # 前缀缓存（可选）
        pc = None
        if prefix_caching:
            from .block_allocator import BlockAllocator
            # 先创建一个临时分配器用于 prefix cache 构建，后面共享
            _tmp_alloc = BlockAllocator(num_kv_blocks, block_size)
            pc = PrefixCache(_tmp_alloc, max_cached_blocks=max_prefix_cached_blocks)
            kv_manager = KVCacheManager(num_kv_blocks, block_size, prefix_cache=pc)
            pc.allocator = kv_manager.allocator  # 共享同一个 allocator
        else:
            kv_manager = KVCacheManager(num_kv_blocks, block_size)

        # CPU Swap Manager（可选）
        swap_manager = None
        if cpu_swap_gb > 0:
            swap_manager = SwapManager(
                gpu_allocator=kv_manager.allocator,
                cpu_memory_gb=cpu_swap_gb,
            )

        scheduler = Scheduler(
            kv_cache_manager=kv_manager,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            chunked_prefill_enabled=chunked_prefill,
            max_prefill_tokens_per_step=max_prefill_tokens_per_step,
            policy=policy,
            swap_manager=swap_manager,
        )

        # Model runner
        if use_real_model:
            runner = GPT2ModelRunner(model_name=model_name, device=device)
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except ImportError:
                tokenizer = None
        else:
            runner = MockModelRunner(
                prefill_time_per_token=prefill_time_per_token,
                decode_time_per_step=decode_time_per_step,
                eos_probability=eos_probability,
                seed=seed,
            )
            tokenizer = None

        return cls(
            scheduler=scheduler,
            model_runner=runner,
            tokenizer=tokenizer,
            collect_metrics=collect_metrics,
        )

    # ── 请求提交 ─────────────────────────────────────────────────────────────

    def add_request(
        self,
        prompt: str,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        priority: int = 0,
        deadline: Optional[float] = None,
    ) -> str:
        """
        向引擎提交一个生成请求。

        Args:
            prompt:            文本 prompt
            prompt_token_ids:  直接提供 token id（优先于 prompt 文本）
            sampling_params:   生成参数
            request_id:        自定义请求 ID
            priority:          优先级（越小越高）
            deadline:          SLO 截止时间（Unix timestamp）

        Returns:
            request_id
        """
        if request_id is None:
            request_id = f"req-{self._next_req_id}"
            self._next_req_id += 1

        if prompt_token_ids is None:
            if self.tokenizer is not None:
                prompt_token_ids = self.tokenizer.encode(prompt)
            else:
                prompt_token_ids = [ord(c) % 50257 for c in prompt[:20]]

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=prompt_token_ids,
            block_size=self.scheduler.kv_cache.block_size,
            sampling_params=sampling_params or SamplingParams(),
            priority=priority,
            deadline=deadline,
        )

        seq_group = SequenceGroup(
            request_id=request_id,
            sequences=[seq],
            priority=priority,
            deadline=deadline,
        )
        self.scheduler.add_seq_group(seq_group)
        self._seq_to_req[seq_id] = request_id
        self._req_prompts[request_id] = prompt
        self._req_meta[request_id] = {
            "prompt_len": len(prompt_token_ids),
            "arrival_time": seq.arrival_time,
        }

        return request_id

    # ── 推理主循环 ────────────────────────────────────────────────────────────

    def step(self) -> List[RequestOutput]:
        """
        执行一个推理步骤，返回本步完成的请求输出列表。
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

        # 3. 更新状态
        finished_groups = self.scheduler.on_step_done(sched_output, new_token_ids)

        # 4. 更新统计
        self.stats.num_steps += 1
        self.stats.peak_kv_utilization = max(
            self.stats.peak_kv_utilization,
            self.scheduler.kv_cache.utilization,
        )

        # 5. 记录 metrics
        if self.metrics is not None:
            self.metrics.record_step(
                step=self.stats.num_steps,
                num_waiting=self.scheduler.num_waiting,
                num_running=self.scheduler.num_running,
                num_finished=self.scheduler.num_finished,
                kv_utilization=self.scheduler.kv_cache.utilization,
                num_prefill_tokens=sum(c.chunk_len for c in sched_output.prefill_chunks),
                num_decode_tokens=len(sched_output.decode_seqs),
            )

        # 6. 构造输出
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

                if self.metrics is not None:
                    meta = self._req_meta.get(req_id, {})
                    self.metrics.record_request_done(
                        request_id=req_id,
                        prompt_len=meta.get("prompt_len", seq.prompt_len),
                        output_len=len(output_token_ids),
                        arrival_time=meta.get("arrival_time", seq.arrival_time),
                        first_token_time=seq.first_token_time,
                        finish_time=seq.finish_time,
                    )

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
        """
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=max_tokens)
        else:
            sampling_params.max_tokens = max_tokens

        req_ids: List[str] = []
        for prompt in prompts:
            rid = self.add_request(prompt, sampling_params=sampling_params)
            req_ids.append(rid)

        start_time = time.monotonic()

        while self.scheduler.has_unfinished_seqs:
            completed = self.step()
            if verbose and completed:
                for out in completed:
                    print(
                        f"  ✓ {out.request_id} done "
                        f"({len(out.output_token_ids)} tokens, "
                        f"latency={out.latency:.2f}s, "
                        f"ttft={out.ttft:.3f}s)"
                    )

        self.stats.total_time = time.monotonic() - start_time

        if verbose:
            print(f"\n{'='*50}")
            print(f"[Engine] All {len(prompts)} requests completed")
            print(self.stats)

        return [self._outputs[rid] for rid in req_ids if rid in self._outputs]

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    def _decode(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return f"[{len(token_ids)} tokens generated]"

    @property
    def kv_cache_manager(self) -> KVCacheManager:
        return self.scheduler.kv_cache
