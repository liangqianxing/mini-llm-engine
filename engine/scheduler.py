"""
Continuous Batching 调度器（完整版）。

在原有 Continuous Batching 基础上，新增：
  1. Chunked Prefill   ── 长 prompt 分块处理，降低 TTFT 影响
  2. 优先级调度         ── 按 priority / deadline（EDF）决定抢占顺序
  3. CPU Swap          ── 抢占时将 KV cache 转移到 CPU，而非丢弃
  4. 调度策略枚举       ── FCFS / PRIORITY / EDF

Ref:
  - vLLM Scheduler: github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
  - Orca (OSDI'22): Continuous batching
  - Sarathi (OSDI'24): Chunked-prefill for stall-free batching
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple
import time

from .kv_cache import KVCacheManager
from .sequence import Sequence, SequenceGroup, SequenceStatus
from .swap_manager import SwapManager


class SchedulerPolicy(Enum):
    """调度 / 抢占策略。"""
    FCFS     = auto()   # First-Come-First-Served（按到达顺序）
    PRIORITY = auto()   # 按 sequence.priority 数值（小值优先）
    EDF      = auto()   # Earliest Deadline First（按 deadline 时间戳）


@dataclass
class PrefillChunk:
    """
    Chunked Prefill 中，本步要处理的一段 prompt token。

    Attributes:
        seq:         要处理的序列
        token_start: prompt 中 chunk 的起始 token 索引
        token_end:   prompt 中 chunk 的结束 token 索引（exclusive）
    """
    seq: Sequence
    token_start: int
    token_end: int

    @property
    def chunk_len(self) -> int:
        return self.token_end - self.token_start


@dataclass
class SchedulerOutput:
    """
    调度器每步的输出，传给 ModelRunner。

    Attributes:
        prefill_chunks:  本步需要 prefill 的 chunk 列表（Chunked Prefill）
        decode_seqs:     本步需要 decode 的序列（已有 KV cache，生成下一 token）
        swap_in_seqs:    本步从 CPU 换入 GPU 的序列
        swap_out_seqs:   本步从 GPU 换出到 CPU 的序列
    """
    prefill_chunks: List[PrefillChunk] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)
    swap_in_seqs: List[Sequence] = field(default_factory=list)
    swap_out_seqs: List[Sequence] = field(default_factory=list)

    @property
    def prefill_seqs(self) -> List[Sequence]:
        """向后兼容：返回所有 prefill 序列。"""
        return [c.seq for c in self.prefill_chunks]

    @property
    def is_empty(self) -> bool:
        return not self.prefill_chunks and not self.decode_seqs

    @property
    def num_batched_tokens(self) -> int:
        """本步处理的总 token 数（用于计算吞吐）。"""
        prefill_tokens = sum(c.chunk_len for c in self.prefill_chunks)
        return prefill_tokens + len(self.decode_seqs)


class Scheduler:
    """
    Continuous Batching 调度器（带 Chunked Prefill + 优先级调度 + CPU Swap）。

    Args:
        kv_cache_manager:           KV Cache 管理器
        max_num_seqs:               最大并发序列数
        max_num_batched_tokens:     每步最多处理的 token 总数
        chunked_prefill_enabled:    是否启用 Chunked Prefill
        max_prefill_tokens_per_step: Chunked Prefill 每步最大 token 数
        policy:                     调度策略（FCFS / PRIORITY / EDF）
        swap_manager:               CPU Swap Manager（None 表示禁用 swap）
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        chunked_prefill_enabled: bool = False,
        max_prefill_tokens_per_step: int = 512,
        policy: SchedulerPolicy = SchedulerPolicy.FCFS,
        swap_manager: Optional[SwapManager] = None,
    ) -> None:
        self.kv_cache = kv_cache_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.max_prefill_tokens_per_step = max_prefill_tokens_per_step
        self.policy = policy
        self.swap_manager = swap_manager

        # 核心队列
        self.waiting:  Deque[SequenceGroup] = deque()   # 未分配 KV cache
        self.running:  List[SequenceGroup]  = []        # 持有 GPU 块（PREFILLING/RUNNING）
        self.swapped:  List[SequenceGroup]  = []        # KV cache 在 CPU（SWAPPED）
        self.finished: List[SequenceGroup]  = []        # 已完成

        # 统计
        self.num_steps: int = 0
        self._step_timestamps: List[float] = []

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """将新请求加入等待队列。"""
        self.waiting.append(seq_group)

    def schedule(self) -> SchedulerOutput:
        """
        执行一步调度，返回本步要运行的序列集合。

        调度流程（每步）：
          Step 1. 处理 running 中的 PREFILLING 序列（推进 chunk 进度）
          Step 2. 处理 running 中的 RUNNING 序列（decode + 检查内存）
          Step 3. 尝试将 swapped 序列换回 GPU（swap in）
          Step 4. 从 waiting 拉入新请求（prefill 第一个 chunk）
        """
        output = SchedulerOutput()
        self.num_steps += 1
        self._step_timestamps.append(time.monotonic())

        # ── Step 1 & 2: 处理 running 队列 ──────────────────────────────────
        still_running: List[SequenceGroup] = []

        for seq_group in self.running:
            prefilling = seq_group.get_seqs(SequenceStatus.PREFILLING)
            decoding   = seq_group.get_seqs(SequenceStatus.RUNNING)

            group_ok = True

            # 处理 PREFILLING 序列（Chunked Prefill）
            for seq in prefilling:
                chunk = self._get_next_prefill_chunk(seq)
                if chunk is None:
                    # 无需更多 chunk（prompt 已全部处理），转为 RUNNING
                    seq.status = SequenceStatus.RUNNING
                    # decode slot 检查
                    if not self.kv_cache.can_append_slot(seq):
                        group_ok = False; break
                    self.kv_cache.append_slot(seq)
                    output.decode_seqs.append(seq)
                else:
                    output.prefill_chunks.append(chunk)

            if not group_ok:
                self._preempt(seq_group, output)
                continue

            # 处理 RUNNING 序列（decode）
            for seq in decoding:
                if not self.kv_cache.can_append_slot(seq):
                    group_ok = False; break
                self.kv_cache.append_slot(seq)
                output.decode_seqs.append(seq)

            if not group_ok:
                self._preempt(seq_group, output)
                continue

            still_running.append(seq_group)

        self.running = still_running

        # ── Step 3: 尝试 swap in ─────────────────────────────────────────
        if self.swap_manager is not None:
            newly_swapped_in: List[SequenceGroup] = []
            remaining_swapped: List[SequenceGroup] = []

            for seq_group in self.swapped:
                seqs = seq_group.get_seqs(SequenceStatus.SWAPPED)
                can_swap_in_all = all(self.swap_manager.can_swap_in(s) for s in seqs)
                num_slots_ok = (
                    sum(g.num_seqs for g in self.running)
                    + len(newly_swapped_in)
                    + seq_group.num_seqs
                ) <= self.max_num_seqs

                if can_swap_in_all and num_slots_ok:
                    for seq in seqs:
                        self.swap_manager.swap_in(seq)
                    newly_swapped_in.append(seq_group)
                    self.running.append(seq_group)
                    output.swap_in_seqs.extend(seqs)
                else:
                    remaining_swapped.append(seq_group)

            self.swapped = remaining_swapped

        # ── Step 4: 从 waiting 拉入新请求 ───────────────────────────────
        num_curr_seqs = sum(g.num_seqs for g in self.running)
        batched_tokens = output.num_batched_tokens

        while self.waiting:
            seq_group = self.waiting[0]
            seqs = seq_group.get_seqs(SequenceStatus.WAITING)

            # 并发数限制
            if num_curr_seqs + len(seqs) > self.max_num_seqs:
                break

            # token 预算限制
            if self.chunked_prefill_enabled:
                chunk_budget = self.max_prefill_tokens_per_step - sum(
                    c.chunk_len for c in output.prefill_chunks
                )
                if chunk_budget <= 0 and output.decode_seqs:
                    break
                prompt_chunk_len = min(seqs[0].prompt_len, max(chunk_budget, 1))
            else:
                prompt_len = sum(s.num_total_tokens for s in seqs)
                if batched_tokens + prompt_len > self.max_num_batched_tokens:
                    if not output.is_empty:
                        break
                prompt_chunk_len = seqs[0].prompt_len

            # 内存检查
            can_alloc = all(self.kv_cache.can_allocate(s) for s in seqs)
            if not can_alloc:
                break

            # 通过检查：分配物理块，设置初始状态
            self.waiting.popleft()

            for seq in seqs:
                self.kv_cache.allocate(seq)
                if self.chunked_prefill_enabled and seq.prompt_len > prompt_chunk_len:
                    seq.status = SequenceStatus.PREFILLING
                    # 生成第一个 chunk
                    chunk = PrefillChunk(
                        seq=seq,
                        token_start=0,
                        token_end=prompt_chunk_len,
                    )
                    seq.advance_prefill(prompt_chunk_len)
                    output.prefill_chunks.append(chunk)
                else:
                    seq.status = SequenceStatus.RUNNING
                    seq.num_prefilled_tokens = seq.prompt_len
                    chunk = PrefillChunk(
                        seq=seq,
                        token_start=0,
                        token_end=seq.prompt_len,
                    )
                    output.prefill_chunks.append(chunk)

            self.running.append(seq_group)
            num_curr_seqs += len(seqs)
            batched_tokens += prompt_chunk_len

        return output

    def on_step_done(
        self,
        output: SchedulerOutput,
        new_token_ids: Dict[int, int],
    ) -> List[SequenceGroup]:
        """
        模型推理完成后，更新序列状态。

        Args:
            output:        本步调度输出
            new_token_ids: seq_id → 新生成 token_id（仅 decode + 完成 prefill 的序列）

        Returns:
            本步完成的 SequenceGroup 列表
        """
        finished_groups: List[SequenceGroup] = []

        # 对 prefill 完成（整块 prefill，非 chunked）和 decode 的序列追加 token
        for seq in output.decode_seqs + [
            c.seq for c in output.prefill_chunks
            if c.seq.is_prompt_fully_prefilled
        ]:
            new_token = new_token_ids.get(seq.seq_id)
            if new_token is None:
                continue
            seq.append_token(new_token)
            if seq.should_stop():
                seq.mark_finished()

        # 检查哪些 seq_group 全部完成
        still_running: List[SequenceGroup] = []
        for seq_group in self.running:
            if seq_group.is_finished:
                for seq in seq_group.seqs:
                    self.kv_cache.free(seq)
                self.finished.append(seq_group)
                finished_groups.append(seq_group)
            else:
                still_running.append(seq_group)
        self.running = still_running

        return finished_groups

    # ── 抢占 ─────────────────────────────────────────────────────────────────

    def _preempt(self, seq_group: SequenceGroup, output: SchedulerOutput) -> None:
        """
        抢占一个 seq_group：
          1. 若 swap_manager 可用，尝试 swap out（保留 KV cache 到 CPU）
          2. 否则直接释放 GPU 块（放回 waiting，需要重新 prefill）
        """
        seqs = [s for s in seq_group.seqs if not s.is_finished]

        if self.swap_manager is not None and all(
            self.swap_manager.can_swap_out(s) for s in seqs
        ):
            # Swap out：KV cache 转移到 CPU
            for seq in seqs:
                self.swap_manager.swap_out(seq)
            output.swap_out_seqs.extend(seqs)
            self.swapped.append(seq_group)
        else:
            # Recompute：丢弃 KV cache，放回 waiting 重新 prefill
            for seq in seqs:
                self.kv_cache.free(seq)
                seq.status = SequenceStatus.WAITING
                seq.num_prefilled_tokens = 0  # 重置 chunked prefill 进度
            self.waiting.appendleft(seq_group)

    def _choose_victim(self) -> Optional[SequenceGroup]:
        """
        选择要被抢占的 seq_group（优先级最低 / urgency 最大的那个）。

        策略：
          FCFS     → 抢占最近加入 running 的序列（列表末尾）
          PRIORITY → 抢占 min_urgency 最大的序列
          EDF      → 与 PRIORITY 相同（urgency = deadline）
        """
        if not self.running:
            return None
        if self.policy == SchedulerPolicy.FCFS:
            return self.running[-1]
        # PRIORITY / EDF：选 urgency 最大（最不紧急）的
        return max(self.running, key=lambda g: g.min_urgency())

    # ── Chunked Prefill 辅助 ──────────────────────────────────────────────────

    def _get_next_prefill_chunk(self, seq: Sequence) -> Optional[PrefillChunk]:
        """
        获取序列的下一个 prefill chunk。
        若 prompt 已全部处理，返回 None。
        """
        if seq.is_prompt_fully_prefilled:
            return None
        chunk_size = self.max_prefill_tokens_per_step
        start, end = seq.get_next_prefill_range(chunk_size)
        seq.advance_prefill(end - start)
        return PrefillChunk(seq=seq, token_start=start, token_end=end)

    # ── 状态查询 ─────────────────────────────────────────────────────────────

    @property
    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting) or bool(self.running) or bool(self.swapped)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_swapped(self) -> int:
        return len(self.swapped)

    @property
    def num_finished(self) -> int:
        return len(self.finished)

    def get_stats(self) -> Dict:
        stats = {
            "num_steps": self.num_steps,
            "num_waiting": self.num_waiting,
            "num_running": self.num_running,
            "num_swapped": self.num_swapped,
            "num_finished": self.num_finished,
            "kv_utilization": self.kv_cache.utilization,
            "policy": self.policy.name,
            "chunked_prefill": self.chunked_prefill_enabled,
        }
        if self.swap_manager is not None:
            stats["swap"] = self.swap_manager.stats()
        return stats

    def __repr__(self) -> str:
        return (
            f"Scheduler("
            f"waiting={self.num_waiting}, "
            f"running={self.num_running}, "
            f"swapped={self.num_swapped}, "
            f"finished={self.num_finished}, "
            f"policy={self.policy.name})"
        )
