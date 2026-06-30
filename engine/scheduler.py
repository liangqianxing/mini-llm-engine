"""
Continuous Batching 调度器。

核心思路（来自 vLLM / Orca 论文）：
  - 传统静态批处理：等一批请求全部完成再处理下一批 → GPU 利用率低。
  - Continuous Batching：每个 decode 步后立即检查是否有请求完成，
    若有则从等待队列中拉入新请求，填满 GPU 算力 → 吞吐量大幅提升。

调度器职责：
  1. 维护三个队列：waiting / running / finished
  2. 每步调用 schedule() 返回本步要运行的序列列表
  3. 通过 KVCacheManager 做内存合法性检查（避免 OOM）
  4. 支持简单的抢占（preemption）：内存不足时暂停低优先级序列

Ref: "Efficient Memory Management for Large Language Model Serving
       with PagedAttention", Kwon et al., SOSP 2023
"""

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import time

from .kv_cache import KVCacheManager
from .sequence import Sequence, SequenceGroup, SequenceStatus


@dataclass
class SchedulerOutput:
    """
    调度器每步的输出，传给 ModelRunner。

    Attributes:
        prefill_seqs:   本步需要做 prefill 的序列（新请求，处理完整 prompt）
        decode_seqs:    本步需要做 decode 的序列（已有 KV cache，生成下一 token）
        blocks_to_swap_in:   被换入的序列（抢占恢复，暂未实现）
        blocks_to_swap_out:  被换出的序列（抢占，暂未实现）
    """
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)
    blocks_to_swap_in: List[Sequence] = field(default_factory=list)
    blocks_to_swap_out: List[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def num_batched_tokens(self) -> int:
        """本步处理的总 token 数（用于计算吞吐）。"""
        prefill_tokens = sum(s.num_total_tokens for s in self.prefill_seqs)
        decode_tokens = len(self.decode_seqs)  # 每条 decode 序列生成 1 token
        return prefill_tokens + decode_tokens


class Scheduler:
    """
    Continuous Batching 调度器。

    Args:
        kv_cache_manager:  负责物理块分配的 KVCacheManager
        max_num_seqs:      同时运行的最大序列数（类比 GPU SM 数限制）
        max_num_batched_tokens: 每步最多处理的 token 总数（显存/带宽约束）
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
    ) -> None:
        self.kv_cache = kv_cache_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        # 三条核心队列
        self.waiting:  Deque[SequenceGroup] = deque()   # 等待 prefill
        self.running:  List[SequenceGroup]  = []        # 正在 decode
        self.finished: List[SequenceGroup]  = []        # 已结束

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
          Step 1. 先处理 running 队列：
                  - 为每条 decode 序列追加槽位（可能需要新物理块）
                  - 若内存不足，抢占优先级最低的序列（放回 waiting）

          Step 2. 从 waiting 队列拉入新请求（prefill）：
                  - 检查内存和并发数上限
                  - 批量 token 上限（max_num_batched_tokens）

          Step 3. 返回 SchedulerOutput
        """
        output = SchedulerOutput()
        self.num_steps += 1
        self._step_timestamps.append(time.monotonic())

        # ── Step 1: 处理 running 中的 decode 序列 ──────────────────────────
        decode_groups: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []

        for seq_group in self.running:
            seqs = seq_group.get_seqs(SequenceStatus.RUNNING)
            can_run = True

            for seq in seqs:
                if not self.kv_cache.can_append_slot(seq):
                    # 内存不足，抢占该 seq_group（最简单策略：FIFO 抢占末尾）
                    can_run = False
                    break

            if can_run:
                # 为每条序列追加槽（可能新分配物理块）
                for seq in seqs:
                    self.kv_cache.append_slot(seq)
                decode_groups.append(seq_group)
                output.decode_seqs.extend(seqs)
            else:
                # 抢占：释放物理块，序列状态回退到 WAITING
                for seq in seqs:
                    self.kv_cache.free(seq)
                    seq.status = SequenceStatus.PREEMPTED
                preempted.append(seq_group)

        # 更新 running 队列（排除被抢占的）
        self.running = decode_groups

        # 被抢占的放回等待队列头部（优先恢复）
        for seq_group in reversed(preempted):
            self.waiting.appendleft(seq_group)
            for seq in seq_group.get_seqs(SequenceStatus.PREEMPTED):
                seq.status = SequenceStatus.WAITING

        # ── Step 2: 从 waiting 拉入新请求（prefill）────────────────────────
        num_curr_seqs = sum(g.num_seqs for g in self.running)
        batched_tokens = output.num_batched_tokens  # 已有 decode tokens

        while self.waiting:
            seq_group = self.waiting[0]
            seqs = seq_group.get_seqs(SequenceStatus.WAITING)

            # 并发数限制
            if num_curr_seqs + len(seqs) > self.max_num_seqs:
                break

            # prefill token 数限制
            prompt_len = sum(s.num_total_tokens for s in seqs)
            if batched_tokens + prompt_len > self.max_num_batched_tokens:
                # 如果等待队列头部本身就超限（单条超长 prompt），允许特例
                if not output.prefill_seqs and not output.decode_seqs:
                    pass  # 单条超长 prompt：允许通过
                else:
                    break

            # 内存检查：所有序列都能分配物理块？
            can_alloc = all(
                self.kv_cache.can_allocate(seq) for seq in seqs
            )
            if not can_alloc:
                break

            # 通过所有检查：分配物理块，切换状态，加入本轮 prefill
            self.waiting.popleft()
            for seq in seqs:
                self.kv_cache.allocate(seq)
                seq.status = SequenceStatus.RUNNING

            self.running.append(seq_group)
            output.prefill_seqs.extend(seqs)
            num_curr_seqs += len(seqs)
            batched_tokens += prompt_len

        return output

    def on_step_done(self, output: SchedulerOutput, new_token_ids: Dict[int, int]) -> List[SequenceGroup]:
        """
        模型推理完成后，调用此方法更新序列状态。

        Args:
            output:        本步的调度输出
            new_token_ids: seq_id → 新生成 token_id 的映射

        Returns:
            本步完成的 SequenceGroup 列表（供调用方收集结果）
        """
        finished_groups: List[SequenceGroup] = []

        all_seqs = output.prefill_seqs + output.decode_seqs
        for seq in all_seqs:
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
                # 释放所有物理块
                for seq in seq_group.seqs:
                    self.kv_cache.free(seq)
                self.finished.append(seq_group)
                finished_groups.append(seq_group)
            else:
                still_running.append(seq_group)

        self.running = still_running
        return finished_groups

    # ── 状态查询 ─────────────────────────────────────────────────────────────

    @property
    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_finished(self) -> int:
        return len(self.finished)

    def get_stats(self) -> Dict:
        return {
            "num_steps": self.num_steps,
            "num_waiting": self.num_waiting,
            "num_running": self.num_running,
            "num_finished": self.num_finished,
            "kv_utilization": self.kv_cache.utilization,
        }

    def __repr__(self) -> str:
        return (
            f"Scheduler(waiting={self.num_waiting}, "
            f"running={self.num_running}, "
            f"finished={self.num_finished})"
        )
