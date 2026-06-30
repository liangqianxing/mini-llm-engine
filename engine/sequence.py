"""
序列（Sequence）和请求组（SequenceGroup）的定义。

Sequence      ── 对应一个生成中的文本流（prompt → generated tokens）
SequenceGroup ── 对应一个用户请求（目前每个请求只含一条序列；
                  beam search 时会含多条）

新增功能：
  - PREFILLING 状态：Chunked Prefill 时序列处于分批 prefill 阶段
  - SWAPPED 状态：被 CPU Swap 抢占（GPU 块已释放，数据在 CPU RAM）
  - priority / deadline：优先级调度 + SLO 感知抢占
  - num_prefilled_tokens：跟踪 Chunked Prefill 的进度
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from .block import LogicalTokenBlock, PhysicalBlock


class SequenceStatus(Enum):
    """序列的生命周期状态。"""
    WAITING    = auto()   # 在等待队列中，尚未分配 KV cache
    PREFILLING = auto()   # Chunked Prefill：正在分批处理 prompt（持有 GPU 块）
    RUNNING    = auto()   # Prefill 完成，正在 decode（持有 GPU 块）
    PREEMPTED  = auto()   # 被抢占，GPU 块已回收，等待重新调度
    SWAPPED    = auto()   # 被 CPU Swap 抢占，KV cache 已转移到 CPU RAM
    FINISHED   = auto()   # 生成结束（遇到 EOS 或达到 max_tokens）


@dataclass
class SamplingParams:
    """生成超参数。"""
    max_tokens: int = 256         # 最多生成的新 token 数
    temperature: float = 1.0      # 采样温度（0 = greedy）
    top_p: float = 1.0            # nucleus sampling
    eos_token_id: int = 50256     # GPT-2 的 <|endoftext|>


class Sequence:
    """
    单条生成序列。

    维护：
      - 已生成的 token id 列表（prompt + output）
      - 逻辑块列表（logical blocks）
      - 到物理块的映射表（block_table: logical_idx → PhysicalBlock）
      - 调度状态、优先级、SLO deadline
      - Chunked Prefill 进度（num_prefilled_tokens）

    Args:
        seq_id:           唯一序列 ID
        prompt_token_ids: Prompt 的 token id 列表
        block_size:       每个 KV Cache 块的容量
        sampling_params:  采样超参数
        priority:         优先级（越小优先级越高，默认 0）
        deadline:         SLO 截止时间（Unix timestamp，None 表示无 SLO）
    """

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        block_size: int,
        sampling_params: Optional[SamplingParams] = None,
        priority: int = 0,
        deadline: Optional[float] = None,
    ) -> None:
        self.seq_id = seq_id
        self.block_size = block_size
        self.sampling_params = sampling_params or SamplingParams()
        self.priority = priority
        self.deadline = deadline

        # token id 历史（prompt + 已生成）
        self.token_ids: List[int] = list(prompt_token_ids)
        self.prompt_len: int = len(prompt_token_ids)

        # Chunked Prefill 进度：已完成 prefill 的 token 数（<= prompt_len）
        self.num_prefilled_tokens: int = 0

        # 逻辑块（随 token 增长而新增块）
        self.logical_blocks: List[LogicalTokenBlock] = []
        self._init_logical_blocks(prompt_token_ids)

        # 物理块映射表：logical block index → PhysicalBlock
        self.block_table: Dict[int, PhysicalBlock] = {}

        # 状态与统计
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.arrival_time: float = time.monotonic()
        self.first_token_time: Optional[float] = None   # Time-to-first-token
        self.finish_time: Optional[float] = None

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def _init_logical_blocks(self, token_ids: List[int]) -> None:
        """将 prompt token 填入逻辑块。"""
        for token_id in token_ids:
            self._append_token_to_logical_blocks(token_id)

    def _append_token_to_logical_blocks(self, token_id: int) -> None:
        """在逻辑块列表末尾追加一个 token，必要时新建块。"""
        if not self.logical_blocks or self.logical_blocks[-1].is_full:
            self.logical_blocks.append(
                LogicalTokenBlock(
                    block_number=len(self.logical_blocks),
                    block_size=self.block_size,
                )
            )
        self.logical_blocks[-1].append_token(token_id)

    # ── Chunked Prefill ───────────────────────────────────────────────────────

    @property
    def is_prompt_fully_prefilled(self) -> bool:
        """Prompt 是否已全部 prefill 完成。"""
        return self.num_prefilled_tokens >= self.prompt_len

    def get_next_prefill_range(self, chunk_size: int):
        """
        返回下一个 prefill chunk 的 (start, end) token 范围。

        Args:
            chunk_size: 每步最多处理的 token 数
        Returns:
            (start_pos, end_pos) in [0, prompt_len]
        """
        start = self.num_prefilled_tokens
        end = min(start + chunk_size, self.prompt_len)
        return start, end

    def advance_prefill(self, num_tokens: int) -> None:
        """推进 prefill 进度。"""
        self.num_prefilled_tokens = min(
            self.num_prefilled_tokens + num_tokens, self.prompt_len
        )
        if self.is_prompt_fully_prefilled and self.first_token_time is None:
            # Prefill 完成时开始计时（decode 还未开始）
            pass

    # ── 生成步骤 ─────────────────────────────────────────────────────────────

    def append_token(self, token_id: int) -> None:
        """追加一个新生成的 token（decode 阶段每步调用一次）。"""
        self.token_ids.append(token_id)
        self._append_token_to_logical_blocks(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.monotonic()

    def needs_new_block(self) -> bool:
        """当前最后一个逻辑块是否已满（即下一步需要新块）？"""
        return not self.logical_blocks or self.logical_blocks[-1].is_full

    # ── 状态与属性 ───────────────────────────────────────────────────────────

    @property
    def num_logical_blocks(self) -> int:
        return len(self.logical_blocks)

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_len

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def is_eos(self) -> bool:
        """最后一个 token 是否为 EOS。"""
        return (
            bool(self.token_ids)
            and self.token_ids[-1] == self.sampling_params.eos_token_id
        )

    def should_stop(self) -> bool:
        """是否应该停止生成。"""
        return (
            self.is_eos()
            or self.num_output_tokens >= self.sampling_params.max_tokens
        )

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED
        self.finish_time = time.monotonic()

    # ── 优先级比较（用于 heapq）──────────────────────────────────────────────

    def urgency(self) -> float:
        """
        用于抢占决策：数值越小越紧急（越不应该被抢占）。

        EDF（Earliest Deadline First）模式：urgency = deadline
        PRIORITY 模式：urgency = priority（越小越紧急）
        无 deadline：urgency = priority
        """
        if self.deadline is not None:
            return self.deadline
        return float(self.priority)

    # ── 性能统计 ─────────────────────────────────────────────────────────────

    @property
    def latency(self) -> Optional[float]:
        """端到端延迟（秒）。"""
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def ttft(self) -> Optional[float]:
        """Time-to-first-token（秒）。"""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def __repr__(self) -> str:
        return (
            f"Seq(id={self.seq_id}, "
            f"status={self.status.name}, "
            f"tokens={self.num_total_tokens}, "
            f"prio={self.priority})"
        )


@dataclass
class SequenceGroup:
    """
    一个用户请求，当前包含一条序列（未来可扩展为 beam search 的多条）。

    Args:
        request_id:   请求 ID
        sequences:    序列列表（通常只有 1 条）
        arrival_time: 到达时间戳
        priority:     请求优先级（越小越高，传递给所有子序列）
        deadline:     SLO 截止时间（Unix timestamp）
    """
    request_id: str
    sequences: List[Sequence]
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0
    deadline: Optional[float] = None

    @property
    def seqs(self) -> List[Sequence]:
        return self.sequences

    @property
    def is_finished(self) -> bool:
        return all(s.is_finished for s in self.sequences)

    @property
    def num_seqs(self) -> int:
        return len(self.sequences)

    def get_seqs(
        self, status: Optional[SequenceStatus] = None
    ) -> List[Sequence]:
        if status is None:
            return self.sequences
        return [s for s in self.sequences if s.status == status]

    def min_urgency(self) -> float:
        """最紧急的子序列的 urgency（用于抢占比较）。"""
        seqs = [s for s in self.sequences if not s.is_finished]
        if not seqs:
            return float("inf")
        return min(s.urgency() for s in seqs)

    def __repr__(self) -> str:
        return f"SeqGroup(req={self.request_id}, seqs={self.sequences})"
