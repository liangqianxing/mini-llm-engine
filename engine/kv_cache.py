"""
KV Cache 管理器。

负责：
  1. 为序列的逻辑块分配 / 释放物理块
  2. 维护 block_table（逻辑块 → 物理块 映射）
  3. 在调度前检查是否有足够空闲块

与 BlockAllocator 的关系：
  KVCacheManager 是"策略层"（知道序列和块的关系），
  BlockAllocator 是"机制层"（只管分配/回收物理块）。
"""

from typing import Dict, List

from .block import PhysicalBlock
from .block_allocator import BlockAllocator
from .sequence import Sequence


class KVCacheManager:
    """
    管理所有序列的 KV Cache 分配。

    Args:
        num_blocks: 物理块总数
        block_size: 每块 token 数
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks, block_size)

        # 记录每个序列持有的物理块列表，方便批量释放
        # seq_id → List[PhysicalBlock]
        self._seq_blocks: Dict[int, List[PhysicalBlock]] = {}

    # ── 分配 ─────────────────────────────────────────────────────────────────

    def can_allocate(self, seq: Sequence) -> bool:
        """
        判断是否能为该序列的所有逻辑块分配物理块。

        用于调度器在决定接收新请求前的检查。
        """
        return self.allocator.can_allocate(seq.num_logical_blocks)

    def allocate(self, seq: Sequence) -> None:
        """
        为序列的每个逻辑块分配一个物理块，建立 block_table。

        调用时机：序列从 WAITING → RUNNING（prefill 阶段开始前）。
        """
        if seq.seq_id in self._seq_blocks:
            raise ValueError(f"Seq {seq.seq_id} already has blocks allocated")

        blocks: List[PhysicalBlock] = []
        for logical_idx in range(seq.num_logical_blocks):
            phys_block = self.allocator.allocate()
            seq.block_table[logical_idx] = phys_block
            blocks.append(phys_block)

        self._seq_blocks[seq.seq_id] = blocks

    def can_append_slot(self, seq: Sequence) -> bool:
        """
        判断是否能为序列追加下一个 token 的槽位。

        如果最后一个逻辑块已满，需要新物理块；否则直接写入当前块。
        """
        if seq.needs_new_block():
            return self.allocator.can_allocate(1)
        return True  # 当前块还有空位，无需新块

    def append_slot(self, seq: Sequence) -> bool:
        """
        为序列的下一个 token 预留槽位。

        如果最后一个逻辑块已满，分配新物理块并更新 block_table。

        Returns:
            True  表示分配了新物理块
            False 表示复用了当前块（无需新分配）

        调用时机：decode 阶段，每次生成新 token 前。
        """
        if not seq.needs_new_block():
            return False  # 当前块仍有空位

        # 分配新物理块
        new_block = self.allocator.allocate()
        new_logical_idx = seq.num_logical_blocks  # 即将新增的逻辑块编号
        seq.block_table[new_logical_idx] = new_block
        self._seq_blocks[seq.seq_id].append(new_block)
        return True

    # ── 释放 ─────────────────────────────────────────────────────────────────

    def free(self, seq: Sequence) -> None:
        """
        释放序列持有的所有物理块。

        调用时机：序列结束（FINISHED）或被抢占（PREEMPTED）。
        """
        if seq.seq_id not in self._seq_blocks:
            return
        self.allocator.free_blocks(self._seq_blocks.pop(seq.seq_id))
        seq.block_table.clear()

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    @property
    def num_used_blocks(self) -> int:
        return self.allocator.num_used_blocks

    @property
    def utilization(self) -> float:
        return self.allocator.utilization

    def get_block_table(self, seq: Sequence) -> List[int]:
        """
        返回序列的物理块 ID 列表（按逻辑顺序）。

        这个列表传给模型的 attention 计算，告诉它 KV cache 存在哪些物理块里。
        """
        return [
            seq.block_table[i].block_id
            for i in range(len(seq.block_table))
        ]

    def __repr__(self) -> str:
        return (
            f"KVCacheManager("
            f"free={self.num_free_blocks}, "
            f"used={self.num_used_blocks}, "
            f"util={self.utilization:.1%})"
        )
