"""
物理 Block 分配器。

维护一个空闲物理块池（free list），提供 O(1) 的分配与释放。
类比操作系统的帧分配器（frame allocator）。
"""

from collections import deque
from typing import Deque, List

from .block import PhysicalBlock


class BlockAllocator:
    """
    GPU KV Cache 物理块分配器。

    初始化时预分配 num_blocks 个 PhysicalBlock，放入空闲队列。
    调度器通过 allocate / free 管理这些块，无需提前知道序列的最终长度。

    Args:
        num_blocks: 物理块总数（决定 GPU 显存上限）
        block_size: 每块容纳的 token 数（例如 16）
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size

        # 所有物理块对象（固定，只创建一次）
        self._blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i) for i in range(num_blocks)
        ]
        # 空闲队列（双端队列，O(1) popleft）
        self._free_blocks: Deque[PhysicalBlock] = deque(self._blocks)

    # ── 核心操作 ──────────────────────────────────────────────────────────────

    def allocate(self) -> PhysicalBlock:
        """
        分配一个空闲物理块。

        Returns:
            一个 ref_count=1 的 PhysicalBlock。
        Raises:
            MemoryError: 空闲块耗尽时。
        """
        if not self._free_blocks:
            raise MemoryError(
                f"KV Cache OOM: no free blocks "
                f"(total={self.num_blocks}, block_size={self.block_size})"
            )
        block = self._free_blocks.popleft()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        """
        释放一个物理块，引用计数归零后归还空闲池。

        Args:
            block: 要释放的物理块。
        """
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_blocks.append(block)

    def free_blocks(self, blocks: List[PhysicalBlock]) -> None:
        """批量释放物理块。"""
        for block in blocks:
            self.free(block)

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        """块使用率 [0.0, 1.0]。"""
        return self.num_used_blocks / self.num_blocks

    def can_allocate(self, num_blocks_needed: int = 1) -> bool:
        """检查是否有足够的空闲块。"""
        return self.num_free_blocks >= num_blocks_needed

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(total={self.num_blocks}, "
            f"free={self.num_free_blocks}, "
            f"used={self.num_used_blocks}, "
            f"block_size={self.block_size})"
        )
