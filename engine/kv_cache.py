"""
KV Cache 管理器。

负责：
  1. 为序列的逻辑块分配 / 释放物理块
  2. 维护 block_table（逻辑块 → 物理块 映射）
  3. 在调度前检查是否有足够空闲块
  4. （可选）与 PrefixCache 集成，实现 Copy-on-Write 前缀复用

与 BlockAllocator 的关系：
  KVCacheManager 是"策略层"（知道序列和块的关系），
  BlockAllocator 是"机制层"（只管分配/回收物理块）。
"""

from typing import Dict, List, Optional, Tuple

from .block import PhysicalBlock
from .block_allocator import BlockAllocator
from .sequence import Sequence


class KVCacheManager:
    """
    管理所有序列的 KV Cache 分配。

    Args:
        num_blocks:   物理块总数
        block_size:   每块 token 数
        prefix_cache: （可选）PrefixCache 实例，启用前缀复用
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        prefix_cache=None,  # Optional[PrefixCache]
    ) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.prefix_cache = prefix_cache

        # seq_id → List[PhysicalBlock]（持有的物理块列表，方便批量释放）
        self._seq_blocks: Dict[int, List[PhysicalBlock]] = {}

    # ── 分配 ─────────────────────────────────────────────────────────────────

    def can_allocate(self, seq: Sequence) -> bool:
        """
        判断是否能为该序列的所有逻辑块分配物理块（考虑 prefix cache 命中）。
        """
        if self.prefix_cache is None:
            return self.allocator.can_allocate(seq.num_logical_blocks)

        # 有 prefix cache：统计需要新分配的块数
        num_new_blocks = self._count_new_blocks_needed(seq)
        return self.allocator.can_allocate(num_new_blocks)

    def _count_new_blocks_needed(self, seq: Sequence) -> int:
        """计算需要从 allocator 新分配的块数（前缀 cache 命中的可以复用）。"""
        if self.prefix_cache is None:
            return seq.num_logical_blocks
        count = 0
        for logical_block in seq.logical_blocks:
            if logical_block.is_full and logical_block.content_hash is not None:
                if self.prefix_cache.lookup.__module__:
                    # 查 cache（不消耗，只检查存在性）
                    h = logical_block.content_hash
                    if h not in self.prefix_cache._cache:
                        count += 1
                    # 命中则 ref_count 会在 allocate 时+1，不需要新块
            else:
                count += 1
        return count

    def allocate(self, seq: Sequence) -> None:
        """
        为序列的每个逻辑块分配物理块，建立 block_table。

        调用时机：序列从 WAITING → PREFILLING/RUNNING（prefill 开始前）。

        若启用 PrefixCache：
          - 满块（is_full=True）先查缓存，命中则共享物理块（CoW）
          - 未命中则正常分配，分配后注册到 prefix cache
        """
        if seq.seq_id in self._seq_blocks:
            raise ValueError(f"Seq {seq.seq_id} already has blocks allocated")

        blocks: List[PhysicalBlock] = []

        for i, logical_block in enumerate(seq.logical_blocks):
            phys_block = self._allocate_or_hit(logical_block)
            seq.block_table[i] = phys_block
            blocks.append(phys_block)

        self._seq_blocks[seq.seq_id] = blocks

    def _allocate_or_hit(self, logical_block) -> PhysicalBlock:
        """为单个逻辑块分配物理块（优先命中 prefix cache）。"""
        if (
            self.prefix_cache is not None
            and logical_block.is_full
            and logical_block.content_hash is not None
        ):
            cached = self.prefix_cache.lookup(logical_block.content_hash)
            if cached is not None:
                return cached  # ref_count 已在 lookup 内+1

        # cache miss 或块未满：分配新物理块
        phys_block = self.allocator.allocate()

        # 满块注册到 prefix cache
        if (
            self.prefix_cache is not None
            and logical_block.is_full
            and logical_block.content_hash is not None
        ):
            self.prefix_cache.register(logical_block.content_hash, phys_block)

        return phys_block

    def can_append_slot(self, seq: Sequence) -> bool:
        """
        判断是否能为序列追加下一个 token 的槽位。

        如果最后一个逻辑块已满，需要新物理块；否则直接写入当前块。
        """
        if seq.needs_new_block():
            return self.allocator.can_allocate(1)
        return True

    def append_slot(self, seq: Sequence) -> Tuple[bool, bool]:
        """
        为序列的下一个 token 预留槽位。

        如果最后一个逻辑块已满，分配新物理块并更新 block_table。
        如果当前块被多个序列共享（prefix cache CoW），触发复制。

        Returns:
            (new_block_allocated, cow_happened)
        """
        last_logical = seq.logical_blocks[-1] if seq.logical_blocks else None

        # 当前块已满 → 需要新块
        if seq.needs_new_block():
            new_block = self.allocator.allocate()
            new_logical_idx = seq.num_logical_blocks  # 将要新增的逻辑块编号
            seq.block_table[new_logical_idx] = new_block
            self._seq_blocks[seq.seq_id].append(new_block)
            return True, False

        # 当前块未满 → 检查是否需要 CoW
        if self.prefix_cache is not None and last_logical:
            last_phys = seq.block_table.get(len(seq.logical_blocks) - 1)
            if last_phys is not None and last_phys.is_shared:
                new_block, did_cow = self.prefix_cache.cow_if_needed(last_phys)
                if did_cow:
                    logical_idx = len(seq.logical_blocks) - 1
                    seq.block_table[logical_idx] = new_block
                    # 更新 seq_blocks 列表
                    blk_list = self._seq_blocks[seq.seq_id]
                    blk_list[-1] = new_block
                    return False, True

        return False, False

    # ── 释放 ─────────────────────────────────────────────────────────────────

    def free(self, seq: Sequence) -> None:
        """
        释放序列持有的所有物理块。

        调用时机：序列结束（FINISHED）或被抢占（GPU 块释放给 swap/recompute）。
        """
        if seq.seq_id not in self._seq_blocks:
            return
        self.allocator.free_blocks(self._seq_blocks.pop(seq.seq_id))
        seq.block_table.clear()

    def free_without_cache_invalidation(self, seq: Sequence) -> None:
        """
        释放块，但保留 prefix cache 中的引用（被 swap out 时调用）。
        swap out 时 prefix cache 的引用仍然有效，不应 invalidate。
        """
        self.free(seq)  # 当前实现与 free 相同；生产实现中会有差异

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

    @property
    def prefix_cache_stats(self) -> Optional[Dict]:
        if self.prefix_cache is None:
            return None
        return self.prefix_cache.stats()

    def __repr__(self) -> str:
        return (
            f"KVCacheManager("
            f"free={self.num_free_blocks}, "
            f"used={self.num_used_blocks}, "
            f"util={self.utilization:.1%})"
        )
