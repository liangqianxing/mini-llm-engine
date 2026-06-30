"""
CPU Swap Manager（KV Cache CPU 卸载）。

当 GPU 显存不足时，可以将被抢占序列的 KV cache 转移到 CPU RAM，
等资源释放后再换回 GPU。这比直接丢弃 KV cache（需要重新 prefill）更高效。

性能权衡
────────
  - Swap out 延迟：GPU→CPU PCIe 带宽，通常比 GPU 内存带宽低 10-50x
  - Swap in 延迟：CPU→GPU，同上
  - 优势：避免昂贵的重新 prefill（长序列时 prefill >> swap 延迟）

模拟实现
────────
真实系统中，swap in/out 操作 CUDA kernel 拷贝 tensor 数据。
本实现用字典模拟 CPU 内存（记录块 ID 映射），不实际存储 KV 数值，
专注于展示内存管理策略层的正确性。

与 vLLM 的对应关系
────────────────────
  - vLLM 的 cpu_swap_space 参数控制 CPU 内存池大小
  - preemption_mode 可选 "swap" 或 "recompute"
"""

from typing import Dict, List, Optional, Tuple

from .block import PhysicalBlock
from .block_allocator import BlockAllocator
from .sequence import Sequence, SequenceStatus


class SwapManager:
    """
    KV Cache CPU Swap Manager。

    管理 GPU 块和 CPU 内存之间的数据搬运。

    Args:
        gpu_allocator:     GPU 物理块分配器
        cpu_memory_gb:     模拟 CPU 内存池大小（GB，用于检查容量）
        block_size_bytes:  每个 KV cache 块的字节数（用于容量计算）
    """

    # 典型值：GPT-2 small, block_size=16
    # = 2 * 12 layers * 12 heads * 64 head_dim * 16 tokens * 2 bytes (fp16)
    # ≈ 47KB per block
    DEFAULT_BLOCK_SIZE_BYTES = 48 * 1024  # 48 KB

    def __init__(
        self,
        gpu_allocator: BlockAllocator,
        cpu_memory_gb: float = 4.0,
        block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
    ) -> None:
        self.gpu_allocator = gpu_allocator
        self.block_size_bytes = block_size_bytes

        # CPU 内存容量（块数）
        self.cpu_num_blocks = int(
            cpu_memory_gb * 1024 ** 3 / block_size_bytes
        )

        # CPU 存储：seq_id → [(logical_idx, cpu_slot_id), ...]
        # cpu_slot_id 模拟 CPU 内存中的位置
        self._cpu_storage: Dict[int, List[Tuple[int, int]]] = {}
        self._cpu_used_slots: int = 0
        self._next_cpu_slot: int = 0

        # 统计
        self.num_swap_outs: int = 0
        self.num_swap_ins: int = 0
        self.total_swap_out_blocks: int = 0
        self.total_swap_in_blocks: int = 0

    # ── Swap Out（GPU → CPU）────────────────────────────────────────────────

    def can_swap_out(self, seq: Sequence) -> bool:
        """检查是否有足够的 CPU 空间存放该序列的 KV cache。"""
        return self._cpu_used_slots + seq.num_logical_blocks <= self.cpu_num_blocks

    def swap_out(self, seq: Sequence) -> bool:
        """
        将序列的 KV cache 从 GPU 转移到 CPU。

        操作：
          1. 记录 logical_block_idx → cpu_slot 的映射
          2. 将 GPU 物理块归还给分配器（模拟释放显存）
          3. 序列状态改为 SWAPPED

        Args:
            seq: 要 swap out 的序列（必须处于 RUNNING 或 PREFILLING 状态）

        Returns:
            True  成功
            False CPU 空间不足
        """
        if not self.can_swap_out(seq):
            return False

        # 记录 logical block → cpu slot 映射
        cpu_mapping: List[Tuple[int, int]] = []
        for logical_idx, phys_block in list(seq.block_table.items()):
            cpu_slot = self._next_cpu_slot
            cpu_mapping.append((logical_idx, cpu_slot))
            self._next_cpu_slot += 1
            self._cpu_used_slots += 1

            # 释放 GPU 物理块（模拟：实际应先拷贝数据到 CPU）
            self.gpu_allocator.free(phys_block)

        seq.block_table.clear()
        self._cpu_storage[seq.seq_id] = cpu_mapping
        seq.status = SequenceStatus.SWAPPED

        self.num_swap_outs += 1
        self.total_swap_out_blocks += len(cpu_mapping)

        return True

    # ── Swap In（CPU → GPU）────────────────────────────────────────────────

    def can_swap_in(self, seq: Sequence) -> bool:
        """检查是否有足够的 GPU 块来恢复该序列。"""
        if seq.seq_id not in self._cpu_storage:
            return False
        num_blocks_needed = len(self._cpu_storage[seq.seq_id])
        return self.gpu_allocator.can_allocate(num_blocks_needed)

    def swap_in(self, seq: Sequence) -> bool:
        """
        将序列的 KV cache 从 CPU 恢复到 GPU。

        操作：
          1. 分配新 GPU 物理块
          2. 恢复 logical → physical 的 block_table
          3. 释放 CPU 槽位
          4. 序列状态改为 RUNNING

        Args:
            seq: 要 swap in 的序列（必须处于 SWAPPED 状态）

        Returns:
            True  成功
            False GPU 空间不足
        """
        if not self.can_swap_in(seq):
            return False

        cpu_mapping = self._cpu_storage.pop(seq.seq_id)

        for logical_idx, _cpu_slot in cpu_mapping:
            new_block = self.gpu_allocator.allocate()
            seq.block_table[logical_idx] = new_block
            self._cpu_used_slots -= 1

        seq.status = SequenceStatus.RUNNING

        self.num_swap_ins += 1
        self.total_swap_in_blocks += len(cpu_mapping)

        return True

    def discard(self, seq: Sequence) -> None:
        """
        丢弃 CPU 中的 KV cache（不恢复）。

        当决定放弃该序列或改为 recompute 策略时调用。
        """
        if seq.seq_id in self._cpu_storage:
            cpu_mapping = self._cpu_storage.pop(seq.seq_id)
            self._cpu_used_slots -= len(cpu_mapping)

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    @property
    def cpu_utilization(self) -> float:
        return self._cpu_used_slots / self.cpu_num_blocks if self.cpu_num_blocks > 0 else 0.0

    @property
    def num_swapped_seqs(self) -> int:
        return len(self._cpu_storage)

    def stats(self) -> Dict:
        return {
            "num_swapped_seqs": self.num_swapped_seqs,
            "cpu_used_slots": self._cpu_used_slots,
            "cpu_total_slots": self.cpu_num_blocks,
            "cpu_utilization": self.cpu_utilization,
            "num_swap_outs": self.num_swap_outs,
            "num_swap_ins": self.num_swap_ins,
            "total_swap_out_blocks": self.total_swap_out_blocks,
            "total_swap_in_blocks": self.total_swap_in_blocks,
        }

    def __repr__(self) -> str:
        return (
            f"SwapManager("
            f"swapped={self.num_swapped_seqs}, "
            f"cpu_util={self.cpu_utilization:.1%})"
        )
