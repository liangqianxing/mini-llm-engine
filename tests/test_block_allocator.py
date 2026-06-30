"""BlockAllocator 单元测试。"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.block_allocator import BlockAllocator


class TestBlockAllocator:

    def test_initial_state(self):
        alloc = BlockAllocator(num_blocks=16, block_size=8)
        assert alloc.num_blocks == 16
        assert alloc.num_free_blocks == 16
        assert alloc.num_used_blocks == 0
        assert alloc.utilization == 0.0

    def test_allocate_single(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        block = alloc.allocate()
        assert block.ref_count == 1
        assert alloc.num_free_blocks == 3
        assert alloc.num_used_blocks == 1

    def test_allocate_all(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        blocks = [alloc.allocate() for _ in range(4)]
        assert alloc.num_free_blocks == 0
        assert alloc.num_used_blocks == 4
        assert alloc.utilization == 1.0

    def test_oom_raises(self):
        alloc = BlockAllocator(num_blocks=2, block_size=16)
        alloc.allocate()
        alloc.allocate()
        with pytest.raises(MemoryError):
            alloc.allocate()

    def test_free_returns_to_pool(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        block = alloc.allocate()
        alloc.free(block)
        assert alloc.num_free_blocks == 4
        assert alloc.num_used_blocks == 0

    def test_free_blocks_batch(self):
        alloc = BlockAllocator(num_blocks=8, block_size=16)
        blocks = [alloc.allocate() for _ in range(5)]
        alloc.free_blocks(blocks)
        assert alloc.num_free_blocks == 8

    def test_can_allocate(self):
        alloc = BlockAllocator(num_blocks=4, block_size=16)
        assert alloc.can_allocate(4)
        assert not alloc.can_allocate(5)
        alloc.allocate()
        assert alloc.can_allocate(3)
        assert not alloc.can_allocate(4)

    def test_block_ids_are_unique(self):
        alloc = BlockAllocator(num_blocks=8, block_size=16)
        blocks = [alloc.allocate() for _ in range(8)]
        ids = {b.block_id for b in blocks}
        assert len(ids) == 8

    def test_freed_blocks_reusable(self):
        alloc = BlockAllocator(num_blocks=2, block_size=16)
        b1 = alloc.allocate()
        b2 = alloc.allocate()
        alloc.free(b1)
        b3 = alloc.allocate()  # 应该能重新分配
        assert b3.ref_count == 1
        assert alloc.num_free_blocks == 0
