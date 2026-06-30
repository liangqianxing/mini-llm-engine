"""KVCacheManager 单元测试。"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.kv_cache import KVCacheManager
from engine.sequence import Sequence, SamplingParams


def make_seq(seq_id, prompt_len, block_size=4, max_tokens=20):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        block_size=block_size,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class TestKVCacheManager:

    def test_can_allocate_enough_blocks(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)  # needs 2 blocks
        assert mgr.can_allocate(seq)

    def test_cannot_allocate_too_many_blocks(self):
        mgr = KVCacheManager(num_blocks=1, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)  # needs 2 blocks
        assert not mgr.can_allocate(seq)

    def test_allocate_creates_block_table(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)
        mgr.allocate(seq)
        assert len(seq.block_table) == seq.num_logical_blocks
        assert all(v is not None for v in seq.block_table.values())

    def test_allocate_reduces_free_blocks(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)  # 2 logical blocks
        before = mgr.num_free_blocks
        mgr.allocate(seq)
        assert mgr.num_free_blocks == before - seq.num_logical_blocks

    def test_free_returns_all_blocks(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)
        mgr.allocate(seq)
        used = mgr.num_used_blocks
        mgr.free(seq)
        assert mgr.num_used_blocks == 0
        assert mgr.num_free_blocks == 16

    def test_append_slot_no_new_block_needed(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        # prompt_len=3：填满不完整的最后一个块，decode 时先不需要新块
        seq = make_seq(0, prompt_len=3, block_size=4)
        mgr.allocate(seq)
        assert mgr.can_append_slot(seq)
        new_block_allocated, cow_happened = mgr.append_slot(seq)
        assert new_block_allocated is False  # 不需要新物理块
        assert cow_happened is False

    def test_append_slot_needs_new_block(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        # prompt_len=4：正好填满，下一个 token 需要新块
        seq = make_seq(0, prompt_len=4, block_size=4)
        mgr.allocate(seq)
        assert seq.needs_new_block()
        new_block_allocated, cow_happened = mgr.append_slot(seq)
        assert new_block_allocated is True  # 分配了新物理块

    def test_get_block_table_returns_ids(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)
        mgr.allocate(seq)
        table = mgr.get_block_table(seq)
        assert len(table) == seq.num_logical_blocks
        assert all(isinstance(bid, int) for bid in table)

    def test_double_allocate_raises(self):
        mgr = KVCacheManager(num_blocks=16, block_size=4)
        seq = make_seq(0, prompt_len=4, block_size=4)
        mgr.allocate(seq)
        with pytest.raises(ValueError):
            mgr.allocate(seq)

    def test_utilization_metric(self):
        mgr = KVCacheManager(num_blocks=10, block_size=4)
        seq = make_seq(0, prompt_len=8, block_size=4)  # 2 blocks
        mgr.allocate(seq)
        assert abs(mgr.utilization - 0.2) < 0.01  # 2/10
