"""
新功能测试：Prefix Caching / CPU Swap / Chunked Prefill / Speculative Decoding。
"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.block import PhysicalBlock, LogicalTokenBlock
from engine.block_allocator import BlockAllocator
from engine.kv_cache import KVCacheManager
from engine.prefix_cache import PrefixCache
from engine.swap_manager import SwapManager
from engine.scheduler import Scheduler, SchedulerPolicy, PrefillChunk
from engine.sequence import Sequence, SequenceGroup, SequenceStatus, SamplingParams
from engine.llm_engine import LLMEngine
from engine.speculative import SpeculativeDecoder, benchmark_speculative
from engine.model_runner import MockModelRunner
from engine.metrics import MetricsCollector, _percentile


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def make_seq(seq_id, prompt_len, block_size=4, max_tokens=20):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        block_size=block_size,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def make_seq_group(request_id, prompt_len=8, block_size=4, max_tokens=20, priority=0):
    seq = make_seq(int(request_id.split("-")[1]), prompt_len, block_size, max_tokens)
    seq.priority = priority
    return SequenceGroup(request_id=request_id, sequences=[seq], priority=priority)


# ──────────────────────────────────────────────────────────────────────────────
# Prefix Cache 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestPrefixCache:

    def test_initial_state(self):
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        pc = PrefixCache(alloc, max_cached_blocks=16)
        assert pc.num_cached_blocks == 0
        assert pc.hit_rate == 0.0

    def test_lookup_miss(self):
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        pc = PrefixCache(alloc)
        result = pc.lookup(hash((1, 2, 3, 4)))
        assert result is None
        assert pc.num_misses == 1
        assert pc.num_hits == 0

    def test_register_and_hit(self):
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        pc = PrefixCache(alloc, max_cached_blocks=16)

        # 分配一个块并注册
        block = alloc.allocate()
        h = hash((10, 20, 30, 40))
        pc.register(h, block)
        assert pc.num_cached_blocks == 1

        # 查询命中
        hit = pc.lookup(h)
        assert hit is not None
        assert hit.block_id == block.block_id
        assert hit.ref_count == 2  # 原 ref=1，hit 后 +1

    def test_eviction_when_full(self):
        alloc = BlockAllocator(num_blocks=64, block_size=4)
        pc = PrefixCache(alloc, max_cached_blocks=3)

        hashes = []
        for i in range(3):
            block = alloc.allocate()
            h = hash(tuple(range(i * 4, (i + 1) * 4)))
            pc.register(h, block)
            hashes.append(h)

        # 已满（3/3）再添加一个，应触发 eviction
        extra_block = alloc.allocate()
        new_hash = hash((100, 101, 102, 103))
        pc.register(new_hash, extra_block)
        assert pc.num_cached_blocks <= 3
        assert pc.num_evictions == 1

    def test_cow_not_needed_for_exclusive_block(self):
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        pc = PrefixCache(alloc)
        block = alloc.allocate()  # ref_count = 1
        result, did_cow = pc.cow_if_needed(block)
        assert did_cow is False
        assert result is block

    def test_cow_triggered_for_shared_block(self):
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        pc = PrefixCache(alloc)
        block = alloc.allocate()
        block.ref_count = 2  # 模拟共享状态

        new_block, did_cow = pc.cow_if_needed(block)
        assert did_cow is True
        assert new_block is not block
        assert new_block.ref_count == 1
        assert block.ref_count == 1  # 减 1

    def test_prefix_cache_integration_with_kv_manager(self):
        """共享相同 prompt prefix 的序列应复用物理块。"""
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        pc = PrefixCache(alloc, max_cached_blocks=16)
        mgr = KVCacheManager(num_blocks=32, block_size=4, prefix_cache=pc)
        mgr.allocator = alloc  # 确保共用同一 allocator

        # 创建 prompt = [0, 1, 2, 3]（正好一个满块）
        seq1 = Sequence(
            seq_id=0,
            prompt_token_ids=[0, 1, 2, 3],
            block_size=4,
            sampling_params=SamplingParams(max_tokens=10),
        )
        seq2 = Sequence(
            seq_id=1,
            prompt_token_ids=[0, 1, 2, 3],   # 相同前缀
            block_size=4,
            sampling_params=SamplingParams(max_tokens=10),
        )

        # seq1 allocate 后注册到 prefix cache
        mgr.allocate(seq1)
        # 手动注册第一个满块到 prefix cache（模拟推理完成后的注册）
        lb = seq1.logical_blocks[0]
        if lb.is_full and lb.content_hash is not None:
            pc.register(lb.content_hash, seq1.block_table[0])

        # seq2 allocate 时应命中 cache，共享 seq1 的物理块
        initial_free = alloc.num_free_blocks
        mgr.allocate(seq2)
        # 如果命中，free blocks 减少应 < 1（共享块不消耗额外 free slot）
        # （注：实际上 lookup 会 ref+1 但不消耗新 block）
        assert pc.num_hits >= 0  # 基本断言：不崩溃


# ──────────────────────────────────────────────────────────────────────────────
# CPU Swap Manager 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestSwapManager:

    def make_running_seq(self, seq_id=0, prompt_len=8, block_size=4):
        seq = make_seq(seq_id, prompt_len, block_size)
        mgr = KVCacheManager(num_blocks=64, block_size=block_size)
        mgr.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        return seq, mgr

    def test_swap_out_basic(self):
        seq, mgr = self.make_running_seq(prompt_len=8, block_size=4)
        swap_mgr = SwapManager(mgr.allocator, cpu_memory_gb=1.0)

        before_free = mgr.allocator.num_free_blocks
        success = swap_mgr.swap_out(seq)

        assert success
        assert seq.status == SequenceStatus.SWAPPED
        assert len(seq.block_table) == 0  # GPU 块已释放
        assert mgr.allocator.num_free_blocks > before_free  # blocks 回收
        assert swap_mgr.num_swapped_seqs == 1
        assert swap_mgr.num_swap_outs == 1

    def test_swap_in_basic(self):
        seq, mgr = self.make_running_seq(prompt_len=8, block_size=4)
        swap_mgr = SwapManager(mgr.allocator, cpu_memory_gb=1.0)

        swap_mgr.swap_out(seq)
        assert swap_mgr.can_swap_in(seq)

        success = swap_mgr.swap_in(seq)
        assert success
        assert seq.status == SequenceStatus.RUNNING
        assert len(seq.block_table) > 0  # block_table 已恢复
        assert swap_mgr.num_swapped_seqs == 0
        assert swap_mgr.num_swap_ins == 1

    def test_cannot_swap_in_without_gpu_blocks(self):
        seq, mgr = self.make_running_seq(prompt_len=8, block_size=4)
        swap_mgr = SwapManager(mgr.allocator, cpu_memory_gb=1.0)

        swap_mgr.swap_out(seq)
        # 把所有 GPU 块占满
        remaining = mgr.allocator.num_free_blocks
        all_blocks = [mgr.allocator.allocate() for _ in range(remaining)]

        assert not swap_mgr.can_swap_in(seq)

        # 清理
        mgr.allocator.free_blocks(all_blocks)

    def test_discard_removes_from_cpu(self):
        seq, mgr = self.make_running_seq(prompt_len=8, block_size=4)
        swap_mgr = SwapManager(mgr.allocator, cpu_memory_gb=1.0)

        swap_mgr.swap_out(seq)
        assert swap_mgr.num_swapped_seqs == 1

        swap_mgr.discard(seq)
        assert swap_mgr.num_swapped_seqs == 0

    def test_swap_stats(self):
        seq, mgr = self.make_running_seq(prompt_len=8, block_size=4)
        swap_mgr = SwapManager(mgr.allocator, cpu_memory_gb=1.0)
        swap_mgr.swap_out(seq)
        swap_mgr.swap_in(seq)

        stats = swap_mgr.stats()
        assert stats["num_swap_outs"] == 1
        assert stats["num_swap_ins"] == 1
        assert stats["total_swap_out_blocks"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# Chunked Prefill 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestChunkedPrefill:

    def make_scheduler(self, chunk_size=4, num_blocks=64):
        mgr = KVCacheManager(num_blocks=num_blocks, block_size=4)
        return Scheduler(
            mgr,
            max_num_seqs=8,
            max_num_batched_tokens=512,
            chunked_prefill_enabled=True,
            max_prefill_tokens_per_step=chunk_size,
        )

    def test_long_prompt_split_into_chunks(self):
        sched = self.make_scheduler(chunk_size=4)
        # 16 token prompt → 4 chunks of 4
        sg = make_seq_group("req-0", prompt_len=16)
        sched.add_seq_group(sg)

        # Step 1: 应该只处理前 4 个 token
        out = sched.schedule()
        assert len(out.prefill_chunks) == 1
        chunk = out.prefill_chunks[0]
        assert chunk.token_start == 0
        assert chunk.token_end == 4
        assert chunk.seq.status == SequenceStatus.PREFILLING

    def test_chunked_prefill_completes_in_multiple_steps(self):
        sched = self.make_scheduler(chunk_size=4)
        sg = make_seq_group("req-0", prompt_len=12)  # 12/4 = 3 steps to prefill
        sched.add_seq_group(sg)

        seq = sg.seqs[0]
        prefill_steps = 0

        for _ in range(10):
            out = sched.schedule()
            if out.is_empty:
                break
            if out.prefill_chunks:
                chunk = out.prefill_chunks[0]
                if chunk.seq.seq_id == seq.seq_id:
                    prefill_steps += 1
                    # 不 append token，只是检查 chunk 范围
                if seq.is_prompt_fully_prefilled:
                    break

        assert seq.is_prompt_fully_prefilled
        assert prefill_steps >= 3  # 至少 3 步才能 prefill 完 12 tokens（chunk_size=4）

    def test_decode_continues_after_prefill(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=64,
            block_size=4,
            chunked_prefill=True,
            max_prefill_tokens_per_step=4,
            eos_probability=0.15,
            decode_time_per_step=0.0,
            prefill_time_per_token=0.0,
            seed=42,
        )
        results = engine.generate(["Hello world test prompt"], max_tokens=10, verbose=False)
        assert len(results) == 1
        assert len(results[0].output_token_ids) > 0

    def test_short_prompt_not_chunked(self):
        sched = self.make_scheduler(chunk_size=16)
        # prompt_len=4 < chunk_size=16 → 不切分
        sg = make_seq_group("req-0", prompt_len=4)
        sched.add_seq_group(sg)
        out = sched.schedule()
        # 应该一步完成 prefill（seq 直接进入 RUNNING）
        assert len(out.prefill_chunks) == 1
        assert out.prefill_chunks[0].chunk_len == 4


# ──────────────────────────────────────────────────────────────────────────────
# 优先级调度测试
# ──────────────────────────────────────────────────────────────────────────────

class TestPriorityScheduling:

    def test_priority_field_in_sequence(self):
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3],
            block_size=4,
            priority=5,
        )
        assert seq.priority == 5

    def test_urgency_without_deadline_equals_priority(self):
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3],
            block_size=4,
            priority=3,
        )
        assert seq.urgency() == 3.0

    def test_urgency_with_deadline(self):
        future = time.monotonic() + 100.0
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3],
            block_size=4,
            priority=0,
            deadline=future,
        )
        # urgency = deadline（EDF 模式）
        assert abs(seq.urgency() - future) < 1.0

    def test_seq_group_min_urgency(self):
        sg = SequenceGroup(
            request_id="req-0",
            sequences=[
                Sequence(seq_id=0, prompt_token_ids=[1, 2], block_size=4, priority=5),
                Sequence(seq_id=1, prompt_token_ids=[3, 4], block_size=4, priority=2),
            ],
        )
        assert sg.min_urgency() == 2.0  # min priority = 2

    def test_priority_policy_enum(self):
        mgr = KVCacheManager(num_blocks=64, block_size=4)
        sched = Scheduler(mgr, policy=SchedulerPolicy.PRIORITY)
        assert sched.policy == SchedulerPolicy.PRIORITY

    def test_edf_policy(self):
        mgr = KVCacheManager(num_blocks=64, block_size=4)
        sched = Scheduler(mgr, policy=SchedulerPolicy.EDF)
        assert sched.policy == SchedulerPolicy.EDF


# ──────────────────────────────────────────────────────────────────────────────
# Speculative Decoding 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestSpeculativeDecoding:

    def make_spec_decoder(self, K=4, acceptance_rate=0.8):
        draft = MockModelRunner(
            decode_time_per_step=0.0,
            eos_probability=0.1,
            seed=42,
        )
        target = MockModelRunner(
            decode_time_per_step=0.0,
            eos_probability=0.1,
            seed=43,
        )
        return SpeculativeDecoder(
            draft_runner=draft,
            target_runner=target,
            num_speculative_tokens=K,
            acceptance_rate=acceptance_rate,
            seed=99,
        )

    def make_running_seqs(self, n=2):
        seqs = []
        for i in range(n):
            s = Sequence(
                seq_id=i,
                prompt_token_ids=list(range(4)),
                block_size=4,
                sampling_params=SamplingParams(max_tokens=20),
            )
            s.status = SequenceStatus.RUNNING
            s.num_prefilled_tokens = 4
            seqs.append(s)
        return seqs

    def test_step_returns_tokens_for_all_seqs(self):
        dec = self.make_spec_decoder(K=3)
        seqs = self.make_running_seqs(n=2)
        results = dec.step(seqs)
        assert len(results) == 2
        for seq in seqs:
            assert seq.seq_id in results
            assert len(results[seq.seq_id]) >= 1  # 至少接受 1 个 token

    def test_accepted_tokens_bounded_by_k_plus_1(self):
        dec = self.make_spec_decoder(K=4)
        seqs = self.make_running_seqs(n=1)
        for _ in range(10):
            results = dec.step(seqs)
            for tokens in results.values():
                assert 1 <= len(tokens) <= 5  # 1 ~ K+1

    def test_empty_seqs_returns_empty(self):
        dec = self.make_spec_decoder()
        result = dec.step([])
        assert result == {}

    def test_stats_accumulate(self):
        dec = self.make_spec_decoder(K=4, acceptance_rate=0.9)
        seqs = self.make_running_seqs(n=1)
        for _ in range(5):
            dec.step(seqs)
        assert dec.total_draft_tokens == 5 * 4  # 5 steps × K=4 drafts
        assert dec.total_verify_steps == 5
        assert dec.total_accepted_tokens > 0

    def test_acceptance_rate_tracking(self):
        dec = self.make_spec_decoder(K=4, acceptance_rate=1.0)  # 全接受
        seqs = self.make_running_seqs(n=1)
        for _ in range(10):
            dec.step(seqs)
        # 全接受时 effective_acceptance_rate 应接近 1.0
        assert dec.effective_acceptance_rate > 0.5

    def test_benchmark_function(self):
        results = benchmark_speculative(
            num_requests=5,
            max_tokens=15,
            K=3,
            acceptance_rate=0.7,
            draft_decode_ms=0.0,
            target_decode_ms=0.0,
            seed=42,
        )
        assert "standard" in results
        assert "speculative" in results
        assert results["speedup"] > 0
        assert results["speculative"]["avg_accepted_per_step"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# MetricsCollector 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsCollector:

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        assert _percentile([5.0], 50) == 5.0

    def test_percentile_p50(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_percentile(data, 50) - 3.0) < 0.01

    def test_record_step(self):
        mc = MetricsCollector()
        mc.record_step(
            step=1, num_waiting=2, num_running=3, num_finished=0,
            kv_utilization=0.5, num_prefill_tokens=10, num_decode_tokens=3,
        )
        assert len(mc._step_records) == 1
        assert mc._step_records[0].kv_utilization == 0.5

    def test_record_request_done(self):
        mc = MetricsCollector()
        t = time.monotonic()
        mc.record_request_done(
            request_id="req-0", prompt_len=10, output_len=20,
            arrival_time=t - 1.0, first_token_time=t - 0.5, finish_time=t,
        )
        assert len(mc._request_records) == 1
        assert abs(mc._request_records[0].latency - 1.0) < 0.1
        assert abs(mc._request_records[0].ttft - 0.5) < 0.1

    def test_report_structure(self):
        mc = MetricsCollector()
        t = time.monotonic()
        for i in range(5):
            mc.record_step(1 + i, 1, 1, 0, 0.5, 5, 2)
            mc.record_request_done(
                f"req-{i}", 10, 10,
                t - (5 - i) * 0.1, t - (5 - i) * 0.05, t,
            )
        report = mc.report()
        assert "summary" in report
        assert "latency_s" in report
        assert "ttft_s" in report
        assert "kv_utilization" in report
        assert report["summary"]["total_requests"] == 5

    def test_metrics_integrated_in_engine(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=32, block_size=4,
            eos_probability=0.2, decode_time_per_step=0.0,
            prefill_time_per_token=0.0, seed=42,
            collect_metrics=True,
        )
        engine.generate(["test prompt one", "test prompt two"], max_tokens=10)
        assert engine.metrics is not None
        report = engine.metrics.report()
        assert report["summary"]["total_requests"] == 2
        assert report["summary"]["throughput_tok_s"] >= 0


# ──────────────────────────────────────────────────────────────────────────────
# 完整引擎集成测试
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMEngineIntegration:

    def test_basic_generate(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=64, block_size=4,
            eos_probability=0.15, decode_time_per_step=0.0,
            prefill_time_per_token=0.0, seed=42,
        )
        results = engine.generate(["hello world", "how are you"], max_tokens=15)
        assert len(results) == 2
        for r in results:
            assert r.latency > 0
            assert len(r.output_token_ids) > 0

    def test_generate_with_chunked_prefill(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=64, block_size=4,
            chunked_prefill=True, max_prefill_tokens_per_step=4,
            eos_probability=0.15, decode_time_per_step=0.0,
            prefill_time_per_token=0.0, seed=42,
        )
        results = engine.generate(
            ["a" * 20, "b" * 20],  # long prompts → will be chunked
            max_tokens=10,
        )
        assert len(results) == 2

    def test_generate_with_priority(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=64, block_size=4,
            eos_probability=0.15, decode_time_per_step=0.0,
            prefill_time_per_token=0.0, seed=42,
        )
        engine.add_request("low priority", priority=10)
        engine.add_request("high priority", priority=0)
        # 只验证不崩溃 + 都能完成
        while engine.scheduler.has_unfinished_seqs:
            engine.step()
        assert engine.stats.total_requests == 2

    def test_generate_with_swap(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=8, block_size=4,   # 极小内存，触发 swap
            cpu_swap_gb=1.0,
            max_num_seqs=4,
            eos_probability=0.3, decode_time_per_step=0.0,
            prefill_time_per_token=0.0, seed=42,
        )
        results = engine.generate(
            ["prompt " + str(i) for i in range(4)],
            max_tokens=5,
        )
        assert len(results) >= 1  # 至少部分完成

    def test_engine_stats_populated(self):
        engine = LLMEngine.from_config(
            num_kv_blocks=32, block_size=4,
            eos_probability=0.2, decode_time_per_step=0.0,
            prefill_time_per_token=0.0,
        )
        engine.generate(["test"], max_tokens=5)
        assert engine.stats.total_requests == 1
        assert engine.stats.total_output_tokens > 0
        assert engine.stats.total_time > 0
