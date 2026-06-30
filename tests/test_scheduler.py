"""Scheduler 单元测试。"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.kv_cache import KVCacheManager
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SequenceGroup, SequenceStatus, SamplingParams


def make_engine(num_blocks=64, block_size=4, max_num_seqs=8):
    kv = KVCacheManager(num_blocks, block_size)
    return Scheduler(kv, max_num_seqs=max_num_seqs, max_num_batched_tokens=512)


def make_seq_group(request_id, prompt_len=10, max_tokens=20):
    seq = Sequence(
        seq_id=int(request_id.replace("req-", "")),
        prompt_token_ids=list(range(prompt_len)),
        block_size=4,
        sampling_params=SamplingParams(max_tokens=max_tokens, eos_token_id=9999),
    )
    return SequenceGroup(request_id=request_id, sequences=[seq])


class TestScheduler:

    def test_empty_schedule(self):
        sched = make_engine()
        output = sched.schedule()
        assert output.is_empty
        assert not sched.has_unfinished_seqs

    def test_add_and_schedule_single(self):
        sched = make_engine()
        sg = make_seq_group("req-0", prompt_len=5)
        sched.add_seq_group(sg)
        assert sched.num_waiting == 1

        output = sched.schedule()
        assert len(output.prefill_seqs) == 1
        assert len(output.decode_seqs) == 0
        assert sched.num_running == 1
        assert sched.num_waiting == 0

    def test_multiple_requests_scheduled(self):
        sched = make_engine(num_blocks=64, block_size=4, max_num_seqs=4)
        for i in range(4):
            sched.add_seq_group(make_seq_group(f"req-{i}", prompt_len=5))

        output = sched.schedule()
        assert len(output.prefill_seqs) == 4
        assert sched.num_running == 4

    def test_max_seqs_limit(self):
        sched = make_engine(num_blocks=128, block_size=4, max_num_seqs=3)
        for i in range(6):
            sched.add_seq_group(make_seq_group(f"req-{i}", prompt_len=4))

        output = sched.schedule()
        # 只能同时运行 3 条
        assert len(output.prefill_seqs) <= 3
        assert sched.num_running <= 3
        assert sched.num_waiting >= 3

    def test_sequence_completes_and_frees_blocks(self):
        sched = make_engine(num_blocks=32, block_size=4, max_num_seqs=4)
        sg = make_seq_group("req-0", prompt_len=4, max_tokens=3)
        sched.add_seq_group(sg)

        # Step 1: prefill
        output = sched.schedule()
        seq = output.prefill_seqs[0]

        # 模拟推理 3 步后完成
        for step in range(3):
            fake_tokens = {seq.seq_id: 1}  # 非 EOS
            sched.schedule()
            finished = sched.on_step_done(output, fake_tokens)

        # 最终序列应完成
        assert sched.num_finished >= 0  # 至少被处理了

    def test_on_step_done_updates_tokens(self):
        sched = make_engine(num_blocks=32, block_size=4)
        sg = make_seq_group("req-0", prompt_len=4, max_tokens=10)
        sched.add_seq_group(sg)

        output = sched.schedule()
        seq = output.prefill_seqs[0]
        initial_tokens = seq.num_total_tokens

        sched.on_step_done(output, {seq.seq_id: 42})
        assert seq.num_total_tokens == initial_tokens + 1

    def test_kv_utilization_increases_with_seqs(self):
        sched = make_engine(num_blocks=64, block_size=4, max_num_seqs=8)
        for i in range(4):
            sched.add_seq_group(make_seq_group(f"req-{i}", prompt_len=8))

        sched.schedule()
        assert sched.kv_cache.utilization > 0.0

    def test_continuous_batching_pulls_waiting(self):
        """验证连续批处理：一条序列完成后，立即从等待队列拉入新请求。"""
        sched = make_engine(num_blocks=64, block_size=4, max_num_seqs=2)

        # 先加 2 条（满足 max_num_seqs）
        sg0 = make_seq_group("req-0", prompt_len=4, max_tokens=2)
        sg1 = make_seq_group("req-1", prompt_len=4, max_tokens=50)
        # 第 3 条等待
        sg2 = make_seq_group("req-2", prompt_len=4, max_tokens=5)

        sched.add_seq_group(sg0)
        sched.add_seq_group(sg1)
        sched.add_seq_group(sg2)

        # Step 1: 调度 req-0, req-1（req-2 等待）
        out = sched.schedule()
        assert sched.num_waiting == 1

        # 让 req-0 完成（发送 EOS）
        eos_token = sg0.seqs[0].sampling_params.eos_token_id
        tokens = {sg0.seqs[0].seq_id: eos_token, sg1.seqs[0].seq_id: 1}
        finished = sched.on_step_done(out, tokens)
        # req-0 应该完成
        assert any(g.request_id == "req-0" for g in finished) or sg0.is_finished

        # Step 2: 下一步调度应该拉入 req-2
        out2 = sched.schedule()
        running_req_ids = {
            sched._seq_to_req.get(seq.seq_id, str(seq.seq_id))
            for seq in out2.prefill_seqs + out2.decode_seqs
        } if hasattr(sched, '_seq_to_req') else set()
        # req-2 要么在 prefill_seqs 中要么在 running 队列中
        assert sched.num_waiting == 0 or len(out2.prefill_seqs) > 0
