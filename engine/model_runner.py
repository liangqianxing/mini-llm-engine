"""
模型执行层。

提供两种 ModelRunner：
  1. MockModelRunner   - 纯 Python 模拟，无 GPU 要求，用于调度算法研究和 CI
  2. GPT2ModelRunner   - 真实 GPT-2（HuggingFace），可在 CPU/GPU 上跑

两种 Runner 实现相同接口：
    step(prefill_seqs, decode_seqs) -> Dict[seq_id, new_token_id]

调度器与 Runner 解耦：换模型不需要改调度逻辑。
"""

import time
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .sequence import Sequence


class BaseModelRunner(ABC):
    """ModelRunner 基类，定义公共接口。"""

    @abstractmethod
    def step(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> Dict[int, int]:
        """
        执行一步推理，返回每条序列的下一个 token id。

        Args:
            prefill_seqs: 需要处理完整 prompt 的序列
            decode_seqs:  需要生成下一 token 的序列

        Returns:
            Dict[seq_id → new_token_id]
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MockModelRunner(BaseModelRunner):
    """
    模拟推理引擎（不调用真实模型）。

    用途：
      - 在没有 GPU 的环境下测试调度逻辑
      - 控制变量，专注于调度策略的对比

    模拟行为：
      - prefill 延迟 ∝ prompt token 数
      - decode 延迟固定（类似真实推理的 memory-bound 特性）
      - token 分布可设为均匀随机或固定（用于可复现测试）

    Args:
        prefill_time_per_token: prefill 每个 token 的模拟耗时（秒）
        decode_time_per_step:   decode 每步的模拟耗时（秒）
        eos_token_id:           EOS token id
        eos_probability:        每步随机 EOS 的概率（控制序列长度分布）
        vocab_size:             词表大小
        seed:                   随机种子（None 表示不固定）
    """

    def __init__(
        self,
        prefill_time_per_token: float = 0.0001,  # 0.1ms/token
        decode_time_per_step: float = 0.002,      # 2ms/step
        eos_token_id: int = 50256,
        eos_probability: float = 0.05,            # 平均 ~20 tokens 结束
        vocab_size: int = 50257,
        seed: Optional[int] = 42,
    ) -> None:
        self.prefill_time_per_token = prefill_time_per_token
        self.decode_time_per_step = decode_time_per_step
        self.eos_token_id = eos_token_id
        self.eos_probability = eos_probability
        self.vocab_size = vocab_size
        self._rng = random.Random(seed)

        # 统计
        self.total_prefill_tokens: int = 0
        self.total_decode_tokens: int = 0
        self.total_steps: int = 0

    @property
    def name(self) -> str:
        return "MockModelRunner"

    def step(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> Dict[int, int]:
        """模拟一步推理，返回新 token id 映射。"""
        new_tokens: Dict[int, int] = {}
        self.total_steps += 1

        # prefill：模拟与 prompt 长度成正比的耗时
        if prefill_seqs:
            total_prompt_len = sum(s.num_total_tokens for s in prefill_seqs)
            time.sleep(self.prefill_time_per_token * total_prompt_len)
            self.total_prefill_tokens += total_prompt_len

            for seq in prefill_seqs:
                new_tokens[seq.seq_id] = self._sample_token(seq)

        # decode：所有序列并行，耗时固定（memory-bound）
        if decode_seqs:
            time.sleep(self.decode_time_per_step)
            self.total_decode_tokens += len(decode_seqs)

            for seq in decode_seqs:
                new_tokens[seq.seq_id] = self._sample_token(seq)

        return new_tokens

    def _sample_token(self, seq: Sequence) -> int:
        """采样下一个 token（模拟）。"""
        # 按概率返回 EOS，否则返回随机 token
        if self._rng.random() < self.eos_probability:
            return self.eos_token_id
        return self._rng.randint(0, self.vocab_size - 2)

    def get_stats(self) -> Dict:
        return {
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "total_steps": self.total_steps,
        }


class GPT2ModelRunner(BaseModelRunner):
    """
    真实 GPT-2 推理引擎（基于 HuggingFace Transformers）。

    注意：这里为了展示接口兼容性，采用"一次调用处理一条序列"的简化方式。
    生产级实现（如 vLLM）会将所有序列拼成一个大 batch，并用 PagedAttention
    CUDA kernel 直接操作显存中的物理块；此处用 past_key_values 近似演示。

    Args:
        model_name: HuggingFace 模型名称（默认 gpt2）
        device:     运行设备（"cpu" / "cuda"）
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "GPT2ModelRunner requires: pip install torch transformers"
            )

        self._device = device
        self._model_name = model_name
        print(f"[GPT2ModelRunner] Loading {model_name} on {device} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        # 用字典缓存每条序列的 past_key_values（简化版 KV cache）
        # 真实实现中这里会被 PagedAttention 取代
        self._kv_cache: Dict[int, any] = {}  # seq_id → past_key_values

        import torch as _torch
        self._torch = _torch
        print(f"[GPT2ModelRunner] Ready. (params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M)")

    @property
    def name(self) -> str:
        return f"GPT2ModelRunner({self._model_name})"

    def step(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> Dict[int, int]:
        """执行一步真实推理。"""
        new_tokens: Dict[int, int] = {}

        with self._torch.no_grad():
            # prefill：处理完整 prompt，建立 KV cache
            for seq in prefill_seqs:
                token_ids = self._torch.tensor(
                    [seq.token_ids], dtype=self._torch.long, device=self._device
                )
                outputs = self.model(token_ids, use_cache=True)
                self._kv_cache[seq.seq_id] = outputs.past_key_values
                # 取最后一个位置的 logits，greedy
                next_token = int(outputs.logits[0, -1].argmax())
                new_tokens[seq.seq_id] = next_token

            # decode：只输入最新 token + past_key_values
            for seq in decode_seqs:
                last_token = self._torch.tensor(
                    [[seq.token_ids[-1]]], dtype=self._torch.long, device=self._device
                )
                past = self._kv_cache.get(seq.seq_id)
                outputs = self.model(last_token, past_key_values=past, use_cache=True)
                self._kv_cache[seq.seq_id] = outputs.past_key_values
                next_token = int(outputs.logits[0, -1].argmax())
                new_tokens[seq.seq_id] = next_token

        return new_tokens

    def free_seq(self, seq_id: int) -> None:
        """释放序列的 KV cache（配合调度器使用）。"""
        self._kv_cache.pop(seq_id, None)

    def decode_tokens(self, token_ids: List[int]) -> str:
        """将 token id 列表解码为文本。"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
