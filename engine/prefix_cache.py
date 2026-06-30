"""
Prefix KV Cache（前缀缓存 / Radix Cache）。

核心思路
────────
当多个请求共享相同的 prompt 前缀（例如 system prompt），它们对应的
KV cache 可以共享同一组物理块，而不必重复计算。

实现方式：Copy-on-Write (CoW)
─────────────────────────────
  - 每个"满块"（full block）计算 content_hash = hash(tuple(token_ids))
  - 全局 PrefixCache 维护 hash → PhysicalBlock 的映射
  - 新序列请求 allocate 时，先查 cache：
      命中 → ref_count+1，直接复用该物理块（不需要重新计算）
      未命中 → 分配新物理块，等块满后注册到 cache
  - 对共享块写入（decode 新 token）时触发 CoW：
      复制物理块内容到新块，ref_count 归 1，再写入

性能收益
────────
  - 内存：共享 system prompt 的 N 个请求只需 1 份 KV cache（节省 (N-1)/N）
  - 计算：prefill 阶段跳过已缓存的前缀 token（节省 TTFT）

与 vLLM 的对应关系
────────────────────
  - vLLM >= 0.4.0 的 prefix_caching 选项实现了本文件的核心逻辑
  - SGLang 的 RadixAttention 是更高级的树形版本

Ref: "SGLang: Efficient Execution of Structured Language Model Programs"
     Zheng et al., 2024
"""

from typing import Dict, List, Optional, Tuple

from .block import PhysicalBlock
from .block_allocator import BlockAllocator


class PrefixCache:
    """
    全局 Prefix KV Cache。

    只缓存**完整块**（num_tokens == block_size 的块），
    因为不完整块的内容在 decode 阶段还会变化。

    Args:
        allocator: 同一个 BlockAllocator 实例（共享物理块池）
        max_cached_blocks: 最多缓存多少个物理块（防止 cache 占满所有内存）
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        max_cached_blocks: int = 128,
    ) -> None:
        self.allocator = allocator
        self.max_cached_blocks = max_cached_blocks

        # content_hash → PhysicalBlock（只存满块）
        self._cache: Dict[int, PhysicalBlock] = {}

        # 统计
        self.num_hits: int = 0
        self.num_misses: int = 0
        self.num_evictions: int = 0

    # ── 查询与注册 ────────────────────────────────────────────────────────────

    def lookup(self, content_hash: int) -> Optional[PhysicalBlock]:
        """
        查询 hash 对应的物理块。

        Args:
            content_hash: LogicalTokenBlock.content_hash

        Returns:
            命中时返回 PhysicalBlock（ref_count 已+1）；未命中返回 None。
        """
        block = self._cache.get(content_hash)
        if block is not None:
            block.ref_count += 1
            self.num_hits += 1
        else:
            self.num_misses += 1
        return block

    def register(self, content_hash: int, block: PhysicalBlock) -> None:
        """
        将一个满块注册到 prefix cache。

        Args:
            content_hash: 块内容哈希
            block:        对应的物理块（必须已分配）
        """
        if content_hash in self._cache:
            return  # 已注册，不重复
        if len(self._cache) >= self.max_cached_blocks:
            self._evict_one()
        block.content_hash = content_hash
        self._cache[content_hash] = block

    def invalidate(self, content_hash: int) -> None:
        """使某个缓存条目失效。"""
        self._cache.pop(content_hash, None)

    # ── Copy-on-Write ─────────────────────────────────────────────────────────

    def cow_if_needed(self, block: PhysicalBlock) -> Tuple[PhysicalBlock, bool]:
        """
        如果块被多个序列共享（ref_count > 1），执行 Copy-on-Write：
          1. 分配一个新物理块
          2. 将旧块的 ref_count -1
          3. 返回新块（ref_count=1）

        Args:
            block: 即将被写入的物理块

        Returns:
            (result_block, did_cow)
            - result_block: 可安全写入的物理块（可能是新块）
            - did_cow: 是否发生了 CoW
        """
        if block.ref_count <= 1:
            return block, False  # 独占，无需 CoW

        # 触发 CoW
        new_block = self.allocator.allocate()
        block.ref_count -= 1  # 原块减少引用

        # 如果原块是 prefix cache 的来源，不要从 cache 移除
        # （其他序列还在用它）
        return new_block, True

    # ── 缓存淘汰 ─────────────────────────────────────────────────────────────

    def _evict_one(self) -> None:
        """
        淘汰一个缓存条目。

        策略：优先淘汰 ref_count == 1（只有 prefix cache 自己在引用）的块。
        若全部 ref_count > 1，随机淘汰一个（保守策略）。
        """
        evict_hash = None
        for h, block in self._cache.items():
            if block.ref_count == 1:
                evict_hash = h
                break
        if evict_hash is None and self._cache:
            evict_hash = next(iter(self._cache))

        if evict_hash is not None:
            evicted_block = self._cache.pop(evict_hash)
            evicted_block.content_hash = None
            self.allocator.free(evicted_block)
            self.num_evictions += 1

    # ── 统计 ──────────────────────────────────────────────────────────────────

    @property
    def num_cached_blocks(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.num_hits + self.num_misses
        return self.num_hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        return {
            "num_cached_blocks": self.num_cached_blocks,
            "num_hits": self.num_hits,
            "num_misses": self.num_misses,
            "num_evictions": self.num_evictions,
            "hit_rate": self.hit_rate,
        }

    def __repr__(self) -> str:
        return (
            f"PrefixCache("
            f"cached={self.num_cached_blocks}/{self.max_cached_blocks}, "
            f"hit_rate={self.hit_rate:.1%})"
        )
