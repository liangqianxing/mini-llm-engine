"""
KV Cache 的基本内存单元定义。

类比虚拟内存：
  - PhysicalBlock       ≈ 物理内存页
  - LogicalTokenBlock   ≈ 虚拟内存页
  - block_table         ≈ 页表（logical → physical 的映射）

每个 block 存放固定数量（block_size）个 token 的 KV cache。
ref_count > 1 时触发 Copy-on-Write（用于 Prefix Caching）。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PhysicalBlock:
    """
    物理 KV Cache Block。

    对应 GPU 显存中的一段固定大小缓冲区，存放 block_size 个 token 的
    Key/Value 张量（所有 Transformer 层共享同一块区域的 block_id 编号）。

    Attributes:
        block_id:       物理块编号（GPU 内存池中的索引）
        ref_count:      引用计数；>1 时表示多个序列共享此块（Copy-on-Write）
        content_hash:   该块所存内容的哈希（用于 Prefix Cache 命中检测）
        is_shared:      是否被前缀缓存引用（=ref_count > 1）
    """
    block_id: int
    ref_count: int = 0
    content_hash: Optional[int] = None  # set after writing tokens, for prefix cache

    @property
    def is_shared(self) -> bool:
        """是否被多个序列共享（Prefix Caching 中的状态）。"""
        return self.ref_count > 1

    def __repr__(self) -> str:
        shared = " shared" if self.is_shared else ""
        return f"PhysBlock(id={self.block_id}, refs={self.ref_count}{shared})"


@dataclass
class LogicalTokenBlock:
    """
    逻辑 Token Block。

    每个 Sequence 维护自己的逻辑块列表；调度器在运行时将逻辑块映射到物理块。
    逻辑块只记录 token id，不持有任何 GPU 内存。

    Attributes:
        block_number: 该序列内的逻辑块编号（0, 1, 2, ...）
        block_size:   每块可容纳的 token 数
        token_ids:    已填入的 token id 列表
    """
    block_number: int
    block_size: int
    token_ids: List[int] = field(default_factory=list)

    # ── 属性 ──────────────────────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) == self.block_size

    @property
    def num_empty_slots(self) -> int:
        return self.block_size - len(self.token_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.token_ids) == 0

    @property
    def content_hash(self) -> Optional[int]:
        """当块满时返回内容哈希（用于 Prefix Cache）；未满时返回 None。"""
        if not self.is_full:
            return None
        return hash(tuple(self.token_ids))

    # ── 操作 ──────────────────────────────────────────────────────────────────

    def append_token(self, token_id: int) -> None:
        """向块尾追加一个 token。"""
        if self.is_full:
            raise ValueError(
                f"LogicalBlock {self.block_number} is full "
                f"(block_size={self.block_size})"
            )
        self.token_ids.append(token_id)

    def append_tokens(self, token_ids: List[int]) -> None:
        """批量追加 tokens（用于 prefill 阶段）。"""
        for tok in token_ids:
            self.append_token(tok)

    def __repr__(self) -> str:
        return (
            f"LogBlock(#{self.block_number}, "
            f"tokens={self.num_tokens}/{self.block_size})"
        )
