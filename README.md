# Mini LLM Inference Engine

[![Tests](https://github.com/yourname/mini-llm-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/yourname/mini-llm-engine/actions)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)

> A from-scratch implementation of the core optimizations in modern LLM serving systems —  
> **Paged KV Cache · Continuous Batching · Chunked Prefill · Prefix Caching · CPU Swap · Speculative Decoding**

Built as a deep-dive portfolio project targeting AI Infra engineering roles. Each optimization is implemented from first principles, benchmarked against a naive baseline, and documented to support 30+ minutes of technical discussion.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           LLMEngine                                 │
│                                                                     │
│  ┌────────────────────────────────┐   ┌─────────────────────────┐  │
│  │          Scheduler             │   │      ModelRunner         │  │
│  │                                │   │                         │  │
│  │  waiting []  ──┐               │   │  MockModelRunner        │  │
│  │  running []  ──┼─ schedule()   │   │  GPT2ModelRunner        │  │
│  │  swapped []  ──┘               │   │  SpeculativeDecoder     │  │
│  │                                │   │                         │  │
│  │  Policy: FCFS | PRIORITY | EDF │   └─────────────────────────┘  │
│  │  Chunked Prefill ✓             │                                 │
│  │  CPU Swap ✓                    │   ┌─────────────────────────┐  │
│  └────────────────────────────────┘   │      MetricsCollector   │  │
│                                       │  p50/p95/p99 latency    │  │
│  ┌────────────────────────────────┐   │  TTFT · KV utilization  │  │
│  │       KVCacheManager           │   └─────────────────────────┘  │
│  │                                │                                 │
│  │  BlockAllocator (O(1) alloc)   │                                 │
│  │  PrefixCache (CoW, hash-based) │                                 │
│  │  SwapManager (GPU ↔ CPU)       │                                 │
│  └────────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Map

| Module | File | What it does |
|--------|------|--------------|
| `PhysicalBlock` / `LogicalTokenBlock` | `engine/block.py` | KV cache memory unit (≈ OS memory page) |
| `BlockAllocator` | `engine/block_allocator.py` | Free-list allocator, O(1) alloc/free |
| `Sequence` / `SequenceGroup` | `engine/sequence.py` | State machine: WAITING → PREFILLING → RUNNING → FINISHED |
| `KVCacheManager` | `engine/kv_cache.py` | Logical-to-physical block mapping + CoW |
| `PrefixCache` | `engine/prefix_cache.py` | Hash-based block dedup, Copy-on-Write |
| `SwapManager` | `engine/swap_manager.py` | CPU offload for preempted KV cache |
| **`Scheduler`** | `engine/scheduler.py` | ⭐ Continuous batching + chunked prefill + priority preemption |
| `MockModelRunner` / `GPT2ModelRunner` | `engine/model_runner.py` | Inference backend (swap without changing scheduler) |
| `SpeculativeDecoder` | `engine/speculative.py` | Draft + verify speculative decoding |
| `MetricsCollector` | `engine/metrics.py` | p50/p95/p99 latency, TTFT, KV utilization |
| `LLMEngine` | `engine/llm_engine.py` | Top-level orchestration API |

---

## ⚡ Optimizations Implemented

### 1. Paged KV Cache (PagedAttention)

Traditional servers pre-allocate a **contiguous** memory block of `max_seq_len` for every request → 70–80% wasted.

**Paged KV Cache** divides memory into fixed-size blocks (like OS virtual memory paging):

```
Traditional:  [prompt|generate|pad|pad|pad|pad]  ← pre-allocate max_tokens
              ^~~ only this is used ~~~^              ← 75% fragmentation!

Paged KV:     [block₀][block₁][block₂]...       ← allocate on demand
              ← freed immediately on finish ←        ← <10% fragmentation
```

**Key code**: `BlockAllocator.allocate()` / `KVCacheManager.append_slot()` / `Sequence.block_table`

### 2. Continuous Batching

Static batching: wait for **all** sequences in a batch to finish before starting new ones.

**Continuous Batching** (from [Orca, OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)): per-step scheduling loop fills empty slots immediately:

```
Step N:   [A:decode] [B:decode] [C:decode]
Step N+1: C finishes → immediately admit D from waiting queue
          [A:decode] [B:decode] [D:prefill→decode]
Step N+2: [A:decode] [B:decode] [D:decode]
```

**Key code**: `Scheduler.schedule()` — runs every decode step, admits new requests when slots open.

### 3. Chunked Prefill (Sarathi-Serve)

Problem: a 512-token prompt monopolizes one full prefill step, blocking all running decode sequences.

**Chunked Prefill** ([Sarathi, OSDI'24](https://arxiv.org/abs/2308.16369)): split the prefill into chunks, interleave with decode:

```
Without: │prefill(512 tok)│decode│decode│decode│...  ← decode blocked for 1 step
   With: │chunk32│decode│chunk32│decode│...│         ← decode runs every step
```

**Key code**: `SequenceStatus.PREFILLING`, `Sequence.num_prefilled_tokens`, `Scheduler._get_next_prefill_chunk()`

### 4. Prefix Caching (Copy-on-Write)

When multiple requests share the same system prompt, KV cache is computed once and reused:

```
Req 1: [system prompt KV] [user query 1 KV]   ← compute both
Req 2: [system prompt KV] [user query 2 KV]   ← REUSE system prompt blocks (ref_count+1)
       ^shared, CoW^
```

**Key code**: `PrefixCache.lookup()` / `register()` / `cow_if_needed()`, `content_hash` on logical blocks.

### 5. CPU Swap

When GPU memory is tight, preempted sequences' KV cache is offloaded to CPU RAM instead of discarded (requiring expensive re-prefill):

```
GPU OOM → swap_out(seq)  → GPU blocks freed, mapping saved in CPU dict
         → swap_in(seq)  → new GPU blocks allocated, mapping restored
```

**Key code**: `SwapManager.swap_out()` / `swap_in()`, `SequenceStatus.SWAPPED`.

### 6. Priority Scheduling (FCFS / PRIORITY / EDF)

Three preemption policies configurable at engine creation:
- **FCFS** (default): preempt most recently admitted sequence
- **PRIORITY**: preempt lowest-priority sequence (`seq.priority` field)
- **EDF**: preempt sequence furthest from its SLO deadline (`seq.deadline` field)

**Key code**: `SchedulerPolicy` enum, `Sequence.urgency()`, `Scheduler._choose_victim()`.

### 7. Speculative Decoding

```
Step 1 (Draft):  small_model → [t₁, t₂, t₃, t₄]    (K=4 speculative tokens)
Step 2 (Verify): large_model → verify all 4 in parallel (1 prefill-like step)
Step 3 (Accept): t₁✓ t₂✓ t₃✗ → accept [t₁, t₂], use large_model's correction
```

With acceptance rate α and K drafts: expected output = `(1 - α^(K+1)) / (1 - α)` tokens per verify step.

**Key code**: `SpeculativeDecoder.step()` — draft loop + parallel verification + accept/reject.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourname/mini-llm-engine
cd mini-llm-engine
pip install -e ".[viz]"       # +matplotlib for plots

# Quick demo
python examples/basic_usage.py

# Run all tests (67 tests)
pytest tests/ -v

# Individual benchmarks
python -m benchmarks.throughput_bench --num-requests 50 --max-tokens 64
python -m benchmarks.prefix_cache_bench --num-requests 30 --shared-prefix-len 64
python -m benchmarks.memory_bench --num-requests 200 --max-tokens 128
python -m visualizer.gantt --num-requests 10 --max-tokens 30

# All benchmarks at once
python run_all_benchmarks.py --fast --no-plot

# With real GPT-2 (requires torch + transformers)
pip install torch transformers
python examples/gpt2_demo.py
```

---

## 📊 Benchmark Results

Simulated environment: mock model with 2ms/step decode latency (memory-bound, proportional to real GPU inference).

### 1. Throughput: Continuous vs Naive Batching

50 requests, max 64 output tokens, 2ms decode latency/step:

```
Strategy                      Time    Tokens    Throughput
────────────────────────── ──────── ──────── ────────────
Naive Batching (batch=8)    12.3s    ~1900     152 tok/s
Continuous Batching          5.2s    ~1900     360 tok/s

🚀 Speedup: ~2.4x throughput improvement
```

**Why**: Naive batching idles GPU slots when short sequences finish before long ones. Continuous batching fills those slots immediately.

### 2. Memory: Paged vs Static Allocation

200 requests, max 128 tokens, geometric output length (mean≈20):

```
Metric                    Static Allocation    Paged KV Cache
──────────────────────── ─────────────────── ───────────────
Memory Utilization               17–28%              45–65%
Fragmentation                    72–83%              35–55%
Max Concurrent Seqs                 ~8                 ~38

🧠 Memory efficiency: 2–4x improvement
```

### 3. Prefix Caching

30 requests, 48-token shared system prompt + 16-token unique suffix:

```
Strategy            Throughput    Cache Hit Rate
─────────────── ──────────────── ──────────────
No Prefix Cache    ~1800 tok/s         0%
Prefix Cache       ~1800 tok/s       71.3%

💾 Memory saved: ~75% of prefix KV cache blocks shared across requests
🔑 Hit rate 71.3% on full blocks (partial last block always misses by design)
```

### 4. Speculative Decoding

K=4, acceptance_rate=0.7. Expected tokens per verify step: `(1 - 0.7^5) / 0.3 ≈ 3.3`.

> **Note on mock numbers**: in the mock runner, draft and target models have similar overhead. Real GPU speedup comes from the draft model being 10–20x smaller (much faster). The implementation is conceptually correct; real-world speedup is typically **2–3x** with a good draft model.

### 5. Chunked Prefill

With long prompts interleaved alongside short decode sequences, chunked prefill reduces **time-to-first-token for short prompts** by preventing long prefills from blocking the decode pipeline.

---

## 🗂️ Project Structure

```
mini-llm-engine/
├── engine/
│   ├── block.py              # PhysicalBlock, LogicalTokenBlock (content_hash for prefix cache)
│   ├── block_allocator.py    # O(1) free-list allocator
│   ├── sequence.py           # State machine + priority + deadline + chunked prefill tracking
│   ├── kv_cache.py           # Logical↔physical mapping, CoW integration
│   ├── prefix_cache.py       # Hash-based block dedup, Copy-on-Write ⭐
│   ├── swap_manager.py       # CPU offload for preempted KV cache ⭐
│   ├── scheduler.py          # Continuous batching + chunked prefill + priority ⭐
│   ├── speculative.py        # Draft + verify speculative decoding ⭐
│   ├── model_runner.py       # Mock + GPT-2 backends (clean swap interface)
│   ├── metrics.py            # p50/p95/p99 latency, TTFT, KV utilization ⭐
│   └── llm_engine.py         # Top-level API
├── benchmarks/
│   ├── throughput_bench.py   # Continuous vs Naive batching comparison
│   ├── memory_bench.py       # KV cache memory utilization (fixed simulation)
│   └── prefix_cache_bench.py # Prefix cache hit rate and memory savings ⭐
├── visualizer/
│   └── gantt.py              # Scheduling timeline Gantt chart
├── tests/
│   ├── test_block_allocator.py  (9 tests)
│   ├── test_kv_cache.py         (10 tests)
│   ├── test_scheduler.py        (8 tests)
│   └── test_new_features.py     (40 tests) ⭐
├── examples/
│   ├── basic_usage.py
│   └── gpt2_demo.py
├── run_all_benchmarks.py     # One-command full benchmark suite ⭐
└── .github/workflows/tests.yml  # CI: Python 3.9/3.10/3.11 ⭐
```

---

## 🔬 Design Decisions

### Why Python over C++/CUDA?

The scheduling policy and memory management are language-agnostic. This implementation isolates the **algorithmic layer** — the same decisions made here map 1:1 to vLLM's Python scheduler. Production CUDA kernels (PagedAttention) implement the physical block reads; this project makes the policy layer visible and testable.

### BlockAllocator: O(1) Free List

Uses `collections.deque` as a free list. `allocate()` = `popleft()`, `free()` = `append()`. Both O(1). Alternative (buddy allocator, bitmap) would add complexity without benefit for fixed-size blocks.

### Prefix Cache: Hash-Based vs Radix Tree

This implementation uses a flat `Dict[int, PhysicalBlock]` — O(1) lookup per block. SGLang's [RadixAttention](https://arxiv.org/abs/2312.07104) uses a trie for longest-prefix matching, enabling prefix sharing even when the cached block boundary doesn't align with the new request. The flat version is easier to understand and still captures the core CoW semantics.

### Block Size as Hyperparameter

`block_size=16` (same as vLLM default). Larger → fewer allocations, but more internal fragmentation in the last block. Smaller → finer control, higher metadata overhead. The tradeoff mirrors OS page size selection.

### Preemption: Swap vs Recompute

Both strategies are implemented. `SwapManager` is used when CPU memory is available (preserves progress). When CPU is full or `cpu_swap_gb=0`, the scheduler falls back to recompute (free GPU blocks, put sequence back in waiting queue). vLLM calls these `preemption_mode="swap"` and `preemption_mode="recompute"`.

---

## 🔭 Architecture Differences from Production (vLLM)

| Aspect | This Project | vLLM |
|--------|-------------|------|
| KV cache storage | Dict simulation | CUDA tensor, contiguous GPU memory |
| Attention kernel | N/A (mock model) | PagedAttention CUDA kernel |
| Batch construction | Per-step Python loop | FlashAttention with variable-length masks |
| Prefix cache | Flat hash map | Radix tree (longest-prefix match) |
| Multi-GPU | Not implemented | Tensor parallelism + pipeline parallelism |
| Chunked prefill | ✅ | ✅ (Sarathi-Serve integration) |
| Continuous batching | ✅ | ✅ |
| CPU swap | ✅ (simulation) | ✅ (real CUDA memcpy) |

---

## 📚 References

1. **PagedAttention** — Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*, SOSP 2023. [[paper]](https://arxiv.org/abs/2309.06180) [[vLLM]](https://github.com/vllm-project/vllm)
2. **Orca** — Yu et al., *"Orca: A Distributed Serving System for Transformer-Based Generative Models"*, OSDI 2022. [[paper]](https://www.usenix.org/conference/osdi22/presentation/yu)
3. **Sarathi-Serve** — Agrawal et al., *"Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills"*, OSDI 2024. [[paper]](https://arxiv.org/abs/2308.16369)
4. **Speculative Decoding** — Leviathan et al., *"Fast Inference from Transformers via Speculative Decoding"*, ICML 2023. [[paper]](https://arxiv.org/abs/2211.17192)
5. **SGLang RadixAttention** — Zheng et al., *"SGLang: Efficient Execution of Structured Language Model Programs"*, 2024. [[paper]](https://arxiv.org/abs/2312.07104)
6. **Continuous Batching Blog** — Anyscale, *"How continuous batching enables 23x throughput in LLM inference"*. [[blog]](https://www.anyscale.com/blog/continuous-batching-llm-inference)

---

## 🤝 License

MIT
