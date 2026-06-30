"""
推理引擎指标收集器。

每步记录调度状态、KV cache 利用率、延迟分布，
最终输出 JSON 报告 + 可选的 matplotlib 图表。

支持指标：
  - Throughput     : output tokens / second
  - Latency        : 端到端延迟（p50 / p95 / p99）
  - TTFT           : Time-to-First-Token
  - KV Utilization : 每步 KV cache 占用率
  - Queue depth    : 每步等待 / 运行 / 完成序列数
  - Step stats     : 每步 prefill / decode token 数
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _percentile(data: List[float], p: float) -> float:
    """计算分位数（线性插值）。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


@dataclass
class StepRecord:
    """单步调度的快照。"""
    step: int
    timestamp: float
    num_waiting: int
    num_running: int
    num_finished: int
    kv_utilization: float
    num_prefill_tokens: int
    num_decode_tokens: int


@dataclass
class RequestRecord:
    """单个请求的完成记录。"""
    request_id: str
    prompt_len: int
    output_len: int
    arrival_time: float
    first_token_time: Optional[float]
    finish_time: Optional[float]

    @property
    def latency(self) -> float:
        if self.finish_time is None:
            return 0.0
        return self.finish_time - self.arrival_time

    @property
    def ttft(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float:
        """Time-per-output-token（decode 阶段的平均每 token 延迟）。"""
        if self.output_len <= 1 or self.finish_time is None or self.first_token_time is None:
            return 0.0
        return (self.finish_time - self.first_token_time) / (self.output_len - 1)


class MetricsCollector:
    """
    推理引擎指标收集器。

    用法：
        collector = MetricsCollector()
        # 每步调用
        collector.record_step(step, scheduler_stats, sched_output)
        # 请求完成时调用
        collector.record_request_done(request_output)
        # 最终报告
        report = collector.report()
        collector.plot(save_path="results/metrics.png")
    """

    def __init__(self) -> None:
        self._start_time: float = time.monotonic()
        self._step_records: List[StepRecord] = []
        self._request_records: List[RequestRecord] = []

    # ── 记录 ──────────────────────────────────────────────────────────────────

    def record_step(
        self,
        step: int,
        num_waiting: int,
        num_running: int,
        num_finished: int,
        kv_utilization: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
    ) -> None:
        """记录一步调度状态。"""
        self._step_records.append(StepRecord(
            step=step,
            timestamp=time.monotonic() - self._start_time,
            num_waiting=num_waiting,
            num_running=num_running,
            num_finished=num_finished,
            kv_utilization=kv_utilization,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
        ))

    def record_request_done(
        self,
        request_id: str,
        prompt_len: int,
        output_len: int,
        arrival_time: float,
        first_token_time: Optional[float],
        finish_time: Optional[float],
    ) -> None:
        """记录一个请求的完成情况。"""
        self._request_records.append(RequestRecord(
            request_id=request_id,
            prompt_len=prompt_len,
            output_len=output_len,
            arrival_time=arrival_time,
            first_token_time=first_token_time,
            finish_time=finish_time,
        ))

    # ── 统计计算 ──────────────────────────────────────────────────────────────

    def _latency_stats(self) -> Dict:
        latencies = [r.latency for r in self._request_records if r.latency > 0]
        return {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
        }

    def _ttft_stats(self) -> Dict:
        ttfts = [r.ttft for r in self._request_records if r.ttft > 0]
        return {
            "p50": _percentile(ttfts, 50),
            "p95": _percentile(ttfts, 95),
            "p99": _percentile(ttfts, 99),
            "mean": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        }

    def _tpot_stats(self) -> Dict:
        tpots = [r.tpot for r in self._request_records if r.tpot > 0]
        return {
            "p50": _percentile(tpots, 50),
            "p95": _percentile(tpots, 95),
            "p99": _percentile(tpots, 99),
            "mean": sum(tpots) / len(tpots) if tpots else 0.0,
        }

    def _throughput(self) -> float:
        """系统吞吐量：total output tokens / total wall-clock time。"""
        total_tokens = sum(r.output_len for r in self._request_records)
        total_time = time.monotonic() - self._start_time
        return total_tokens / total_time if total_time > 0 else 0.0

    def _kv_utilization_stats(self) -> Dict:
        utils = [r.kv_utilization for r in self._step_records]
        return {
            "mean": sum(utils) / len(utils) if utils else 0.0,
            "peak": max(utils) if utils else 0.0,
        }

    # ── 报告生成 ──────────────────────────────────────────────────────────────

    def report(self) -> Dict:
        """生成完整性能报告（字典形式）。"""
        return {
            "summary": {
                "total_requests": len(self._request_records),
                "total_steps": len(self._step_records),
                "throughput_tok_s": round(self._throughput(), 2),
                "wall_time_s": round(time.monotonic() - self._start_time, 3),
            },
            "latency_s": self._latency_stats(),
            "ttft_s": self._ttft_stats(),
            "tpot_s": self._tpot_stats(),
            "kv_utilization": self._kv_utilization_stats(),
        }

    def print_report(self) -> None:
        """打印格式化报告到 stdout。"""
        r = self.report()
        s = r["summary"]
        lat = r["latency_s"]
        ttft = r["ttft_s"]
        kv = r["kv_utilization"]

        print("\n" + "─" * 55)
        print("  Performance Metrics")
        print("─" * 55)
        print(f"  Requests        : {s['total_requests']}")
        print(f"  Steps           : {s['total_steps']}")
        print(f"  Throughput      : {s['throughput_tok_s']:.1f} tok/s")
        print(f"  Wall time       : {s['wall_time_s']:.3f}s")
        print(f"  Latency  p50/p95/p99 : {lat['p50']:.3f}s / {lat['p95']:.3f}s / {lat['p99']:.3f}s")
        print(f"  TTFT     p50/p95/p99 : {ttft['p50']:.3f}s / {ttft['p95']:.3f}s / {ttft['p99']:.3f}s")
        print(f"  KV util  mean / peak : {kv['mean']:.1%} / {kv['peak']:.1%}")
        print("─" * 55)

    def save_json(self, path: str) -> None:
        """将报告保存为 JSON。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2)
        print(f"[Metrics] Report saved → {path}")

    def plot(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """生成 4 格指标图。"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Metrics] Install matplotlib for plots: pip install matplotlib")
            return

        steps = [r.step for r in self._step_records]
        timestamps = [r.timestamp for r in self._step_records]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("LLM Engine Performance Metrics", fontsize=14, fontweight="bold")

        # 图 1：KV cache 利用率
        ax1 = axes[0][0]
        ax1.plot(timestamps, [r.kv_utilization for r in self._step_records],
                 color="#2ecc71", linewidth=1.5)
        ax1.set_title("KV Cache Utilization", fontsize=11)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Utilization")
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # 图 2：队列深度
        ax2 = axes[0][1]
        ax2.stackplot(
            timestamps,
            [r.num_running for r in self._step_records],
            [r.num_waiting for r in self._step_records],
            labels=["Running", "Waiting"],
            colors=["#2ecc71", "#e74c3c"],
            alpha=0.7,
        )
        ax2.set_title("Queue Depth Over Time", fontsize=11)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Sequences")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # 图 3：端到端延迟分布
        ax3 = axes[1][0]
        latencies = [r.latency for r in self._request_records if r.latency > 0]
        if latencies:
            ax3.hist(latencies, bins=min(20, len(latencies)), color="#3498db", alpha=0.8,
                     edgecolor="white")
            for p, color in [(50, "#2ecc71"), (95, "#f39c12"), (99, "#e74c3c")]:
                val = _percentile(latencies, p)
                ax3.axvline(val, color=color, linestyle="--", linewidth=1.5,
                            label=f"p{p}={val:.2f}s")
        ax3.set_title("E2E Latency Distribution", fontsize=11)
        ax3.set_xlabel("Latency (s)")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=8)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # 图 4：TTFT 分布
        ax4 = axes[1][1]
        ttfts = [r.ttft for r in self._request_records if r.ttft > 0]
        if ttfts:
            ax4.hist(ttfts, bins=min(20, len(ttfts)), color="#9b59b6", alpha=0.8,
                     edgecolor="white")
            for p, color in [(50, "#2ecc71"), (95, "#f39c12"), (99, "#e74c3c")]:
                val = _percentile(ttfts, p)
                ax4.axvline(val, color=color, linestyle="--", linewidth=1.5,
                            label=f"p{p}={val:.3f}s")
        ax4.set_title("Time-to-First-Token (TTFT) Distribution", fontsize=11)
        ax4.set_xlabel("TTFT (s)")
        ax4.set_ylabel("Count")
        ax4.legend(fontsize=8)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Metrics] Plot saved → {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
