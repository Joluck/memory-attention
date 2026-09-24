"""Thread-aware CPU/CUDA stage tracing; also renders saved traces without torch.

Usage: python ma_profile.py profiles/ma_offload_decode.timeline.json
CUDA ranges are stream elapsed intervals, not kernel busy time. CPU and GPU
have separate origins; only GPU compute/copy lanes share an exact time base.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
import json
from pathlib import Path
import threading
import time

_active = None
_kernel_trace = False


def span(name, *, gpu=False):
    """Return an inert scope outside the separate profiling pass."""
    if _active is None:
        if _kernel_trace:
            import torch
            return torch.profiler.record_function(name)
        return nullcontext()
    return _active.span(name, gpu=gpu)


class Timeline:
    def __init__(self, device, metadata=None):
        import torch

        self.torch = torch
        self.device = device
        self.metadata = metadata or {}
        self.cpu_records = []
        self.gpu_records = []
        self.lock = threading.Lock()

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError("Nested profiling sessions are unsupported")
        torch = self.torch
        torch.cuda.synchronize(self.device)
        self.main_thread = threading.get_ident()
        self.compute_stream = torch.cuda.current_stream(self.device).cuda_stream
        # Materialize CUDA events before collection; first-record allocation
        # would otherwise inflate many of the short decode stages.
        layers = self.metadata.get("config", {}).get("num_layers", 24)
        self.event_pool = [torch.cuda.Event(enable_timing=True)
                           for _ in range(128 + 64 * layers)]
        for event in self.event_pool:
            event.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record(torch.cuda.current_stream(self.device))
        self.origin.synchronize()
        self.cpu_origin = time.perf_counter_ns()
        # GPU origin precedes CPU origin by event synchronization/host wakeup.
        # Never combine these clocks to compute CPU-to-GPU latency.
        _active = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active
        _active = None
        self.torch.cuda.synchronize(self.device)

    @contextmanager
    def span(self, name, *, gpu=False):
        torch = self.torch
        thread = threading.get_ident()
        cpu_lane = "CPU main" if thread == self.main_thread else "CPU producer"
        if name == "model/forward_cpu":
            cpu_lane = "CPU forward (inclusive)"
        stream = start = end = None
        if gpu:
            stream = torch.cuda.current_stream(self.device)
            with self.lock:
                if len(self.event_pool) < 2:
                    raise RuntimeError("CUDA event pool exhausted; capture one forward per Timeline")
                start, end = self.event_pool.pop(), self.event_pool.pop()
        cpu_start = time.perf_counter_ns()
        if gpu:
            start.record(stream)
        try:
            yield
        finally:
            if gpu:
                end.record(stream)
            cpu_end = time.perf_counter_ns()
            with self.lock:
                self.cpu_records.append({
                    "name": name, "lane": cpu_lane, "domain": "cpu",
                    "start_ms": (cpu_start - self.cpu_origin) / 1e6,
                    "duration_ms": (cpu_end - cpu_start) / 1e6,
                })
                if gpu:
                    lane = ("GPU compute" if stream.cuda_stream == self.compute_stream
                            else f"GPU copy [{stream.cuda_stream}]")
                    self.gpu_records.append((name, lane, start, end))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        events = list(self.cpu_records)
        for name, lane, start, end in self.gpu_records:
            events.append({
                "name": name, "lane": lane, "domain": "gpu",
                "start_ms": self.origin.elapsed_time(start),
                "duration_ms": start.elapsed_time(end),
            })
        events.sort(key=lambda e: (e["domain"], e["start_ms"]))
        payload = {
            "metadata": self.metadata,
            "measurement": (
                "Separate CPU and GPU origins; do not align CPU/GPU timestamps. "
                "All GPU lanes share one CUDA-event clock. GPU durations are "
                "stream elapsed intervals, including possible host submission gaps, "
                "not kernel busy time. Instrumented timings are diagnostic only. "
                "Nested CPU scopes are inclusive; do not sum them as wall time."
            ),
            "events": events,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        groups = defaultdict(list)
        for e in events:
            # Aggregate equivalent stages across layers/groups.
            stage = e["name"].split("/", 1)[-1]
            groups[(e["domain"], e["lane"], stage)].append(e["duration_ms"])
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["domain", "lane", "stage", "count", "total_ms", "mean_ms", "max_ms"])
            for (domain, lane, stage), values in sorted(groups.items()):
                writer.writerow([domain, lane, stage, len(values), sum(values),
                                 sum(values) / len(values), max(values)])
        return path


def capture_kernels(run, device, path):
    """Separate CUPTI-backed pass for actual kernels/copies and CPU operators.

    Worker-thread record_function visibility depends on the PyTorch version;
    use the custom timeline for guaranteed producer-thread CPU scopes.
    """
    global _kernel_trace
    import torch

    torch.cuda.synchronize(device)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False, profile_memory=False,
        ) as prof:
            _kernel_trace = True
            run()
            torch.cuda.synchronize(device)
    finally:
        _kernel_trace = False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not any(e.get("cat") == "kernel" for e in trace.get("traceEvents", [])):
        print("WARNING: kernel trace contains no CUDA kernels; check CUPTI availability.")


def render(path, output=None, start_ms=None, end_ms=None):
    """Render CPU and GPU on distinct axes; zoom values apply to each origin."""
    try:
        import matplotlib
    except ImportError:
        if output is not None and Path(output).suffix.lower() != ".svg":
            raise ImportError("Install matplotlib for PNG output, or use --output timeline.svg")
        return render_svg(path, output, start_ms, end_ms)
    if output is not None and Path(output).suffix.lower() == ".svg":
        return render_svg(path, output, start_ms, end_ms)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data["events"]
    colors = {
        "wait": "#d64550", "gather": "#e7a43b", "h2d": "#29a99b",
        "projection": "#4889cf", "attention": "#9565b8", "mlp": "#5971b8",
        "norm": "#8c9ba8", "other": "#618c70",
    }

    def category(name):
        if "wait" in name: return "wait"
        if "gather" in name or "lookup" in name: return "gather"
        if "h2d" in name: return "h2d"
        if "proj" in name or "lm_head" in name: return "projection"
        if "flash_attention" in name: return "attention"
        if "mlp" in name: return "mlp"
        if "norm" in name: return "norm"
        return "other"

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), constrained_layout=True)
    meta = data.get("metadata", {})
    fig.suptitle(
        f"MA stage timeline | {meta.get('variant', '')} {meta.get('mode', '')}\n"
        "Diagnostic instrumentation; CUDA ranges include possible submission gaps",
        fontsize=14,
    )
    for ax, domain in zip(axes, ("cpu", "gpu")):
        selected = [e for e in events if e["domain"] == domain]
        lanes = sorted({e["lane"] for e in selected})
        extent = max((e["start_ms"] + e["duration_ms"] for e in selected), default=1)
        lo = 0 if start_ms is None else start_ms
        hi = extent if end_ms is None else end_ms
        if hi <= lo:
            hi = lo + 1
        # Put enclosing CPU intervals behind nested detail.
        for e in sorted(selected, key=lambda e: -e["duration_ms"]):
            x, width = e["start_ms"], e["duration_ms"]
            if x + width < lo or x > hi:
                continue
            y = lanes.index(e["lane"])
            ax.broken_barh([(x, max(width, 1e-6))], (y - 0.32, 0.64),
                           facecolors=colors[category(e["name"])], alpha=0.85)
            if width > (hi - lo) * 0.045:
                ax.text(max(x, lo) + 0.002 * (hi - lo), y, e["name"],
                        va="center", fontsize=7, clip_on=True)
        ax.set_yticks(range(len(lanes)), labels=lanes)
        ax.set_ylim(-0.7, max(len(lanes) - 0.3, 0.7))
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel(f"Milliseconds from independent {domain.upper()} origin")
        ax.set_title("CPU wall time (inclusive scopes)" if domain == "cpu" else
                     "GPU stream elapsed time (compute and copy share the same clock)",
                     loc="left", fontsize=11)
    axes[0].legend(handles=[Patch(color=c, label=n) for n, c in colors.items()],
                   ncol=8, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8)
    output = Path(output) if output else path.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def render_svg(path, output=None, start_ms=None, end_ms=None):
    """Dependency-free vector image with tooltips for every stage."""
    from html import escape

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="740" viewBox="0 0 1600 740">',
        '<rect width="1600" height="740" fill="#f7f9fc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#223047} .muted{fill:#596779}</style>',
        '<text x="30" y="36" font-size="22">MA stage timeline</text>',
        f'<text x="30" y="63" font-size="15">{escape(str(meta.get("variant", "")))} / {escape(str(meta.get("mode", "")))}</text>',
        '<text x="30" y="90" font-size="13" class="muted">Diagnostic only: CPU/GPU origins differ. GPU ranges are stream elapsed time, not kernel busy time.</text>',
    ]
    palette = {"wait": "#d64550", "gather": "#e7a43b", "h2d": "#29a99b",
               "projection": "#4889cf", "attention": "#9565b8", "mlp": "#5971b8",
               "norm": "#8c9ba8", "other": "#618c70"}

    def color(name):
        key = ("wait" if "wait" in name else "gather" if "gather" in name or "lookup" in name
               else "h2d" if "h2d" in name else "projection" if "proj" in name or "lm_head" in name
               else "attention" if "flash_attention" in name else "mlp" if "mlp" in name
               else "norm" if "norm" in name else "other")
        return palette[key]

    for panel, domain in enumerate(("cpu", "gpu")):
        top = 150 + panel * 270
        events = [e for e in data["events"] if e["domain"] == domain]
        lanes = sorted({e["lane"] for e in events})
        lo = 0 if start_ms is None else start_ms
        hi = (max((e["start_ms"] + e["duration_ms"] for e in events), default=1)
              if end_ms is None else end_ms)
        hi = max(hi, lo + 0.001)
        scale = 1320 / (hi - lo)
        title = "CPU wall time (inclusive)" if domain == "cpu" else "GPU compute / copy (shared CUDA clock)"
        svg.append(f'<text x="30" y="{top}" font-size="17">{title}</text>')
        for i in range(7):
            x, value = 230 + i * 220, lo + (hi - lo) * i / 6
            svg.append(f'<path d="M{x} {top+22} V{top+166}" stroke="#dce2eb"/>')
            svg.append(f'<text x="{x}" y="{top+190}" text-anchor="middle" font-size="12">{value:.3f}</text>')
        for i, lane in enumerate(lanes):
            y = top + 38 + i * 50
            svg.append(f'<text x="215" y="{y+21}" text-anchor="end" font-size="12">{escape(lane)}</text>')
        for e in sorted(events, key=lambda e: -e["duration_ms"]):
            left, right = max(e["start_ms"], lo), min(e["start_ms"] + e["duration_ms"], hi)
            if right < left:
                continue
            x, width = 230 + (left-lo)*scale, max(0.7, (right-left)*scale)
            y = top + 38 + lanes.index(e["lane"]) * 50
            name = escape(e["name"])
            svg.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="32" fill="{color(e["name"])}" opacity="0.88"><title>{name}: start {e["start_ms"]:.6f} ms; duration {e["duration_ms"]:.6f} ms</title></rect>')
            if width > 130:
                label = escape(e["name"][:int(width / 7)])
                svg.append(f'<text x="{x+3:.2f}" y="{y+21}" font-size="11">{label}</text>')
        svg.append(f'<text x="890" y="{top+220}" text-anchor="middle" font-size="12">Milliseconds from independent {domain.upper()} origin</text>')
    for i, (name, c) in enumerate(palette.items()):
        x = 40 + i * 190
        svg.append(f'<rect x="{x}" y="692" width="15" height="15" fill="{c}"/><text x="{x+23}" y="704" font-size="12">{name}</text>')
    svg.append('</svg>')
    output = Path(output) if output else path.with_suffix('.svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg), encoding='utf-8')
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-ms", type=float)
    parser.add_argument("--end-ms", type=float)
    args = parser.parse_args()
    print(render(args.trace, args.output, args.start_ms, args.end_ms))
