# Based on user-provided Attention code.
# Original copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Transformer-block inference benchmark: Standard MHA vs GPU MA vs offloaded MA.

Dependencies: torch (CUDA/BF16), flash-attn >= 2.1, fla (same as original model).
Run: python bmk.py --mode both --json bmk_results.json
Prefill keeps the original pipeline; decode defaults to one bulk gather/H2D.
Use --decode-offload pipeline to compare with the original decode scheduler.
Default: pre-norm attention + residual, pre-norm GatedMLP + residual.
Parameter totals include untied input embedding, final RMSNorm and LM head.
Timing always includes input embedding, all blocks, final RMSNorm and LM head.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn

from ma_profile import Timeline, span

try:
    import flash_attn
    from flash_attn import flash_attn_func
    from fla.modules import RMSNorm, RotaryEmbedding
    from fla.modules.activations import swiglu, swiglu_linear
except ImportError as exc:
    raise ImportError("Requires flash-attn >= 2.1 and the fla package used by your model") from exc


class _GroupTicket:
    """One group generation. CPU readiness and CUDA completion are distinct."""

    def __init__(self, owner, slot, start):
        self.owner, self.slot, self.start = owner, slot, start
        self.count = min(owner.group, owner.layers - start)
        self.ready = threading.Event()
        self.released = threading.Event()
        self.error = None
        self.waited = False
        self.release_count = 0

    def acquire(self, offset):
        # Called only AFTER the block norm and Q/K kernels have been submitted.
        with span(f"G{self.start}/cpu_ready_wait", gpu=False):
            self.ready.wait()
        if self.error is not None:
            raise RuntimeError("CPU MA prefetch failed") from self.error
        if not self.waited:
            with span(f"G{self.start}/gpu_m_wait", gpu=True):
                torch.cuda.current_stream(self.owner.device).wait_event(self.slot["copied"])
            self.waited = True
        values = self.owner._view(self.slot["gpu"], self.start)
        return values[:, offset].view(self.owner.batch, self.owner.seq_len, self.owner.dim)

    def release(self, offset):
        if offset != self.release_count:
            raise RuntimeError("MA group slices must be consumed in layer order")
        self.release_count += 1
        if self.release_count == self.count:
            # All uses of M are in k + m. Later attention/MLP use the NEW V tensor.
            self.slot["consumed"].record(torch.cuda.current_stream(self.owner.device))
            self.released.set()


class PendingM:
    """Deferred M dependency. acquire immediately before addition, then release."""

    def __init__(self, ticket, offset):
        self.ticket, self.offset = ticket, offset
        self.acquired = self.released = False

    def acquire(self):
        if self.acquired:
            raise RuntimeError("M handle acquired twice")
        value = self.ticket.acquire(self.offset)
        self.acquired = True
        return value

    def release(self):
        if not self.acquired or self.released:
            raise RuntimeError("M handle must be acquired once before release")
        self.ticket.release(self.offset)
        self.released = True


class MAOffloader:
    """Persistent producer: one CPU task per forward, bounded lookahead slots.

    The producer gathers AND submits H2D, independently of block execution.
    consume(layer, PendingM) must resolve the handle at K+M, release it after
    enqueueing that addition, and keep all compute on the calling CUDA stream.
    GPU storage may be overwritten immediately after that addition completes.
    """

    def __init__(self, weights, batch, seq_len, group_size, device="cuda:0", prefetch_depth=4):
        if weights.device.type != "cpu" or weights.ndim != 3 or weights.requires_grad:
            raise ValueError("weights must be detached CPU [vocab, layers, dim]")
        if min(batch, seq_len, group_size, prefetch_depth) < 1:
            raise ValueError("shapes and prefetch depth must be positive")
        self.weights = weights
        self.batch, self.seq_len = batch, seq_len
        self.tokens = batch * seq_len
        self.layers, self.dim = weights.shape[1:]
        self.group = min(group_size, self.layers)
        self.groups = list(range(0, self.layers, self.group))
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ma-prefetch")
        self.lock = threading.Lock()
        self.closed = self.broken = False
        capacity = self.tokens * self.group * self.dim
        self.slots = [dict(host=torch.empty(capacity, dtype=weights.dtype, pin_memory=True),
                           gpu=torch.empty(capacity, dtype=weights.dtype, device=self.device),
                           copied=torch.cuda.Event(), consumed=torch.cuda.Event(), previous=None)
                      for _ in range(min(prefetch_depth, len(self.groups)))]

    def _view(self, flat, start):
        count = min(self.group, self.layers - start)
        return flat[:self.tokens * count * self.dim].view(self.tokens, count, self.dim)

    @torch.inference_mode()
    def _produce(self, ids, tickets, entry, cancel):
        try:
            # CUDA current device/stream are thread-local; select them explicitly.
            with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(entry)
                for ticket in tickets:
                    slot = ticket.slot
                    previous = slot["previous"]
                    if previous is not None:
                        with span(f"G{ticket.start}/slot_release_wait", gpu=False):
                            while not previous.released.wait(timeout=0.05):
                                if cancel.is_set():
                                    return
                        if cancel.is_set():
                            return
                        # Prevent CPU mutation of a pinned buffer still used by DMA.
                        with span(f"G{ticket.start}/host_dma_wait", gpu=False):
                            slot["copied"].synchronize()
                    if cancel.is_set():
                        return
                    source = self.weights[:, ticket.start:ticket.start + ticket.count]
                    with span(f"G{ticket.start}/cpu_gather", gpu=False):
                        torch.index_select(source, 0, ids, out=self._view(slot["host"], ticket.start))
                    if previous is not None:
                        # Snapshot the prior use before the event can be re-recorded.
                        with span(f"G{ticket.start}/gpu_slot_wait", gpu=True):
                            self.copy_stream.wait_event(slot["consumed"])
                    with span(f"G{ticket.start}/h2d", gpu=True):
                        self._view(slot["gpu"], ticket.start).copy_(
                            self._view(slot["host"], ticket.start), non_blocking=True)
                    slot["copied"].record(self.copy_stream)
                    slot["previous"] = ticket
                    # Ready means copy/its event are SUBMITTED, not completed.
                    ticket.ready.set()
        except BaseException as exc:
            for ticket in tickets:
                if not ticket.ready.is_set():
                    ticket.error = exc
                    ticket.ready.set()
            raise

    @torch.inference_mode()
    def forward(self, ids_cpu, consume):
        if self.closed or self.broken:
            raise RuntimeError("offloader is closed or failed; construct a new one")
        if ids_cpu.device.type != "cpu" or ids_cpu.dtype != torch.long:
            raise ValueError("ids must be CPU int64")
        if tuple(ids_cpu.shape) != (self.batch, self.seq_len):
            raise ValueError("IDs do not match the preallocated shapes")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("concurrent forwards on one offloader are unsupported")
        cancel = threading.Event()
        future = None
        try:
            ids = ids_cpu.reshape(-1).contiguous()
            entry = torch.cuda.Event()
            entry.record(torch.cuda.current_stream(self.device))
            tickets = [_GroupTicket(self, self.slots[i % len(self.slots)], start)
                       for i, start in enumerate(self.groups)]
            with span("offload/producer_submit"):
                future = self.worker.submit(self._produce, ids, tickets, entry, cancel)
            for ticket in tickets:
                for offset in range(ticket.count):
                    handle = PendingM(ticket, offset)
                    # No Future.result(), CPU ready wait or CUDA wait before Q/K.
                    consume(ticket.start + offset, handle)
                    if not handle.released:
                        raise RuntimeError("consumer did not acquire/release its PendingM")
            with span("offload/producer_done_wait"):
                future.result()
        except BaseException:
            self.broken = True
            cancel.set()
            if future is not None:
                try:
                    future.result()
                except BaseException:
                    pass
            raise
        finally:
            self.lock.release()

    def close(self):
        self.closed = True
        self.worker.shutdown(wait=True)
        torch.cuda.synchronize(self.device)


class _BulkTicket:
    """Only first/last layers need a handle; middle layers use persistent views."""

    def __init__(self, owner):
        self.owner = owner

    def acquire(self, offset):
        if offset == 0:
            with span("G0/gpu_m_wait", gpu=True):
                torch.cuda.current_stream(self.owner.device).wait_event(self.owner.copied)
        return self.owner.values[offset]

    def release(self, offset):
        if offset == self.owner.layers - 1:
            self.owner.consumed.record(torch.cuda.current_stream(self.owner.device))


class BulkMAOffloader:
    """Small-input path: one CPU gather and one H2D, no producer thread.

    CPU gather remains inside the measured forward. It can overlap previously
    submitted embedding work, but blocks submission of the following Q/K.
    This trades startup latency for much less per-layer coordination.
    """

    policy = "bulk"

    def __init__(self, weights, batch, seq_len, device):
        self.weights = weights
        self.batch, self.seq_len = batch, seq_len
        self.layers, self.dim = weights.shape[1:]
        self.group = self.layers
        self.device = torch.device(device)
        shape = (batch * seq_len, self.layers, self.dim)
        self.host = torch.empty(shape, dtype=weights.dtype, pin_memory=True)
        self.gpu = torch.empty(shape, dtype=weights.dtype, device=self.device)
        self.values = [self.gpu[:, i].view(batch, seq_len, self.dim)
                       for i in range(self.layers)]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # Establish allocation-stream ordering once, before any asynchronous use.
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.copied = torch.cuda.Event()
        self.consumed = torch.cuda.Event()
        self.slots = [{"host": self.host, "gpu": self.gpu}]
        self.started = False
        self.closed = self.broken = False
        self.lock = threading.Lock()

    @torch.inference_mode()
    def forward(self, ids_cpu, consume):
        if self.closed or self.broken:
            raise RuntimeError("offloader is closed or failed")
        if ids_cpu.device.type != "cpu" or ids_cpu.dtype != torch.long:
            raise ValueError("ids must be CPU int64")
        if tuple(ids_cpu.shape) != (self.batch, self.seq_len):
            raise ValueError("IDs do not match the preallocated shapes")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("concurrent forwards are unsupported")
        try:
            if self.started and not self.copied.query():
                with span("G0/host_dma_wait"):
                    self.copied.synchronize()
            # No token/result cache: every invocation performs its real lookup.
            with span("G0/cpu_gather"):
                torch.index_select(self.weights, 0, ids_cpu.reshape(-1), out=self.host)
            with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                if self.started:
                    with span("G0/gpu_slot_wait", gpu=True):
                        self.copy_stream.wait_event(self.consumed)
                with span("G0/h2d", gpu=True):
                    self.gpu.copy_(self.host, non_blocking=True)
                self.copied.record(self.copy_stream)
            ticket = _BulkTicket(self)
            for layer in range(self.layers):
                if layer == 0 or layer == self.layers - 1:
                    handle = PendingM(ticket, layer)
                    consume(layer, handle)
                    if not handle.released:
                        raise RuntimeError("consumer did not release M")
                else:
                    consume(layer, self.values[layer])
            self.started = True
        except BaseException:
            self.broken = True
            raise
        finally:
            self.lock.release()

    def close(self):
        self.closed = True
        torch.cuda.synchronize(self.device)


def make_offloader(weights, batch, seq_len, group, args, device, *, decode=False):
    if decode and args.decode_offload == "bulk":
        return BulkMAOffloader(weights, batch, seq_len, device)
    loader = MAOffloader(weights, batch, seq_len, group, device,
                         prefetch_depth=args.prefetch_depth)
    loader.policy = "pipeline"
    return loader


def tensor_mib(tensor):
    return tensor.numel() * tensor.element_size() / 2**20


def parameter_placement(stack):
    """Full constructed model weights, including folded non-Parameter tables.

    The CPU table is a shared initialization source for GPU MA, not an active
    CPU model weight in standard/ma_gpu. Count it only for ma_offload.
    Excludes caches, activations, transfer buffers and deterministic RoPE buffers.
    """
    tensors = list(stack.parameters())
    if stack.grouped_table is not None:
        tensors.append(stack.grouped_table)
    if stack.variant == "ma_offload":
        tensors.append(stack.table)
    counts = {"gpu": 0, "cpu": 0}
    byte_counts = {"gpu": 0, "cpu": 0}
    seen = set()
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        location = "gpu" if tensor.device.type == "cuda" else "cpu"
        counts[location] += tensor.numel()
        byte_counts[location] += tensor.numel() * tensor.element_size()
    return dict(gpu_parameters=counts["gpu"], cpu_parameters=counts["cpu"],
                total_parameters=sum(counts.values()),
                gpu_parameter_mib=byte_counts["gpu"] / 2**20,
                cpu_parameter_mib=byte_counts["cpu"] / 2**20)


class StaticKVCache:
    """Preallocated inference cache. Lengths are Python ints, one per layer.

    reset(length) does not erase data: it changes the valid prefix length.
    This avoids measuring concatenation/reallocation instead of attention.
    It intentionally replaces the external fla Cache used by the original code.
    """

    def __init__(self, layers, batch, capacity, kv_heads, head_dim, device, dtype):
        shape = (batch, capacity, kv_heads, head_dim)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)]
        self.values = [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)]
        self.lengths = [0] * layers
        self.capacity = capacity

    def reset(self, length=0):
        if not 0 <= length <= self.capacity:
            raise ValueError("invalid cache length")
        self.lengths[:] = [length] * len(self.lengths)

    def get_seq_length(self, layer):
        return self.lengths[layer]

    def append(self, layer, k, v):
        start = self.lengths[layer]
        end = start + k.shape[1]
        if end > self.capacity:
            raise ValueError("cache capacity exceeded")
        self.keys[layer][:, start:end].copy_(k)
        self.values[layer][:, start:end].copy_(v)
        self.lengths[layer] = end
        return self.keys[layer][:, :end], self.values[layer][:, :end]


class Attention(nn.Module):
    """Dense inference attention with standard or memory-augmented values.

    Standard attention: V = v_proj(hidden_states).
    MA attention: V = K_raw + M, where M = m_proj(input_ids) or precomputed M.
    The M table must already include head-wise RMSNorm and its affine weights.
    K_raw is the projection output before Q/K normalization and RoPE.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float = 10000.0,
        max_position_embeddings: int | None = None,
        layer_idx: int = 0,
        attn_type: str = "standard",
        use_gate: bool = False,
        fused_table: torch.Tensor | None = None,
        norm_eps: float | None = None,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if attn_type not in ("standard", "ma_gpu", "ma_offload"):
            raise ValueError("unsupported attn_type")

        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if hidden_size % num_heads or num_heads % kv_heads:
            raise ValueError(
                "hidden_size must be divisible by num_heads; "
                "num_heads must be divisible by num_kv_heads"
            )

        self.attn_type = attn_type
        self.head_dim = hidden_size // num_heads
        self.kv_dim = kv_heads * self.head_dim
        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings

        factory_kwargs = {"device": device, "dtype": dtype}
        norm_kwargs = {} if norm_eps is None else {"eps": norm_eps}

        # Preserve initialization order so common weights match across variants.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias, **factory_kwargs)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, **factory_kwargs)
        self.q_norm = (
            RMSNorm(self.head_dim, dtype=torch.float32, **norm_kwargs).to(device)
            if qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, dtype=torch.float32, **norm_kwargs).to(device)
            if qk_norm else None
        )
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta).to(device)
        self.gate = (
            nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
            if use_gate else None
        )

        self.v_proj: nn.Linear | None = None
        self.m_proj: nn.Embedding | None = None
        if attn_type == "standard":
            self.v_proj = nn.Linear(
                hidden_size, self.kv_dim, bias=qkv_bias, **factory_kwargs
            )
        elif attn_type == "ma_gpu":
            if fused_table is None or fused_table.shape[1] != self.kv_dim:
                raise ValueError("ma_gpu requires a fused [vocab, kv_dim] table")
            weight = fused_table.contiguous().to(device=device, dtype=dtype)
            self.m_proj = nn.Embedding.from_pretrained(weight, freeze=True)
        # Grouped GPU lookup and CPU offload supply M through precomputed_m.

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        precomputed_m: torch.Tensor | PendingM | None = None,
        cache: StaticKVCache | None = None,
    ) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("use eval() and torch.inference_mode(); this is inference-only")

        batch_size, seq_len, _ = hidden_states.shape
        with span(f"L{self.layer_idx}/q_proj", gpu=True):
            q = self.q_proj(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
        with span(f"L{self.layer_idx}/k_proj", gpu=True):
            k = self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )

        if self.attn_type == "standard":
            with span(f"L{self.layer_idx}/v_proj", gpu=True):
                v = self.v_proj(hidden_states).view(
                    batch_size, seq_len, self.num_kv_heads, self.head_dim
                )
        else:
            pending_m = precomputed_m if isinstance(precomputed_m, PendingM) else None
            if pending_m is not None:
                # Resolve the dependency only after Q/K kernels are submitted.
                precomputed_m = pending_m.acquire()
            if precomputed_m is None:
                if self.attn_type != "ma_gpu" or input_ids is None:
                    raise ValueError("MA requires GPU input_ids or precomputed fused M")
                with span(f"L{self.layer_idx}/m_lookup", gpu=True):
                    precomputed_m = self.m_proj(input_ids)

            m = precomputed_m.reshape(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
            with span(f"L{self.layer_idx}/k_plus_m", gpu=True):
                v = k + m  # Use raw K; allocate V separately from K and the M buffer.
            if pending_m is not None:
                pending_m.release()

        if self.q_norm is not None:
            with span(f"L{self.layer_idx}/qk_norm", gpu=True):
                q, k = self.q_norm(q), self.k_norm(k)

        offset = 0 if cache is None else cache.get_seq_length(self.layer_idx)
        max_seq_len = max(offset + seq_len, self.max_position_embeddings or 0)
        with span(f"L{self.layer_idx}/rope", gpu=True):
            q, k = self.rotary(q, k, seqlen_offset=offset, max_seqlen=max_seq_len)
        if cache is not None:
            with span(f"L{self.layer_idx}/kv_cache", gpu=True):
                k, v = cache.append(self.layer_idx, k, v)

        with span(f"L{self.layer_idx}/flash_attention", gpu=True):
            out = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                dropout_p=0.0,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
        out = out.reshape(batch_size, seq_len, -1)
        if self.gate is not None:
            with span(f"L{self.layer_idx}/gate", gpu=True):
                out = torch.sigmoid(self.gate(hidden_states)) * out
        with span(f"L{self.layer_idx}/o_proj", gpu=True):
            return self.o_proj(out)


@torch.inference_mode()
def make_fused_table(args, device):
    """Synthetic weights, folded with the same fla RMSNorm used by the model.

    For a checkpoint, replace raw rows and norm parameters with that layer's
    trained m_proj and v_norm state. Never normalize across the full kv_dim.
    """
    head_dim = args.hidden_size // args.num_heads
    kv_dim = args.num_kv_heads * head_dim
    table = torch.empty(args.vocab_size, args.num_layers, kv_dim, dtype=torch.bfloat16)
    norm_kw = {} if args.norm_eps is None else dict(eps=args.norm_eps)
    for layer in range(args.num_layers):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + 9000 + layer)
        norm = RMSNorm(head_dim, dtype=torch.float32, **norm_kw).to(device).eval()
        actual_eps = norm.eps
        # Synthetic per-layer affine parameters, folded offline.
        norm.weight.copy_((1 + 0.1 * torch.randn(head_dim, generator=gen)).to(device))
        for start in range(0, args.vocab_size, args.fold_chunk):
            count = min(args.fold_chunk, args.vocab_size - start)
            raw = torch.randn(count, args.num_kv_heads, head_dim,
                              generator=gen, dtype=torch.bfloat16).to(device)
            folded = norm(raw).to(torch.bfloat16)
            table[start:start + count, layer].copy_(folded.flatten(1).cpu())
        del norm, raw, folded
    print(f"MA table ready: head-wise RMSNorm folded, eps={actual_eps}", flush=True)
    return table


class SwiGLULinear(nn.Module):
    def forward(self, gate, up, weight, bias):
        return swiglu_linear(gate, up, weight, bias)


class GatedMLP(nn.Module):
    """Gated MLP with optional fused SwiGLU and output projection."""

    def __init__(self, hidden_size, hidden_ratio=None, intermediate_size=None,
                 hidden_act="swish", fuse_swiglu=True, device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        if hidden_act != "swish":
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu
        kw = dict(device=device, dtype=dtype, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, **kw)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, **kw)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, **kw)
        if fuse_swiglu:
            self.swiglu_linear = SwiGLULinear()

    def forward(self, x, **kwargs):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return self.swiglu_linear(gate, up, self.down_proj.weight, self.down_proj.bias)
        return self.down_proj(swiglu(gate, up))


class AttentionStack(nn.Module):
    def __init__(self, args, variant, table, device):
        super().__init__()
        self.variant, self.table = variant, table
        self.group_size = args.group_size
        self.grouped_table = table.to(device) if variant == "ma_gpu" and args.gpu_ma_lookup == "grouped" else None
        self.layers = nn.ModuleList()
        for index in range(args.num_layers):
            # Same Q/K/O/gate parameters in every variant, irrespective of V RNG.
            torch.manual_seed(args.seed + index)
            module = Attention(hidden_size=args.hidden_size, num_heads=args.num_heads,
                               num_kv_heads=args.num_kv_heads, qkv_bias=args.qkv_bias,
                               qk_norm=args.qk_norm, window_size=args.window_size,
                               rope_theta=args.rope_theta, layer_idx=index,
                               max_position_embeddings=max(args.seq_len, args.context_len + 1),
                               attn_type="ma_offload" if self.grouped_table is not None else variant,
                               use_gate=args.use_gate,
                               fused_table=table[:, index] if variant == "ma_gpu" and self.grouped_table is None else None,
                               norm_eps=args.norm_eps, device=device)
            self.layers.append(module)
        self.mlps = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        norm_kw = {} if args.norm_eps is None else dict(eps=args.norm_eps)
        for index in range(args.num_layers):
            torch.manual_seed(args.seed + 5000 + index)
            self.mlps.append(GatedMLP(args.hidden_size, args.hidden_ratio,
                                      args.intermediate_size,
                                      fuse_swiglu=not args.no_fuse_swiglu,
                                      device=device))
            self.attn_norms.append(RMSNorm(args.hidden_size, dtype=torch.float32, **norm_kw).to(device))
            self.mlp_norms.append(RMSNorm(args.hidden_size, dtype=torch.float32, **norm_kw).to(device))
        # Actual outer modules from the supplied TransformerForCausalLM structure.
        # Included in both parameter counts and every timed model forward.
        torch.manual_seed(args.seed + 20000)
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size,
                                       padding_idx=args.pad_token_id,
                                       device=device, dtype=torch.bfloat16)
        final_norm_kw = {} if args.norm_eps is None else dict(eps=args.norm_eps)
        self.norm = RMSNorm(args.hidden_size, dtype=torch.float32, **final_norm_kw).to(device)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False,
                                 device=device, dtype=torch.bfloat16)
        # Input embeddings and the output head use independent weights.
        if self.embeddings.weight.data_ptr() == self.lm_head.weight.data_ptr():
            raise AssertionError("input embedding and LM head must not share weights")
        self.eval()

    def forward_model(self, ids_cpu, ids_gpu, cache=None, offloader=None,
                      logits_to_keep=0):
        """Inference wrapper: token embedding -> blocks -> final norm -> LM head.

        logits_to_keep=0 computes all positions, matching the supplied default.
        No training loss or sampling is included.
        """
        with span('model/embedding', gpu=True):
            hidden = self.embeddings(ids_gpu)
        hidden = self(hidden, ids_cpu, ids_gpu, cache=cache,
                      offloader=offloader)
        with span('model/final_norm', gpu=True):
            hidden = self.norm(hidden)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        with span('model/lm_head', gpu=True):
            return self.lm_head(hidden)

    def forward(self, hidden, ids_cpu, ids_gpu, cache=None, offloader=None):
        def consume(index, precomputed_m):
            nonlocal hidden
            residual = hidden
            with span(f"L{index}/attn_norm", gpu=True):
                attn_input = self.attn_norms[index](hidden)
            out = self.layers[index](attn_input, input_ids=ids_gpu,
                                     precomputed_m=precomputed_m, cache=cache)
            with span(f"L{index}/attn_residual", gpu=True):
                hidden = residual + out
            with span(f"L{index}/mlp_norm", gpu=True):
                mlp_input = self.mlp_norms[index](hidden)
            with span(f"L{index}/mlp", gpu=True):
                mlp_out = self.mlps[index](mlp_input)
            with span(f"L{index}/mlp_residual", gpu=True):
                hidden = hidden + mlp_out

        if self.variant == "ma_offload":
            if offloader is None:
                raise ValueError("ma_offload requires a prefetcher")
            offloader.forward(ids_cpu, consume)
        elif self.grouped_table is not None:
            group = min(self.group_size, len(self.layers))
            for start in range(0, len(self.layers), group):
                with span(f"G{start}/m_lookup", gpu=True):
                    values = self.grouped_table[:, start:start + group].index_select(0, ids_gpu.reshape(-1))
                for offset in range(values.shape[1]):
                    consume(start + offset, values[:, offset].view(*ids_cpu.shape, -1))
        else:
            for index in range(len(self.layers)):
                consume(index, None)
        return hidden


@torch.inference_mode()
def fill_prefix(stack, cache, ids_cpu, args, device):
    ids_gpu = ids_cpu.to(device)
    hidden = stack.embeddings(ids_gpu)
    loader = (make_offloader(stack.table, args.batch_size, args.context_len,
                             args.group_size, args, device)
              if stack.variant == "ma_offload" else None)
    try:
        stack(hidden, ids_cpu, ids_gpu, cache=cache, offloader=loader)
    finally:
        if loader is not None:
            loader.close()


@torch.inference_mode()
def benchmark_case(args, variant, table, device):
    stack = AttentionStack(args, variant, table, device)
    results = []
    modes = ["prefill", "decode"] if args.mode == "both" else [args.mode]
    head_dim = args.hidden_size // args.num_heads
    for mode in modes:
        seq = args.seq_len if mode == "prefill" else 1
        capacity = seq if mode == "prefill" else args.context_len + 1
        group = min(args.group_size, args.num_layers)
        cache = StaticKVCache(args.num_layers, args.batch_size, capacity,
                              args.num_kv_heads, head_dim, device, torch.bfloat16)
        offloader = (make_offloader(
            table, args.batch_size, seq, group, args, device, decode=(mode == "decode")
        ) if variant == "ma_offload" else None)
        if offloader is not None:
            group = offloader.group
            print(f"{mode}: offload policy={offloader.policy}, layers/group={group}", flush=True)
        try:
            gen = torch.Generator().manual_seed(args.seed + 100)
            ids_cpu = [torch.randint(args.vocab_size, (args.batch_size, seq), generator=gen)
                       for _ in range(args.id_pool)]
            ids_gpu = [ids.to(device) for ids in ids_cpu]
            if mode == "decode" and args.context_len:
                context_ids = torch.randint(args.vocab_size, (args.batch_size, args.context_len), generator=gen)
                fill_prefix(stack, cache, context_ids, args, device)
                del context_ids
                gc.collect()
            prefix = args.context_len if mode == "decode" else 0

            def run(index):
                # Same context length in every repetition; overwrite the new token slot.
                cache.reset(prefix)
                return stack.forward_model(
                    ids_cpu[index], ids_gpu[index], cache=cache,
                    offloader=offloader, logits_to_keep=args.logits_to_keep,
                )

            if args.check_correctness and offloader is not None:
                reference_loader = MAOffloader(
                    table, args.batch_size, seq, 1, device,
                    prefetch_depth=args.prefetch_depth,
                )
                try:
                    # Exercise reuse with different IDs, then revisit the first input.
                    for check_index in (0, min(1, len(ids_cpu)-1), 0):
                        actual = run(check_index)
                        torch.cuda.synchronize(device)
                        saved_k = [k[:, prefix:prefix+seq].clone() for k in cache.keys]
                        saved_v = [v[:, prefix:prefix+seq].clone() for v in cache.values]
                        cache.reset(prefix)
                        expected = stack.forward_model(
                            ids_cpu[check_index], ids_gpu[check_index], cache=cache,
                            offloader=reference_loader, logits_to_keep=args.logits_to_keep,
                        )
                        torch.cuda.synchronize(device)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for layer, (k, v) in enumerate(zip(saved_k, saved_v)):
                            torch.testing.assert_close(k, cache.keys[layer][:, prefix:prefix+seq], rtol=0, atol=0)
                            torch.testing.assert_close(v, cache.values[layer][:, prefix:prefix+seq], rtol=0, atol=0)
                        del actual, expected, saved_k, saved_v
                    print(f"{mode}/{variant}: exact output and KV checks passed (3 forwards).", flush=True)
                finally:
                    reference_loader.close()

            for i in range(args.warmup):
                run(i % args.id_pool)
            torch.cuda.synchronize(device)
            samples = []
            for r in range(args.rounds):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                for i in range(args.repeats):
                    run((r * args.repeats + i) % args.id_pool)
                    if args.timing == "latency":
                        torch.cuda.synchronize(device)
                torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - begin) * 1000 / args.repeats)
            # Count the entire tested stack's active weights by device.
            result = dict(variant=variant, mode=mode, median_ms=statistics.median(samples),
                          min_ms=min(samples), max_ms=max(samples), round_ms=samples,
                          seq_len=seq, context_len=prefix, group_size=group,
                          benchmark_scope="model",
                          offload_policy=offloader.policy if offloader is not None else None,
                          prefetch_slots=len(offloader.slots) if offloader is not None else 0,
                          offload_gpu_buffer_mib=sum(tensor_mib(s["gpu"]) for s in offloader.slots) if offloader is not None else 0,
                          offload_pinned_mib=sum(tensor_mib(s["host"]) for s in offloader.slots) if offloader is not None else 0,
                          outer_gpu_parameters=sum(p.numel() for module in (stack.embeddings, stack.norm, stack.lm_head)
                                                   for p in module.parameters()),
                          **parameter_placement(stack))
            if args.profile_dir is not None:
                trace_path = args.profile_dir / f"{variant}_{mode}.timeline.json"
                metadata = dict(
                    variant=variant, mode=mode, config=vars(args),
                    gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                    unprofiled_median_ms=result["median_ms"],
                )
                # JSON-safe config, including Path-valued arguments.
                metadata = json.loads(json.dumps(metadata, default=str))
                with Timeline(device, metadata) as timeline:
                    with span("model/forward_cpu"):
                        run(0)
                timeline.save(trace_path)
                print(f"Saved diagnostic timeline: {trace_path}", flush=True)
                try:
                    from ma_profile import render
                    print(f"Saved timeline image: {render(trace_path)}", flush=True)
                except ImportError:
                    print("Install matplotlib to render the saved timeline JSON as an image.")
                if args.kernel_trace:
                    from ma_profile import capture_kernels
                    kernel_path = args.profile_dir / f"{variant}_{mode}.chrome.json"
                    capture_kernels(lambda: run(0), device, kernel_path)
                    print(f"Saved kernel trace: {kernel_path}", flush=True)
            results.append(result)
            print(f"{mode} / {variant}: {result['median_ms']:.3f} ms ({result['min_ms']:.3f}..{result['max_ms']:.3f})", flush=True)
        finally:
            if offloader is not None:
                offloader.close()
        del cache, offloader, ids_cpu, ids_gpu
        gc.collect()
        torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    """Parse and validate benchmark arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["both", "prefill", "decode"], default="both")
    p.add_argument("--variants", nargs="+", choices=["standard", "ma_gpu", "ma_offload"],
                   default=["standard", "ma_gpu", "ma_offload"])
    p.add_argument("--decode-offload", choices=["bulk", "pipeline"], default="bulk",
                   help="decode only: one full-layer gather/H2D, or original pipeline")
    p.add_argument("--check-correctness", action="store_true",
                   help="check logits and written KV against legacy before timing")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--logits-to-keep", type=int, default=0,
                   help="0=all positions (default), 1=last token for generation")
    p.add_argument("--pad-token-id", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--context-len", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--hidden-ratio", type=int, default=4)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--no-fuse-swiglu", action="store_true", help="use swiglu followed by down_proj instead of SwiGLULinear")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--group-size", type=int, default=1, help="layers per group; default: 1 for prefill and decode")
    p.add_argument("--prefetch-depth", type=int, default=4,
                   help="number of buffered layer groups for pipeline; independent of group-size")
    p.add_argument("--gpu-ma-lookup", choices=["layerwise", "grouped"], default="layerwise",
                   help="layerwise matches original module; grouped tests MA prelookup optimization")
    p.add_argument("--qkv-bias", action="store_true")
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--use-gate", action="store_true")
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--norm-eps", type=float, default=None, help="default: installed fla RMSNorm default, as in original")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--fold-chunk", type=int, default=1024)
    p.add_argument("--id-pool", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--timing", choices=["latency", "throughput"], default="latency")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--profile-dir", type=Path, default=None,
                   help="after normal timing, capture one diagnostic forward per variant/mode")
    p.add_argument("--kernel-trace", action="store_true",
                   help="with --profile-dir, also collect a separate PyTorch CPU/CUDA trace")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    if args.kernel_trace and args.profile_dir is None:
        p.error("--kernel-trace requires --profile-dir")
    if args.hidden_ratio <= 0 or (args.intermediate_size is not None and args.intermediate_size <= 0):
        p.error("hidden-ratio and intermediate-size must be positive")
    if args.intermediate_size is None:
        raw_size = int(args.hidden_size * args.hidden_ratio * 2 / 3)
        args.intermediate_size = 256 * ((raw_size + 255) // 256)
    args.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
    for key in ("batch_size", "seq_len", "hidden_size", "num_heads", "num_kv_heads",
                "num_layers", "vocab_size", "cpu_threads", "fold_chunk", "id_pool", "repeats", "rounds"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.context_len < 0 or args.warmup < 0:
        p.error("context-len and warmup must be nonnegative")
    if args.prefetch_depth < 1:
        p.error("prefetch-depth must be positive")
    if args.logits_to_keep < 0:
        p.error("logits-to-keep must be nonnegative")
    if args.pad_token_id is not None and not 0 <= args.pad_token_id < args.vocab_size:
        p.error("pad-token-id must be within the vocabulary")
    if any(value is not None and value <= 0 for value in (args.group_size, args.window_size, args.norm_eps)):
        p.error("group-size, window-size and norm-eps must be positive")
    if args.hidden_size % args.num_heads or args.num_heads % args.num_kv_heads:
        p.error("hidden-size must be divisible by num-heads, which must be divisible by num-kv-heads")
    if args.hidden_size // args.num_heads % 2 or args.hidden_size // args.num_heads > 256:
        p.error("use an even head dimension <= 256 for RoPE/FlashAttention")
    return args


def setup_device(args: argparse.Namespace) -> torch.device:
    """Configure the CUDA device and validate runtime capabilities."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("device must be CUDA")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    version = tuple(int(x) for x in flash_attn.__version__.split(".")[:2])
    if version < (2, 1):
        raise RuntimeError("flash-attn >= 2.1 is required for cached causal attention")
    torch.set_num_threads(args.cpu_threads)
    return device


@torch.inference_mode()
def main() -> None:
    """Build and benchmark the selected attention variants."""
    args = parse_args()
    device = setup_device(args)
    print(f"GPU={torch.cuda.get_device_name(device)} | torch={torch.__version__} | flash-attn={flash_attn.__version__}")
    print(f"layers={args.num_layers}, hidden={args.hidden_size}, heads={args.num_heads}, kv_heads={args.num_kv_heads}, batch={args.batch_size}, timing={args.timing}, GPU MA lookup={args.gpu_ma_lookup}")
    print(f"MLP=GatedMLP, block_style=prenorm, intermediate={args.intermediate_size}, fuse_swiglu={not args.no_fuse_swiglu}")
    print(f"Parameter scope=all constructed modules, untied embeddings; timing scope=model, logits_to_keep={args.logits_to_keep}")
    print(f"Offload: prefill=pipeline, decode={args.decode_offload}, pipeline group={args.group_size}, prefetch_depth={args.prefetch_depth}")
    print("Initializing/folding MA weights...", flush=True)
    table = make_fused_table(args, device) if any(v != "standard" for v in args.variants) else None
    rows = []
    for variant in dict.fromkeys(args.variants):
        print(f"\nBuilding {variant}...", flush=True)
        rows.extend(benchmark_case(args, variant, table, device))
        gc.collect()
        torch.cuda.empty_cache()
    report_results(args, rows)


def report_results(args: argparse.Namespace, rows: list[dict]) -> None:
    """Print benchmark results and optionally save them as JSON."""
    for mode in ("prefill", "decode"):
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        standard = next((row["median_ms"] for row in selected if row["variant"] == "standard"), None)
        print(f"\n{mode}: timing scope=model; ALL model parameters including embedding/final norm/LM head")
        print(f"{'variant':13s} {'ms':>9s} {'Std/time':>9s} {'GPU params M':>13s} {'CPU params M':>13s} {'Total M':>12s} {'GPU wt MiB':>12s} {'CPU wt MiB':>12s}")
        for row in selected:
            ratio = None if standard is None else standard / row["median_ms"]
            row["speedup_vs_standard"] = ratio
            label = "n/a" if ratio is None else f"{ratio:.2f}x"
            print(f"{row['variant']:13s} {row['median_ms']:9.3f} {label:>9s} {row['gpu_parameters']/1e6:13.3f} {row['cpu_parameters']/1e6:13.3f} {row['total_parameters']/1e6:12.3f} {row['gpu_parameter_mib']:12.2f} {row['cpu_parameter_mib']:12.2f}")
    print(f"\nPre-norm blocks with explicit BF16 residual adds; correctness checks={args.check_correctness}.")
    print("Input embedding, final RMSNorm and untied LM head are allocated on GPU and INCLUDED in parameter counts.")
    print("Input embedding, final RMSNorm and LM head are INCLUDED in timing. No loss/backward/optimizer/sampling.")
    print("Timing includes Q/K/V, RoPE, cache writes, FlashAttention, gate/O projection, offload and all enabled block operations.")
    print("Decode uses a real prefilled prefix at fixed context length; each repetition overwrites its next-token slot.")
    print("Both CPU/GPU IDs are ready before timing. Prefix construction, norm folding and token-ID D2H are excluded.")
    print("Parameter counts include input embedding/LM head, Q/K/V/O, MLP, all norms, gates and frozen MA tables; M=1,000,000 elements.")
    print("Weight bytes use actual tensor dtype. KV cache, activations and transfer buffers are not parameters.")
    print("CPU staging copies used to initialize GPU weights are excluded from model parameter counts.")
    for row in rows:
        if row["variant"] == "ma_offload":
            print(f"{row['mode']} offload staging (not parameters): slots={row['prefetch_slots']}, "
                  f"GPU={row['offload_gpu_buffer_mib']:.2f} MiB, pinned CPU={row['offload_pinned_mib']:.2f} MiB")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(config=vars(args), results=rows), indent=2, default=str), encoding="utf-8")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()

