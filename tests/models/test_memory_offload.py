# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Regression coverage for offload placement, outputs and buffer lifetimes."""

import pytest
import torch

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM


def _build(seed=1234, layers=4, hidden=512, heads=8, vocab=2048):
    torch.manual_seed(seed)
    config = MemoryConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_heads=heads,
        num_kv_heads=heads,
        vocab_size=vocab,
        qk_norm=True,
        use_gate=True,
        fuse_norm=False,
        use_cache=False,
    )
    model = MemoryForCausalLM(config).to("cuda:0", dtype=torch.bfloat16).eval()
    return model, config


@torch.inference_mode()
def _resident_prefill(model, ids):
    """Capture an independent folded resident reference for every case."""
    model.close_memory_offload()
    model.fold_memory_table_on_gpu(dtype=torch.bfloat16)
    try:
        return model(input_ids=ids, use_cache=False).logits.float().clone()
    finally:
        model.close_memory_offload()


@torch.inference_mode()
def _run_case(policy, batch, seq_len, model, ids, ref):
    model.close_memory_offload()
    model.enable_memory_offload(device="cuda:0", dtype=torch.bfloat16, fold_norm=True)
    model.config.memory_offload_policy = policy
    model.set_offload_offloader(batch, seq_len)

    out = model(input_ids=ids, use_cache=False).logits.float()
    diff = (out - ref).abs()
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item() * 100

    return diff.max().item(), agree


def _storage_bytes(tensors):
    """Count backing storage once, including aliased staging views."""
    seen = {}
    for tensor in tensors:
        if tensor is not None:
            storage = tensor.untyped_storage()
            seen[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


@pytest.fixture
def model():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    pytest.importorskip('flash_attn')
    with torch.inference_mode():
        instance, _ = _build(layers=3, hidden=128, heads=2, vocab=128)
        # Non-unit affine weights expose accidental repeated normalization.
        for layer in instance.model.layers:
            layer.attn.m_norm.weight.copy_(
                torch.linspace(0.5, 1.5, layer.attn.head_dim, device='cuda', dtype=torch.bfloat16)
            )
        try:
            yield instance
        finally:
            instance.close_memory_offload()


@pytest.mark.parametrize('policy', ['bulk', 'pipeline'])
@pytest.mark.parametrize('return_dict', [True, False])
def test_offloaded_hidden_states(model, policy, return_dict):
    ids = torch.randint(0, 128, (2, 8), device='cuda')
    model.fold_memory_table_on_gpu()
    ref = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    model.close_memory_offload()
    model.config.memory_offload_policy = policy
    model.config.memory_offload_group_size = 2
    model.config.memory_offload_prefetch_depth = 1
    model.enable_memory_offload()
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=return_dict)
    logits, hidden = (out.logits, out.hidden_states) if return_dict else out
    assert len(hidden) == len(model.model.layers) + 1
    torch.testing.assert_close(logits, ref.logits, rtol=0, atol=0)
    for actual, expected in zip(hidden, ref.hidden_states):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fold_close_restores_weights_logits_and_device(model):
    ids = torch.randint(0, 128, (2, 8), device='cuda')
    original = [layer.attn.m_proj.weight.clone() for layer in model.model.layers]
    ref = model(input_ids=ids, use_cache=False).logits.clone()
    for _ in range(2):
        model.fold_memory_table_on_gpu()
        with pytest.raises(RuntimeError, match='already folded'):
            model.fold_memory_table_on_gpu()
        with pytest.raises(RuntimeError, match='folded table'):
            model.enable_memory_offload()
        model.close_memory_offload()
        for layer, weight in zip(model.model.layers, original):
            assert not layer.attn.memory_table_folded
            assert layer.attn.m_proj.weight.device == weight.device
            assert layer.attn.m_proj.weight.dtype == weight.dtype
            torch.testing.assert_close(layer.attn.m_proj.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, ref, rtol=0, atol=0)
        model.close_memory_offload()
        assert all(layer.attn.m_proj.weight.is_cuda for layer in model.model.layers)


def test_each_prefill_case_uses_resident_reference(model):
    for policy, batch, length in [('bulk', 2, 8), ('pipeline', 2, 8), ('pipeline', 1, 16)]:
        ids = torch.randint(0, 128, (batch, length), device='cuda')
        # After the first iteration the model is still offloaded. The helper
        # must restore it before producing the next independent reference.
        ref = _resident_prefill(model, ids)
        assert model.model.memory_table is None
        assert all(not layer.attn.memory_table_folded for layer in model.model.layers)
        diff, agree = _run_case(policy, batch, length, model, ids, ref)
        assert diff == 0
        assert agree == 100


@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('seed', [1234, 4321])
@pytest.mark.parametrize('policy', ['bulk', 'pipeline', 'auto'])
def test_growing_cache_exact_with_slot_reuse(seed, policy, padded):
    """128 growing steps, non-unit norms, partial groups, GQA and padding."""
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from fla.models.memory.configuration_memory import MemoryConfig
    from fla.models.memory.modeling_memory import MemoryForCausalLM
    torch.manual_seed(seed)
    config = MemoryConfig(hidden_size=128, num_hidden_layers=3, num_heads=4,
                          num_kv_heads=2, vocab_size=128, qk_norm=True,
                          use_gate=True, fuse_norm=False,
                          memory_offload_policy=policy,
                          memory_offload_mapped_bulk_min_tokens=1,
                          memory_offload_mapped_bulk_max_tokens=1024,
                          memory_offload_group_size=2,
                          memory_offload_bulk_max_tokens=1,
                          memory_offload_prefetch_depth=4 if policy == 'auto' else 1)
    with torch.inference_mode():
        model = MemoryForCausalLM(config).to('cuda', dtype=torch.bfloat16).eval()
        for layer in model.model.layers:
            layer.attn.m_norm.weight.copy_(torch.linspace(.5, 1.5, 32, device='cuda'))
        prefix = torch.randint(0, 128, (2, 17), device='cuda')
        tokens = torch.randint(0, 128, (128, 2, 1), device='cuda').unbind()
        mask = torch.ones_like(prefix)
        if padded:
            mask[0, :5] = 0

        def trajectory():
            records, cache = [], None
            for index, token in enumerate((prefix, *tokens)):
                current_mask = torch.cat((mask, torch.ones((2, index), device='cuda', dtype=mask.dtype)), dim=1)
                out = model(input_ids=token, attention_mask=current_mask,
                            past_key_values=cache, use_cache=True,
                            output_hidden_states=True, logits_to_keep=1)
                cache = out.past_key_values
                tensors = [out.logits, *out.hidden_states]
                # Rotary skips writes at negative (left-pad) positions. Those
                # uninitialized K cells are excluded by attention_mask forever.
                # Compare every usable KV element; unpadded cases cover all cells.
                tensors += [t[current_mask.bool()] for state in cache for t in state['attn_state']]
                records.append([t.clone() for t in tensors])
            return records

        try:
            model.fold_memory_table_on_gpu()
            reference = trajectory()
            model.close_memory_offload()
            model.enable_memory_offload()
            for _ in range(2):
                actual = trajectory()
                for expected_step, actual_step in zip(reference, actual):
                    assert len(expected_step) == len(actual_step)
                    for expected, value in zip(expected_step, actual_step):
                        assert torch.isfinite(value).all()
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        finally:
            model.close_memory_offload()


def test_auto_policy_boundary_and_shape_reuse(model):
    model.config.memory_offload_policy = 'auto'
    assert model.config.memory_offload_bulk_max_tokens == 1024
    for length, policy in [(1024, 'bulk'), (1025, 'pipeline'), (1, 'bulk'), (1025, 'pipeline')]:
        ids = torch.randint(0, 128, (1, length), device='cuda')
        ref = _resident_prefill(model, ids)
        model.enable_memory_offload()
        for _ in range(2):
            actual = model(input_ids=ids, use_cache=False).logits.float()
            assert model.model.memory_offloader.policy == policy
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize('policy', ['auto', 'pipeline'])
def test_selective_depth_boundary_capacity_and_exactness(model, policy):
    model.config.memory_offload_policy = policy
    assert model.config.memory_offload_single_slot_max_tokens == 2048
    model.config.memory_offload_prefetch_depth = 4
    for batch, length in [(1, 1024), (1, 1025), (1, 2048), (1, 2049), (2, 1025), (1, 1025)]:
        ids = torch.randint(0, 128, (batch, length), device='cuda')
        expected = _resident_prefill(model, ids)
        model.enable_memory_offload()
        for _ in range(2):
            actual = model(input_ids=ids, use_cache=False).logits.float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        off = model.model.memory_offloader
        if policy == 'auto' and batch * length <= 1024:
            assert off.policy == 'bulk'
        else:
            slots = 1 if policy == 'auto' and batch * length <= 2048 else min(4, len(model.model.layers))
            assert off.policy == 'pipeline' and len(off.slots) == slots
            expected_bytes = slots * batch * length * off.group * off.dim * off.weights.element_size()
            assert _storage_bytes(slot['host'] for slot in off.slots) == expected_bytes
            assert _storage_bytes(slot['gpu'] for slot in off.slots) == expected_bytes


def test_single_slot_limit_can_be_disabled_and_validated(model):
    from fla.models.memory.configuration_memory import MemoryConfig
    with pytest.raises(ValueError):
        MemoryConfig(memory_offload_single_slot_max_tokens=-1)
    model.config.memory_offload_single_slot_max_tokens = 0
    model.config.memory_offload_policy = 'auto'
    model.enable_memory_offload()
    model.set_offload_offloader(1, 1025)
    assert len(model.model.memory_offloader.slots) == min(4, len(model.model.layers))


@pytest.mark.parametrize('group', [1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_shared_host_dma_lifetime_fresh_values_and_partial_groups(group):
    from fla.layers.memory_offload import MemoryTableOffloader
    with torch.inference_mode():
        weights = torch.randn(73, 7, 64, dtype=torch.bfloat16)
        off = MemoryTableOffloader(weights, 2, 17, group_size=group,
                                   prefetch_depth=4, single_host_buffer=True)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        capacity = 2 * 17 * group * 64 * 2
        assert _storage_bytes(s['host'] for s in off.slots) == capacity
        assert _storage_bytes(s['gpu'] for s in off.slots) == capacity * len(off.slots)
        try:
            for iteration in range(4):
                ids = torch.randint(0, 73, (2, 17))
                weights.add_(1)
                actual = []

                def consume(layer, handle):
                    value = handle.acquire()
                    actual.append(value.clone())
                    handle.release()
                # Delay DMA to expose premature overwrite of the shared host
                # buffer. Per-slot events alone cannot guard other slots.
                with torch.cuda.stream(off.copy_stream):
                    torch.cuda._sleep(1_000_000)
                with torch.cuda.stream(streams[iteration % 2]):
                    off.forward(ids, consume)
                torch.cuda.synchronize()
                expected = weights.index_select(0, ids.reshape(-1)).view(2, 17, 7, 64)
                for layer, value in enumerate(actual):
                    torch.testing.assert_close(value.cpu(), expected[:, :, layer], rtol=0, atol=0)

            def fail(layer, handle):
                handle.acquire()
                raise ValueError('consumer failure')
            with pytest.raises(ValueError, match='consumer failure'):
                off.forward(ids, fail)
            with pytest.raises(RuntimeError, match='closed or failed'):
                off.forward(ids, consume)
        finally:
            off.close()


@pytest.mark.parametrize('policy', ['auto', 'pipeline'])
def test_shared_host_policy_boundaries_and_full_model_exactness(model, policy):
    from fla.models.memory.configuration_memory import MemoryConfig
    with pytest.raises(ValueError, match='single-host'):
        MemoryConfig(memory_offload_single_host_min_tokens=16385,
                     memory_offload_single_host_max_tokens=16384)
    model.config.memory_offload_policy = policy
    model.enable_memory_offload()
    for tokens in (4095, 4096, 8192, 8193, 16384, 16385):
        model.set_offload_offloader(1, tokens)
        off = model.model.memory_offloader
        shared = policy == 'auto' and 4096 <= tokens <= 16384
        assert off.single_host_buffer == shared
        # Extending host sharing must not shrink the GPU prefetch capacity.
        assert len(off.slots) == min(4, len(model.model.layers))
        capacity = tokens * off.group * off.dim * off.weights.element_size()
        assert _storage_bytes(s['host'] for s in off.slots) == capacity * (1 if shared else len(off.slots))
        assert _storage_bytes(s['gpu'] for s in off.slots) == capacity * len(off.slots)
    # Exercise enabled scheduling cheaply in a full model, including tail group.
    model.close_memory_offload()
    model.config.memory_offload_bulk_max_tokens = 0
    model.config.memory_offload_single_slot_max_tokens = 0
    model.config.memory_offload_single_host_min_tokens = 1
    model.config.memory_offload_single_host_max_tokens = 32
    model.config.memory_offload_group_size = 2
    for length in (17, 23, 17):
        ids = torch.randint(0, 128, (1, length), device='cuda')
        expected = _resident_prefill(model, ids)
        model.enable_memory_offload()
        for _ in range(2):
            actual = model(input_ids=ids, use_cache=False).logits.float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('batch', [1, 4, 8, 16])
@pytest.mark.parametrize('ids_device', ['cpu', 'cuda'])
def test_mapped_bulk_fresh_weights_cross_stream_and_host_lifetime(batch, ids_device):
    from fla.layers.memory_offload import MappedBulkMemoryTableOffloader, PendingM
    with torch.inference_mode():
        weights = torch.randn(73, 5, 64, dtype=torch.bfloat16).pin_memory()
        off = MappedBulkMemoryTableOffloader(weights, batch, 1, 'cuda')
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        try:
            for iteration in range(4):
                ids = torch.randint(0, 73, (batch, 1))
                expected = weights.index_select(0, ids.flatten()).clone()
                values = []
                with torch.cuda.stream(streams[iteration % 2]):
                    device_ids = ids.to(ids_device)

                    def consume(layer, value):
                        handle = value if isinstance(value, PendingM) else None
                        if handle is not None:
                            value = handle.acquire()
                        torch.cuda._sleep(100000)
                        values.append(value.clone())
                        if handle is not None:
                            handle.release()
                    off.forward(device_ids, consume)
                # No GPU synchronization before host mutation: forward must
                # already have completed every mapped read of CPU storage.
                weights.add_(torch.tensor(.125, dtype=weights.dtype))
                streams[iteration % 2].synchronize()
                actual = torch.stack(values, dim=2).reshape(batch, 5, 64).cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            off.close()


def test_mapped_bulk_error_cleanup_and_cpu_bounds():
    from fla.layers.memory_offload import MappedBulkMemoryTableOffloader
    with torch.inference_mode():
        weights = torch.randn(17, 3, 64, dtype=torch.bfloat16).pin_memory()
        for invalid in (False, True):
            off = MappedBulkMemoryTableOffloader(weights, 8, 1, 'cuda')
            try:
                if invalid:
                    with pytest.raises(IndexError, match='out of range'):
                        off.forward(torch.full((8, 1), 17), lambda *args: None)
                else:
                    def fail(*args): raise ValueError('consumer failure')
                    with pytest.raises(ValueError, match='consumer failure'):
                        off.forward(torch.zeros((8, 1), device='cuda', dtype=torch.long), fail)
                assert off.broken
                with pytest.raises(RuntimeError, match='closed or failed'):
                    off.forward(torch.zeros((8, 1), dtype=torch.long), lambda *args: None)
            finally:
                off.close()


def test_mapped_bulk_model_selection_and_pinned_table_accounting(model):
    from fla.layers.memory_offload import BulkMemoryTableOffloader, MappedBulkMemoryTableOffloader
    model.enable_memory_offload()
    for batch in (1, 4, 8, 16):
        assert isinstance(model.model._offloader_for(batch, 1), MappedBulkMemoryTableOffloader)
    assert isinstance(model.model._offloader_for(17, 1), BulkMemoryTableOffloader)
    mapped = model.model._offloader_for(8, 1)
    assert isinstance(mapped, MappedBulkMemoryTableOffloader)
    assert mapped.host.numel() == 0
    assert model.model.memory_table.is_pinned()
    assert model.model.memory_table.device.type == 'cpu'
    for offloader in model.model._offloader_cache.values():
        for slot in offloader.slots:
            assert slot['host'].numel() == 0 or slot['host'].is_pinned()


@pytest.mark.parametrize('ids_device', ['cpu', 'cuda'])
def test_offload_dispatch_resolves_current_shape_once_and_direct_fallback(model, monkeypatch, ids_device):
    ids = torch.randint(0, 128, (2, 8), device=ids_device)
    model.fold_memory_table_on_gpu()
    reference = model(input_ids=ids.cuda(), use_cache=False, output_hidden_states=True)
    model.close_memory_offload()
    model.enable_memory_offload()
    backbone = model.model
    original = backbone._offloader_for
    resolved = []

    def track(batch, length):
        resolved.append((batch, length))
        return original(batch, length)

    monkeypatch.setattr(backbone, '_offloader_for', track)
    actual = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    assert resolved == [(2, 8)]
    torch.testing.assert_close(actual.logits, reference.logits, rtol=0, atol=0)
    for got, expected in zip(actual.hidden_states, reference.hidden_states):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    one_token = ids[:, :1]
    model(input_ids=one_token, use_cache=False)
    assert resolved == [(2, 8), (2, 1)]
    embedded = backbone.embeddings(ids.cuda())
    direct = backbone._forward_offloaded(embedded, ids.cpu(), None, None, False, True, True)
    assert resolved == [(2, 8), (2, 1), (2, 8)]
    torch.testing.assert_close(direct.last_hidden_state, reference.hidden_states[-1], rtol=0, atol=0)


@pytest.mark.parametrize('use_cache', [False, True])
@pytest.mark.parametrize('collect_hidden', [False, True])
@pytest.mark.parametrize('return_dict', [False, True])
def test_offload_dispatch_preserves_hooks_replacements_and_callback_state(use_cache, collect_hidden, return_dict):
    from fla.models.memory.modeling_memory import MemoryModel

    class Block(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self, hidden, **kwargs):
            assert kwargs['memory_table'] == self.increment
            assert kwargs['output_attentions'] is False
            return hidden + self.increment, kwargs['past_key_values'] + self.increment

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(1), Block(2)])
            self.norm = torch.nn.Identity()

        def _offloader_for(self, batch, length):
            assert (batch, length) == (1, 1)
            return self

        def forward(self, ids, consume):
            for index, layer in enumerate(self.layers):
                consume(index, layer.increment)

    stub = Stub()
    hidden = torch.zeros(1, 1, 4)
    ids = torch.zeros(1, 1, dtype=torch.long)
    calls = []
    for increment in (2, 4):
        stub.layers[1] = Block(increment)
        handle = stub.layers[1].register_forward_hook(lambda module, args, output: calls.append(module.increment))
        result = MemoryModel._forward_offloaded(stub, hidden, ids, None, 10, use_cache, collect_hidden, return_dict)
        output = result.last_hidden_state if return_dict else result[0]
        cache = result.past_key_values if return_dict else result[1]
        assert torch.equal(output, hidden + 1 + increment)
        assert cache == (11 + increment if use_cache else 10)
        if collect_hidden:
            states = result.hidden_states if return_dict else result[2]
            assert len(states) == 3
            for got, value in zip(states, (0, 1, 1 + increment)):
                assert torch.equal(got, hidden + value)
        elif not return_dict:
            assert len(result) == 2
        handle.remove()
    assert calls == [2, 4]


def test_offloader_cache_eviction_closes_transfers(model):
    model.enable_memory_offload()
    first = model.model._offloader_for(1, 8)
    model.model._offloader_for(1, 16)
    model.model._offloader_for(1, 32)
    assert first.closed
    assert len(model.model._offloader_cache) == 2
    assert (1, 8) not in model.model._offloader_cache


def test_fold_close_refreshes_weight_snapshot(model):
    model.fold_memory_table_on_gpu()
    model.close_memory_offload()
    assert model.model._raw_m_proj_weights is None
    for layer in model.model.layers:
        layer.attn.m_proj.weight.add_(0.25)
    expected = [layer.attn.m_proj.weight.clone() for layer in model.model.layers]
    model.fold_memory_table_on_gpu()
    model.close_memory_offload()
    for layer, weight in zip(model.model.layers, expected):
        torch.testing.assert_close(layer.attn.m_proj.weight, weight, rtol=0, atol=0)
