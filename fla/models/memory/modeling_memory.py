# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.memory_attn import MemoryAttention
from fla.layers.memory_offload import (
    BulkMemoryTableOffloader,
    MappedBulkMemoryTableOffloader,
    MemoryTableOffloader,
    build_cpu_table,
    fold_memory_table,
)
from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as MemoryMLP
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class MemoryBlock(GradientCheckpointingLayer):

    def __init__(self, config: MemoryConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = MemoryAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            use_gate=config.use_gate,
            use_head_gate=config.use_head_gate,
            vocab_size=config.vocab_size,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = MemoryMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class MemoryPreTrainedModel(PreTrainedModel):

    config_class = MemoryConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MemoryBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, nn.Linear | nn.Conv1d):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class MemoryModel(MemoryPreTrainedModel):

    def __init__(
        self,
        config: MemoryConfig,
    ) -> MemoryModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MemoryBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        # CPU offload state. The stacked table is a plain attribute (not a
        # buffer) so it stays out of state_dict/parameter counts; the folded
        # tables are what actually gets transferred, and the per-layer
        # nn.Embedding weights are dropped to free their device copy.
        self.memory_offloader = None
        self.memory_table = None

        self.post_init()

    def enable_memory_offload(
        self,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        fold_norm: bool = True,
    ) -> None:
        """Move the memory table to pinned CPU memory and stream it per forward.

        The per-layer ``m_proj`` embeddings are folded (including the head-wise
        RMSNorm) into a single ``[vocab, layers, kv_dim]`` CPU tensor, then
        removed from the module tree so their device copy is freed. Layers then
        receive memory values through the offloader instead of a local lookup.

        Must be called after weights are loaded and the model is on ``device``.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("memory offload requires CUDA")
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("memory offload is inference-only; call under torch.inference_mode()")

        device = torch.device(device)
        config = self.config
        head_dim = config.hidden_size // config.num_heads

        m_projs = [layer.attn.m_proj for layer in self.layers]
        if any(m is None for m in m_projs):
            raise RuntimeError(
                "m_proj is already offloaded; call close_memory_offload() before re-enabling"
            )
        if any(layer.attn.memory_table_folded for layer in self.layers):
            raise RuntimeError("close_memory_offload() before enabling offload from a folded table")
        # Snapshot the raw weights so the unfolded resident path can be restored.
        # Guard against snapshotting an already-folded table.
        if getattr(self, "_raw_m_proj_weights", None) is None:
            self._raw_m_proj_weights = [
                m.weight.detach().to("cpu", copy=True) for m in m_projs
            ]
        norms = [layer.attn.m_norm for layer in self.layers]

        if fold_norm:
            table = build_cpu_table(
                m_projs, norms, head_dim,
                chunk_size=config.memory_offload_chunk_size,
                dtype=dtype,
            )
        else:
            table = torch.stack(
                [m_proj.weight.detach().to(dtype).cpu() for m_proj in m_projs], dim=1
            ).contiguous()
        for layer in self.layers:
            layer.attn.memory_table_folded = fold_norm

        # Retain the originals on CPU rather than deleting them, so
        # close_memory_offload() can restore the resident path. The folded
        # table is what gets transferred, so the device copies can go.
        self._resident_m_projs = [
            nn.Embedding.from_pretrained(m.weight.detach().to("cpu"), freeze=True)
            for m in m_projs
        ]
        for layer in self.layers:
            layer.attn.m_proj = None

        self.memory_table = table.pin_memory() if config.memory_offload_mapped_bulk else table
        self._offload_device = device
        # One cached offloader per input shape (prefill vs decode differ).
        self._offloader_cache = {}
        self._offloader_cache_size = 2

    def fold_memory_table_on_gpu(self, dtype: torch.dtype = torch.bfloat16) -> None:
        """Fold m_norm into the resident table, keeping it on the GPU.

        Same transformation the offload path uses, but the folded table stays
        in device memory. This isolates the transfer cost: comparing a folded
        resident table against an offloaded one measures the H2D and its
        coordination, with the norm-folding win present in both.
        """
        if self.memory_offloader is not None:
            raise RuntimeError("close_memory_offload() before folding a resident table")
        if any(layer.attn.m_proj is None for layer in self.layers):
            raise RuntimeError("memory table is already offloaded")
        if any(layer.attn.memory_table_folded for layer in self.layers):
            raise RuntimeError("memory table is already folded")
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("memory table folding is inference-only; call under torch.inference_mode()")
        if getattr(self, "_raw_m_proj_weights", None) is None:
            self._raw_m_proj_weights = [
                layer.attn.m_proj.weight.detach().to("cpu", copy=True)
                for layer in self.layers
            ]

        config = self.config
        head_dim = config.hidden_size // config.num_heads
        for layer in self.layers:
            m_proj, norm = layer.attn.m_proj, layer.attn.m_norm
            vocab, kv_dim = m_proj.weight.shape
            raw = m_proj.weight.detach().reshape(vocab, kv_dim // head_dim, head_dim)
            folded = fold_memory_table(norm, raw.to(norm.weight.dtype), config.memory_offload_chunk_size, dtype)
            device = m_proj.weight.device
            layer.attn.m_proj = nn.Embedding.from_pretrained(
                folded.to(device=device, dtype=dtype), freeze=True
            )
            layer.attn.memory_table_folded = True
        self._table_folded = True

    def _build_offloader(self, batch: int, seq_len: int):
        config = self.config
        device = self._offload_device
        policy = config.memory_offload_policy
        automatic = policy == "auto"
        if automatic:
            policy = "bulk" if batch * seq_len <= config.memory_offload_bulk_max_tokens else "pipeline"
        if policy == "bulk":
            if (config.memory_offload_mapped_bulk and
                    config.memory_offload_mapped_bulk_min_tokens <= batch * seq_len <= config.memory_offload_mapped_bulk_max_tokens):
                return MappedBulkMemoryTableOffloader(self.memory_table, batch, seq_len, device)
            return BulkMemoryTableOffloader(self.memory_table, batch, seq_len, device)
        return MemoryTableOffloader(
            self.memory_table, batch, seq_len,
            group_size=config.memory_offload_group_size,
            device=device,
            prefetch_depth=(min(config.memory_offload_prefetch_depth, 1)
                            if automatic and batch * seq_len <= config.memory_offload_single_slot_max_tokens
                            else config.memory_offload_prefetch_depth),
            single_host_buffer=(automatic and
                                config.memory_offload_single_host_min_tokens <= batch * seq_len <=
                                config.memory_offload_single_host_max_tokens),
        )

    def _offloader_for(self, batch: int, seq_len: int):
        """Return an offloader for this shape, building it on first use.

        Prefill and decode have very different shapes (long sequence vs a single
        token), and each offloader preallocates pinned host plus device buffers
        sized to its input. Rebuilding on every shape change would reallocate
        hundreds of MiB per generated token, so instances are cached per shape.
        """
        if self.memory_table is None:
            raise RuntimeError("call enable_memory_offload() first")
        cache = getattr(self, "_offloader_cache", None)
        if cache is None:
            cache = self._offloader_cache = {}
        key = (batch, seq_len)
        if key not in cache:
            if len(cache) >= self._offloader_cache_size:
                # Evicted buffers must finish their transfers before being released.
                cache.pop(next(iter(cache))).close()
            cache[key] = self._build_offloader(batch, seq_len)
        self.memory_offloader = cache[key]
        return cache[key]

    def set_offload_offloader(self, batch: int, seq_len: int) -> None:
        """Build (or reuse) the prefetcher for a given input shape."""
        self._offloader_for(batch, seq_len)

    def close_memory_offload(self) -> None:
        """Stop streaming and restore the GPU-resident m_proj path."""
        for offloader in (getattr(self, "_offloader_cache", None) or {}).values():
            with suppress(Exception):
                offloader.close()
        self._offloader_cache = {}
        if self.memory_offloader is not None:
            self.memory_offloader.close()
            self.memory_offloader = None
        self.memory_table = None
        if getattr(self, "_resident_m_projs", None) is not None:
            device = getattr(self, "_offload_device", torch.device("cpu"))
            for layer, m in zip(self.layers, self._resident_m_projs):
                layer.attn.m_proj = nn.Embedding.from_pretrained(
                    m.weight.detach().to(device), freeze=True
                )
                layer.attn.memory_table_folded = False
            self._resident_m_projs = None
        elif any(layer.attn.memory_table_folded for layer in self.layers):
            # Restore each resident table on its current device and dtype.
            for layer, weight in zip(self.layers, self._raw_m_proj_weights):
                current = layer.attn.m_proj.weight
                layer.attn.m_proj = nn.Embedding.from_pretrained(
                    weight.to(device=current.device, dtype=current.dtype), freeze=True
                )
                layer.attn.memory_table_folded = False
        self._table_folded = False
        self._raw_m_proj_weights = None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        # Key off memory_table, not memory_offloader: enabling offload frees the
        # per-layer m_proj immediately, so the first forward has no offloader
        # yet but must still take the streaming path. _offloader_for() builds
        # one on demand for whatever shape this call uses.
        if self.memory_table is not None:
            # The offload producer gathers from host memory, so it needs the
            # IDs on CPU. Accept them on either device and stage the embedding
            # lookup on the compute device.
            device = self.embeddings.weight.device
            offloader = self._offloader_for(*input_ids.shape) if input_ids is not None else None
            if getattr(offloader, "accepts_gpu_ids", False):
                ids_cpu = input_ids  # This bulk path consumes fresh GPU IDs directly.
            else:
                ids_cpu = input_ids.to("cpu") if input_ids is not None else None
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("offloaded forward requires input_ids or inputs_embeds")
                inputs_embeds = self.embeddings(input_ids.to(device, non_blocking=True))
            return self._forward_offloaded(
                inputs_embeds, ids_cpu, attention_mask, past_key_values,
                use_cache, output_hidden_states, return_dict, offloader=offloader,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                input_ids=input_ids,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )

    def _forward_offloaded(
        self,
        hidden_states: torch.Tensor,
        input_states_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
        offloader=None,
    ):
        """Run the layer stack while the memory table streams from CPU.

        The offloader owns the loop: it hands each layer its slice of M at the
        point of use, so the H2D for a group overlaps the Q/K kernels of the
        layers already running.
        """
        if offloader is None:
            offloader = self._offloader_for(*hidden_states.shape[:2])
        # Resolve references once for this call; replacements between calls
        # remain visible and each layer still runs through Module.__call__.
        layers = tuple(self.layers)
        all_hidden_states = () if output_hidden_states else None

        def consume(index: int, memory_table) -> None:
            nonlocal hidden_states, past_key_values, all_hidden_states
            layer = layers[index]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                memory_table=memory_table,
            )
            hidden_states = outputs[0]
            if use_cache:
                past_key_values = outputs[1]

        # Ordinary producers use CPU IDs; mapped bulk accepts CPU or GPU IDs.
        if input_states_ids is None:
            raise ValueError("offloaded forward requires input_ids")
        offloader.forward(input_states_ids, consume)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class MemoryForCausalLM(MemoryPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MemoryModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Memory-table CPU offload. Thin pass-throughs to the backbone, which owns
    # the folded table and the prefetcher.
    def enable_memory_offload(
        self,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        fold_norm: bool = True,
    ) -> None:
        self.model.enable_memory_offload(device=device, dtype=dtype, fold_norm=fold_norm)

    def fold_memory_table_on_gpu(self, dtype: torch.dtype = torch.bfloat16) -> None:
        self.model.fold_memory_table_on_gpu(dtype=dtype)

    def set_offload_offloader(self, batch: int, seq_len: int) -> None:
        self.model.set_offload_offloader(batch, seq_len)

    def close_memory_offload(self) -> None:
        self.model.close_memory_offload()

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
