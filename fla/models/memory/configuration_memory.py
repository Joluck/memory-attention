# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from transformers.configuration_utils import PretrainedConfig


class MemoryConfig(PretrainedConfig):

    model_type = 'memory'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        use_gate: bool = False,
        use_head_gate: bool = False,
        memory_offload: bool = False,
        memory_offload_policy: str = "auto",
        memory_offload_group_size: int = 1,
        memory_offload_prefetch_depth: int = 4,
        memory_offload_bulk_max_tokens: int = 1024,
        memory_offload_single_slot_max_tokens: int = 2048,
        memory_offload_single_host_min_tokens: int = 4096,
        memory_offload_single_host_max_tokens: int = 16384,
        memory_offload_chunk_size: int = 1024,
        memory_offload_mapped_bulk: bool = True,
        memory_offload_mapped_bulk_min_tokens: int = 1,
        memory_offload_mapped_bulk_max_tokens: int = 16,
        **kwargs,
    ):

        self.use_gate = use_gate
        self.use_head_gate = use_head_gate
        self.memory_offload = memory_offload
        if memory_offload_policy not in ("auto", "pipeline", "bulk"):
            raise ValueError("`memory_offload_policy` must be one of 'auto', 'pipeline', 'bulk'")
        self.memory_offload_policy = memory_offload_policy
        self.memory_offload_group_size = memory_offload_group_size
        self.memory_offload_prefetch_depth = memory_offload_prefetch_depth
        self.memory_offload_bulk_max_tokens = memory_offload_bulk_max_tokens
        if memory_offload_single_slot_max_tokens < 0:
            raise ValueError("single-slot token limit must be nonnegative")
        self.memory_offload_single_slot_max_tokens = memory_offload_single_slot_max_tokens
        if not 0 <= memory_offload_single_host_min_tokens <= memory_offload_single_host_max_tokens:
            raise ValueError("invalid single-host token range")
        self.memory_offload_single_host_min_tokens = memory_offload_single_host_min_tokens
        self.memory_offload_single_host_max_tokens = memory_offload_single_host_max_tokens
        self.memory_offload_chunk_size = memory_offload_chunk_size
        self.memory_offload_mapped_bulk = memory_offload_mapped_bulk
        if not 1 <= memory_offload_mapped_bulk_min_tokens <= memory_offload_mapped_bulk_max_tokens:
            raise ValueError("invalid mapped bulk token range")
        self.memory_offload_mapped_bulk_min_tokens = memory_offload_mapped_bulk_min_tokens
        self.memory_offload_mapped_bulk_max_tokens = memory_offload_mapped_bulk_max_tokens
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
