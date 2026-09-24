# Memory Attention

An experimental implementation of Memory Attention built on Flash Linear Attention (FLA). It introduces a learnable memory table indexed by token IDs into the attention value computation and provides a Hugging Face-style causal language model interface.

The core implementation lives in [`fla/layers/memory_attn.py`](fla/layers/memory_attn.py) and [`fla/models/memory/`](fla/models/memory/). This repository retains FLA's other models and operators. The Python import name remains `fla`, and the package name remains `flash-linear-attention`.

## How It Works

For each layer's hidden states `X` and original token IDs `I`:

```text
Q  = q_proj(X)
K₀ = k_proj(X)
M  = m_proj(I)
V  = K₀ + RMSNorm(M)

Q, K = RoPE(optional_qk_norm(Q, K₀))
O    = causal_attention(Q, K, V)
Y    = o_proj(O)
```

Each attention layer has its own learnable embedding table, `m_proj`, with shape `[vocab_size, num_kv_heads × head_dim]`. Memory vectors are RMS-normalized per head. Values use `K₀` before QK normalization and RoPE, with no separate `v_proj`. When `use_gate` is enabled, a sigmoid gate computed from the hidden states is applied to `O` before the output projection.

The attention operation uses FlashAttention to compute causal softmax attention. Full attention has quadratic compute complexity in sequence length. Here, memory refers to a learned token lookup table stored in the model parameters; it is distinct from external retrieval and the KV cache.

## Installation

See the [Flash Linear Attention repository](https://github.com/fla-org/flash-linear-attention) for installation instructions.

## Performance Profiling

The [`profile/`](profile/) directory contains standalone inference experiments comparing standard attention (`standard`), GPU-resident memory tables (`ma_gpu`), and CPU-offloaded memory tables (`ma_offload`). The offloading implementation belongs to the benchmark and is not integrated into `MemoryForCausalLM`.

Run the prefill and decode benchmark with the default configuration and save the results:

```bash
python profile/bmk.py --mode both --json bmk_results.json
```

Append `--profile-dir profiles` to capture stage timelines and `--kernel-trace` to save Chrome traces. Once a timeline has been captured, render it as an image:

```bash
python -m pip install matplotlib
python profile/ma_profile.py profiles/ma_offload_decode.timeline.json
```

Benchmark timings include the input embedding, all blocks, final RMSNorm, and LM head. CPU and GPU stage timelines use separate time origins. GPU ranges measure elapsed stream time, not kernel busy time. Report the device, model dimensions, context length, and timing configuration alongside performance results.

## Repository Structure

```text
fla/layers/memory_attn.py  Memory Attention layer
fla/models/memory/        Configuration, backbone, and causal language model
fla/modules/              RMSNorm, RoPE, MLP, and other components
fla/ops/                  FLA Triton operators
profile/                  Inference comparisons, CPU offloading, and profiling
benchmarks/               Benchmarks for other models and operators
evals/                    Language model evaluation entry points
legacy/training/          Legacy training code and examples
```

[`eval.sh`](eval.sh) demonstrates the `evals.harness` entry point, which requires the additional `lm-eval` dependency. Replace its local model path and device settings before use. Historical results in its comments are not performance claims for the current implementation. See [`legacy/training/README.md`](legacy/training/README.md) for the legacy training instructions.

## Acknowledgments and License

This project builds on Flash Linear Attention and retains its modules, operators, and source attribution. See [`CITATION.cff`](CITATION.cff) for upstream citation information.

The repository's [`LICENSE`](LICENSE) contains Apache License 2.0, while some license metadata in `pyproject.toml` and `setup.py` still specifies MIT. These declarations have not yet been reconciled.
