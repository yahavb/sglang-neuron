# SPDX-License-Identifier: Apache-2.0
"""The Muse-Glimmer text decoder's shape, from either form of its config.

SGLang loads the checkpoint through its own flat ``MuseGlimmerConfig`` (sglang/srt/configs/muse_glimmer.py), which
translates the Hugging Face export: ``no_rope_layers`` (1 = RoPE), ``output_soft_cap_temp``, and ``qk_scale_factor``
multiplied by sqrt(head_dim). A Hugging Face ``MuseGlimmerConfig`` nests the decoder in ``text_config`` with the
export's own names (``layer_rope_theta``, ``final_logit_softcapping``, ``qk_scale_factor``). Both are read here.

Muse-Glimmer-30B: 52 layers, hidden 6656, 32 Q / 2 KV heads, head_dim 128, SwiGLU 19968, vocab 202048. Layers
[sliding, sliding, sliding, full] x 13: sliding = window 2048 with RoPE (theta 500000, rotate-half), full = NoPE."""

import math
from dataclasses import dataclass

ARCHITECTURE = "MuseGlimmerForConditionalGeneration"


@dataclass(frozen=True)
class MuseGlimmerTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float  # embedding norm, pre-norms, Q/K norm, final norm
    post_norm_eps: float  # post-attention and post-MLP norms
    rope_theta: float
    sliding_window: int
    sliding: tuple  # per layer: sliding window (True) or full attention
    rope: tuple  # per layer: RoPE (True) or NoPE
    q_scale: float  # the weight-less Q norm's gain, the whole softmax scale (3.87 / sqrt(head_dim))
    output_multiplier: float
    logit_cap: float  # logits = cap * tanh(logits * output_multiplier / cap)
    tie_word_embeddings: bool

    def windows(self) -> list:
        return [self.sliding_window if s else None for s in self.sliding]


def _get(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def text_config(hf_config) -> MuseGlimmerTextConfig:
    """The decoder's config, refusing what the graphs do not serve."""
    text = _get(hf_config, "text_config")
    nested = text is not None
    c = text if nested else hf_config
    n = int(_get(c, "num_hidden_layers"))
    head_dim = int(_get(c, "head_dim"))
    layer_types = list(_get(c, "layer_types") or [])
    if len(layer_types) != n:
        raise ValueError(f"Muse-Glimmer: layer_types has {len(layer_types)} entries for {n} layers")
    bad = sorted({t for t in layer_types if t not in ("sliding_attention", "full_attention")})
    if bad:
        raise ValueError(f"Muse-Glimmer: layer kinds {bad} are not served")
    if nested:
        thetas = list(_get(c, "layer_rope_theta"))
        rope = tuple(bool(t) for t in thetas)
        rope_theta = float((_get(c, "rope_parameters") or {}).get("rope_theta", 0) or next(t for t in thetas if t))
        if len({t for t in thetas if t}) > 1:
            raise ValueError(f"Muse-Glimmer: one RoPE theta is served, the config has {sorted(set(thetas))}")
        # the export's defaults where a key is absent, as the validated vLLM Neuron port (contrib muse_glimmer)
        q_scale = float(_get(c, "qk_scale_factor", 3.87)) / math.sqrt(head_dim)
        cap = _get(c, "final_logit_softcapping", 20.0)
    else:
        rope = tuple(int(x) == 1 for x in _get(c, "no_rope_layers"))
        rope_theta = float(_get(c, "rope_theta"))
        # SGLang's config holds the export's factor times sqrt(head_dim), and scales scores by it over head_dim
        q_scale = float(_get(c, "qk_scale_factor")) / head_dim
        cap = _get(c, "output_soft_cap_temp")
        for flag in ("use_qk_norm", "use_attn_output_gate", "normalize_tok_embeddings", "rope_is_neox_style"):
            if not _get(c, flag, True):
                raise ValueError(f"Muse-Glimmer: {flag}=False is not served")
    if len(rope) != n:
        raise ValueError(f"Muse-Glimmer: {len(rope)} RoPE flags for {n} layers")
    if not cap:
        raise ValueError("Muse-Glimmer: a config without the final logit soft cap is not served")
    return MuseGlimmerTextConfig(
        vocab_size=int(_get(c, "vocab_size")),
        hidden_size=int(_get(c, "hidden_size")),
        intermediate_size=int(_get(c, "intermediate_size")),
        num_hidden_layers=n,
        num_attention_heads=int(_get(c, "num_attention_heads")),
        num_key_value_heads=int(_get(c, "num_key_value_heads")),
        head_dim=head_dim,
        rms_norm_eps=float(_get(c, "rms_norm_eps", 1e-5)),
        post_norm_eps=float(_get(c, "post_norm_eps", 1e-8)),
        rope_theta=rope_theta,
        sliding_window=int(_get(c, "sliding_window", 2048)),
        sliding=tuple(t == "sliding_attention" for t in layer_types),
        rope=rope,
        q_scale=q_scale,
        output_multiplier=float(_get(c, "output_multiplier", 0.19611613513818404)),
        logit_cap=float(cap),
        tie_word_embeddings=bool(_get(hf_config, "tie_word_embeddings", False)),
    )
