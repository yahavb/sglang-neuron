# SPDX-License-Identifier: Apache-2.0
"""The Muse-Glimmer text decoder as the runtime's role graphs (one class, ``phase`` prefill or decode), on attention
TP, in plain PyTorch: the same math as the validated vLLM Neuron port (vllm-neuron contrib/models/muse_glimmer,
model_bf16.py), which follows HF MuseGlimmerTextModel. No NKI kernel runs in the model; the KV cache is the runtime's
paged pool, read through the block table and written at the slot mapping.

Muse-Glimmer, as HF:
  - a weight-less RMSNorm after the embedding; a standard RMSNorm (x * w) at the end
  - four centered RMSNorms per layer (x * (1 + w)); the post-attention and post-MLP ones use post_norm_eps
  - gated attention: o_proj(attn * sigmoid(gate_proj(x)))
  - weight-less Q/K RMSNorm, Q scaled by qk_scale_factor / sqrt(head_dim), the whole softmax scale
  - [sliding x3, full] layers: sliding = window 2048 + RoPE (rotate-half), full = NoPE
  - logits = cap * tanh(logits * output_multiplier / cap), on each rank's vocabulary shard
"""

import torch
import torch.nn.functional as F
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.muse_glimmer.config import MuseGlimmerTextConfig
from sglang_neuron.models.transformer import weights
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

DTYPE = torch.bfloat16
# the decoder's prefix in the multimodal checkpoint (HF MuseGlimmerForConditionalGeneration)
SRC = "model.language_model"


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """HF MuseGlimmer's RMS core, in fp32."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class CenteredRMSNorm(nn.Module):
    """HF MuseGlimmerTextCenteredRMSNorm: x * (1 + w)."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        return (_rms(x, self.eps) * (1.0 + self.weight.float())).to(x.dtype)


class RMSNorm(nn.Module):
    """HF MuseGlimmerRMSNorm (the final norm): x * w."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        return (_rms(x, self.eps) * self.weight.float()).to(x.dtype)


def column_rows_loader(rows: int) -> SafetensorsWeightLoader:
    """A column-parallel [out, H] projection (the attention output gate): this rank's output rows, as [H, out / tp]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows].t())


class Attention(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.name = f"layers.{layer_idx}.self_attn"
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = get_attn_tp_group().world_size
        kv = config.num_key_value_heads
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} attention heads do not shard over an attention TP of {tp}")
        if kv % tp and tp % kv:
            raise ValueError(f"{kv} KV heads neither shard over nor replicate across an attention TP of {tp}")
        self.head_dim = d = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, kv // tp)
        self.eps = config.rms_norm_eps
        self.q_scale = config.q_scale
        self.use_rope = config.rope[layer_idx]
        self.sliding_window = config.sliding_window if config.sliding[layer_idx] else None
        q, kvw = self.num_heads * d, self.num_kv_heads * d
        self.qkv_weight = nn.Parameter(torch.empty(config.hidden_size, q + 2 * kvw, dtype=DTYPE))
        self.gate_weight = nn.Parameter(torch.empty(config.hidden_size, q, dtype=DTYPE))
        self.o_weight = nn.Parameter(torch.empty(q, config.hidden_size, dtype=DTYPE))
        set_weight_loader(self.qkv_weight, weights.fused_qkv_weight_loader(tp, q, kvw, kv))
        set_weight_loader(self.gate_weight, column_rows_loader(q))
        set_weight_loader(self.o_weight, weights.row_parallel_loader(q))
        # bound by bind_kv_cache: [pages, kv_heads, page, d] each
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, positions, cos, sin, md):
        """``x`` [T, H] of B rows of S tokens each: a prefill is one request's chunk (B 1), a decode one token a row
        (S 1). Returns this layer's attention output, all-reduced over the attention TP."""
        block_table = md["block_table_tensor"]
        bs = md["block_size"]
        B, nblk = block_table.shape
        T = x.shape[0]
        S = T // B
        nq, nkh, D = self.num_heads, self.num_kv_heads, self.head_dim

        q, k, v = (x @ self.qkv_weight).split([nq * D, nkh * D, nkh * D], dim=-1)
        q = (_rms(q.view(T, nq, D), self.eps) * self.q_scale).to(DTYPE)
        k = _rms(k.view(T, nkh, D), self.eps).to(DTYPE)
        v = v.view(T, nkh, D)
        if self.use_rope:
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)

        # the cached K/V of each row: [B, nkh, nblk * bs, D]
        pages = self.k_cache.shape[0]
        table = torch.clamp(block_table, min=0).to(torch.long)
        kc = self.k_cache[table].permute(0, 2, 1, 3, 4).reshape(B, nkh, nblk * bs, D)
        vc = self.v_cache[table].permute(0, 2, 1, 3, 4).reshape(B, nkh, nblk * bs, D)

        # the absolute position of each cached slot; a decode's sliding table is trimmed to the window and starts at
        # swa_kv_pos_offset
        slot_pos = torch.arange(nblk * bs, device=x.device).view(1, 1, -1)
        offset = md.get("swa_kv_pos_offset")
        if self.sliding_window is not None and offset is not None:
            slot_pos = slot_pos + offset.view(B, 1, 1).to(slot_pos.dtype)
        pos = positions.view(B, S, 1).to(slot_pos.dtype)
        # only tokens of earlier steps are in the cache, the positions before this chunk's first: the chunk's own K/V
        # are written after this read, and come from k / v below
        chunk_start = positions.view(B, S)[:, :1].to(slot_pos.dtype).view(B, 1, 1)
        valid = (block_table >= 0).repeat_interleave(bs, dim=1).view(B, 1, -1)
        prior_ok = valid & (slot_pos < chunk_start) & (slot_pos < pos)
        new_pos = positions.view(B, 1, S).to(slot_pos.dtype)
        # causal among the new tokens by index, not position: the runtime pads a chunk at its end
        idx = torch.arange(S, device=x.device)
        new_ok = (idx.view(1, 1, S) <= idx.view(1, S, 1)).expand(B, S, S)
        if self.sliding_window is not None:
            prior_ok = prior_ok & (slot_pos > pos - self.sliding_window)
            new_ok = new_ok & (new_pos > pos - self.sliding_window)
        allowed = torch.cat([prior_ok, new_ok], dim=-1)  # [B, S, C + S]

        qb = q.view(B, S, nq, D).permute(0, 2, 1, 3)
        kb = torch.cat([kc, k.view(B, S, nkh, D).permute(0, 2, 1, 3).to(kc.dtype)], dim=2)
        vb = torch.cat([vc, v.view(B, S, nkh, D).permute(0, 2, 1, 3).to(vc.dtype)], dim=2)
        kb = kb.repeat_interleave(nq // nkh, dim=1)
        vb = vb.repeat_interleave(nq // nkh, dim=1)
        scores = torch.matmul(qb.float(), kb.float().transpose(-1, -2))
        scores = scores.masked_fill(~allowed.view(B, 1, S, -1), float("-inf"))
        attn = torch.matmul(torch.softmax(scores, dim=-1), vb.float()).to(DTYPE)  # [B, nq, S, D]
        attn = attn.permute(0, 2, 1, 3).reshape(T, nq * D)
        out = (attn * torch.sigmoid(x @ self.gate_weight)) @ self.o_weight

        # write this step's K/V; slots outside the pool (padding) go to slot 0, on SGLang's reserved page
        slots = md["slot_mapping"].reshape(-1).to(torch.long)
        slots = torch.where((slots < 0) | (slots >= pages * bs), torch.zeros_like(slots), slots)
        self.k_cache[slots // bs, :, slots % bs] = k.to(self.k_cache.dtype)
        self.v_cache[slots // bs, :, slots % bs] = v.to(self.v_cache.dtype)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


def _rope(x, cos, sin):
    """x [T, N, D]; cos / sin [T, D]; rotate-half, as HF apply_rotary_pos_emb."""
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos[:, None, :] + rot * sin[:, None, :]).to(x.dtype)


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)), the intermediate dimension sharded over the attention TP."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if config.intermediate_size % tp:
            raise ValueError(f"an intermediate size of {config.intermediate_size} does not shard over {tp} ranks")
        rows = config.intermediate_size // tp
        self.gate_up_weight = nn.Parameter(torch.empty(config.hidden_size, 2 * rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(rows, config.hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up_weight, weights.gate_up_loader(rows, fused=False))
        set_weight_loader(self.down_weight, weights.row_parallel_loader(rows))

    def forward(self, x):
        gate, up = (x @ self.gate_up_weight).chunk(2, dim=-1)
        return neuron_collectives.all_reduce(self.attn_tp_group, (F.silu(gate) * up) @ self.down_weight)


class DecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        h = config.hidden_size
        self.input_layernorm = CenteredRMSNorm(h, config.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(h, config.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(h, config.rms_norm_eps)
        self.post_feedforward_layernorm = CenteredRMSNorm(h, config.post_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, positions, cos, sin, md):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), positions, cos, sin, md))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class MuseGlimmerGraph(nn.Module):
    """One role graph: ``phase`` "prefill" or "decode"."""

    def __init__(self, config: MuseGlimmerTextConfig, phase: str):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = group = groups.get_neuron_attn_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, config.hidden_size))
        set_weight_loader(self.lm_head.weight,
                          weights.vocab_rows_loader(self.lm_head.out_features_per_rank, config.hidden_size))
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float)
                                                / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # the layers the runtime keys attention metadata by, and each one's window
        self.layer_names = [layer.self_attn.name for layer in self.layers]
        self.sliding_windows = config.windows()

    def _rope_tables(self, positions):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(DTYPE), emb.sin().to(DTYPE)

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        h = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        h = _rms(h, self.config.rms_norm_eps).to(h.dtype)
        cos, sin = self._rope_tables(positions)
        for layer in self.layers:
            h = layer(h, positions, cos, sin, attn_metadata[layer.self_attn.name])
        h = self.norm(h.index_select(0, sampling_positions))
        logits = self.lm_head(h)
        cap = self.config.logit_cap
        return (torch.tanh(logits.float() * (self.config.output_multiplier / cap)) * cap).to(logits.dtype)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=name) for name in self.layer_names])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.k_cache, attn.v_cache = kv_caches[attn.name]

    def hf_weight_mappings(self) -> dict:
        m = {"embed_tokens.weight": f"{SRC}.embed_tokens.weight", "norm.weight": f"{SRC}.norm.weight",
             "lm_head.weight": "lm_head.weight"}
        for i in range(len(self.layers)):
            s, d = f"{SRC}.layers.{i}", f"layers.{i}"
            m[f"{d}.self_attn.qkv_weight"] = [f"{s}.self_attn.{n}_proj.weight" for n in "qkv"]
            m[f"{d}.self_attn.gate_weight"] = f"{s}.self_attn.gate_proj.weight"
            m[f"{d}.self_attn.o_weight"] = f"{s}.self_attn.o_proj.weight"
            for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm"):
                m[f"{d}.{n}.weight"] = f"{s}.{n}.weight"
            m[f"{d}.mlp.gate_up_weight"] = [f"{s}.mlp.gate_proj.weight", f"{s}.mlp.up_proj.weight"]
            m[f"{d}.mlp.down_weight"] = f"{s}.mlp.down_proj.weight"
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """This rank's shard of every decoder parameter; the vision tower's keys are not read."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(self.tp_group.rank_in_group, self.tp_group.world_size, self,
                                                  self.hf_weight_mappings(), device).state_dict
        for k, t in state.items():
            if t.dtype != DTYPE:
                state[k] = t.to(DTYPE)
        missing = sorted(set(dict(self.named_parameters())) - set(state))
        if missing:
            raise RuntimeError(f"Muse-Glimmer: the checkpoint did not provide {len(missing)} parameters, e.g. "
                               f"{missing[:4]}")
        self.load_state_dict(state, strict=False, assign=True)
