# SPDX-License-Identifier: Apache-2.0
"""MuseGlimmerForConditionalGeneration: the host SGLang builds for a Muse-Glimmer checkpoint (models/host.py). Text
only: it builds the decoder's graphs (models/muse_glimmer/graph.py) and reads none of the vision tower's weights, so a
launch takes --language-model-only, which makes SGLang refuse image and video input instead of processing it."""

import copy
import logging

from torch import nn

from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.muse_glimmer.config import ARCHITECTURE, text_config
from sglang_neuron.models.muse_glimmer.graph import MuseGlimmerGraph
from sglang_neuron.models.transformer.config import check_bf16_launch, check_no_expert_parallel

logger = logging.getLogger(__name__)


def decoder_config(config):
    """The decoder's config object the runtime reads (layers, heads, layer_types, sliding_window): a Hugging Face
    config's text_config, carrying the checkpoint path the host loads from; SGLang's flat config as it is."""
    text = getattr(config, "text_config", None)
    if text is None:
        return config
    text = copy.copy(text)
    text._name_or_path = getattr(config, "_name_or_path", "") or getattr(text, "_name_or_path", "")
    for name in ("dtype", "torch_dtype"):
        if getattr(text, name, None) is None and getattr(config, name, None) is not None:
            setattr(text, name, getattr(config, name))
    return text


class MuseGlimmerForConditionalGeneration(ModelHost):
    name = "Muse-Glimmer"
    # the prefill attention is the graph's own (plain PyTorch), not the segmented kernel, and runs whole chunks
    segmented_prefill = False
    prefill_ladder = False

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        if not getattr(config, "language_model_only", False):
            raise ValueError(f"{ARCHITECTURE} on Neuron serves text only: launch with --language-model-only")
        super().__init__(decoder_config(config), quant_config, prefix, **kwargs)

    def check_launch(self, config) -> None:
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        if getattr(config, "quantization_config", None):
            raise ValueError(f"{self.name}: the graphs serve bf16 checkpoints; this one is quantized")
        cfg = text_config(config)
        if cfg.tie_word_embeddings:
            raise ValueError(f"{self.name}: tied word embeddings are not served")
        logger.info("%s decoder: %s", self.name, cfg)

    def neuron_compiler_args(self, phase: str, default_args: list[str]) -> list[str]:
        """The platform's arguments, and with MUSE_GLIMMER_NEURON_MODEL_TYPE set (e.g. "transformer", what the vLLM
        Neuron port compiles the same graphs with) that model type in place of the platform's generic one: a list that
        names a model type keeps it (backend/stack.py). Unset: the platform's arguments as they are."""
        import os

        args = super().neuron_compiler_args(phase, default_args)
        model_type = os.environ.get("MUSE_GLIMMER_NEURON_MODEL_TYPE", "")
        if model_type:
            args = [a for a in args if not a.startswith("--model-type")] + [f"--model-type={model_type}"]
            logger.info("%s %s compiler arguments: %s", self.name, phase, args)
        return args

    def build_graph(self, phase: str, config) -> nn.Module:
        return MuseGlimmerGraph(text_config(config), phase)


EntryClass = [MuseGlimmerForConditionalGeneration]
