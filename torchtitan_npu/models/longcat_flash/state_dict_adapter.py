import torch
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class LongCatFlashStateDictAdapter(StateDictAdapter):
    """Adapts between the HuggingFace LongCat-Flash-Chat checkpoint format and
    the torchtitan LongCatFlashModel state dict layout.

    HF keys: model.embed_tokens, model.layers.{L}.self_attn.{0,1}.*, model.layers.{L}.mlps.{0,1}.*,
             model.layers.{L}.mlp.experts.{E}.*, model.layers.{L}.mlp.router.*,
             model.layers.{L}.input_layernorm.{0,1}.*, model.layers.{L}.post_attention_layernorm.{0,1}.*,
             model.norm.*, lm_head.*

    Titan keys: tok_embeddings.*, layers.{L}.self_attn.{0,1}.*, layers.{L}.mlps.{0,1}.*,
                layers.{L}.moe.experts.w1[E], layers.{L}.moe.gate.*, layers.{L}.moe.e_score_correction_bias,
                layers.{L}.input_layernorm.{0,1}.*, layers.{L}.post_attention_layernorm.{0,1}.*,
                norm.*, output.*
    """

    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

    def from_hf(self, hf_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        titan_sd: dict[str, torch.Tensor] = {}
        w13_parts: dict[str, dict[str, torch.Tensor]] = {}
        for hf_key, tensor in hf_sd.items():
            titan_key = _hf_key_to_titan(hf_key)
            if titan_key is None:
                continue
            if ":expert_" in titan_key and titan_key.endswith(
                (":gate_proj.weight", ":up_proj.weight")
            ):
                base, tag = titan_key.rsplit(":", 1)
                w13_parts.setdefault(base, {})[tag] = tensor
            else:
                titan_sd[titan_key] = tensor

        for base, parts in w13_parts.items():
            gate = parts.get("gate_proj.weight")
            up = parts.get("up_proj.weight")
            if gate is not None and up is not None:
                titan_sd[base] = torch.cat([gate, up], dim=0)

        return titan_sd

    def to_hf(self, titan_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hf_sd: dict[str, torch.Tensor] = {}
        for titan_key, tensor in titan_sd.items():
            if ".moe.experts.w13:expert_" in titan_key:
                _export_w13_to_hf(titan_key, tensor, hf_sd)
            else:
                hf_key = _titan_key_to_hf(titan_key)
                if hf_key is not None:
                    hf_sd[hf_key] = tensor
        return hf_sd


def _hf_key_to_titan(key: str) -> str | None:
    if key.startswith("model.embed_tokens."):
        return key.replace("model.embed_tokens.", "tok_embeddings.")
    if key.startswith("model.norm."):
        return key.replace("model.norm.", "norm.")
    if key.startswith("lm_head."):
        return key.replace("lm_head.", "output.")
    if key.startswith("model.layers."):
        return _convert_layer_key_hf_to_titan(key[len("model.layers."):])
    return None


def _convert_layer_key_hf_to_titan(suffix: str) -> str | None:
    parts = suffix.split(".", 1)
    layer_idx = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    prefix = f"layers.{layer_idx}"

    if rest.startswith("mlp.router.classifier."):
        return f"{prefix}.moe.gate.{rest[len('mlp.router.classifier.'):]}"
    if rest == "mlp.router.e_score_correction_bias":
        return f"{prefix}.moe.e_score_correction_bias"

    if rest.startswith("mlp.experts."):
        return _convert_expert_key(prefix, rest[len("mlp.experts."):])

    return f"{prefix}.{rest}"


def _convert_expert_key(prefix: str, suffix: str) -> str | None:
    parts = suffix.split(".", 1)
    expert_idx = int(parts[0])
    param_rest = parts[1] if len(parts) > 1 else ""

    if param_rest == "down_proj.weight":
        return f"{prefix}.moe.experts.w2:expert_{expert_idx}"
    if param_rest in ("gate_proj.weight", "up_proj.weight"):
        return f"{prefix}.moe.experts.w13:expert_{expert_idx}:{param_rest}"
    return None


def _titan_key_to_hf(key: str) -> str | None:
    if ":" in key and ".moe.experts." in key:
        return _convert_expert_key_titan_to_hf(key)
    if key.startswith("tok_embeddings."):
        return key.replace("tok_embeddings.", "model.embed_tokens.")
    if key.startswith("norm."):
        return key.replace("norm.", "model.norm.")
    if key.startswith("output."):
        return key.replace("output.", "lm_head.")
    if key.startswith("layers."):
        parts = key.split(".", 2)
        layer_idx = parts[1]
        rest = parts[2] if len(parts) > 2 else ""
        hf_prefix = f"model.layers.{layer_idx}"

        if rest.startswith("moe.gate."):
            return f"{hf_prefix}.mlp.router.classifier.{rest[len('moe.gate.'):]}"
        if rest == "moe.e_score_correction_bias":
            return f"{hf_prefix}.mlp.router.e_score_correction_bias"

        return f"{hf_prefix}.{rest}"
    return None


def _convert_expert_key_titan_to_hf(key: str) -> str | None:
    """Convert titan expert key like 'layers.0.moe.experts.w2:expert_3' to HF format."""
    base, expert_tag = key.split(":", 1)
    expert_idx = int(expert_tag.replace("expert_", ""))

    parts = base.split(".", 2)
    layer_idx = parts[1]
    rest = parts[2] if len(parts) > 2 else ""

    param_map = {"moe.experts.w2": "down_proj.weight"}
    if rest in param_map:
        return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{param_map[rest]}"
    return None


def _export_w13_to_hf(
    titan_key: str, tensor: torch.Tensor, hf_sd: dict[str, torch.Tensor]
) -> None:
    """Split fused w13 tensor back into gate_proj and up_proj for HF format."""
    base = titan_key.split(":expert_")[0]
    expert_tag = titan_key.split(":expert_")[1].split(":")[0]
    expert_idx = int(expert_tag)

    parts = base.split(".", 2)
    layer_idx = parts[1]
    hf_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"

    half = tensor.shape[0] // 2
    hf_sd[f"{hf_prefix}.gate_proj.weight"] = tensor[:half]
    hf_sd[f"{hf_prefix}.up_proj.weight"] = tensor[half:]
