import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    ### PAI's modification
    if hasattr(self, "use_attn"):
        use_attn = self.use_attn
        img_start_idx = self.img_start_idx
        img_end_idx = self.img_end_idx
    else:
        use_attn = False

    if hasattr(self, "use_cfg"):
        use_cfg = self.use_cfg
    else:
        use_cfg = False

    if use_attn and not use_cfg:
        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * self.alpha
            + attn_weights[:, :, -1, img_start_idx:img_end_idx]
        )
    ### PAI's modification

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_modify(model, start_layer, end_layer, use_attn, alpha, use_cfg,
                 img_start_idx, img_end_idx):
    for i in range(start_layer, end_layer):
        model.model.layers[i].self_attn.use_attn = use_attn
        model.model.layers[i].self_attn.alpha = alpha
        model.model.layers[i].self_attn.use_cfg = use_cfg
        model.model.layers[i].self_attn.img_start_idx = img_start_idx
        model.model.layers[i].self_attn.img_end_idx = img_end_idx
        model.model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.model.layers[i].self_attn)
