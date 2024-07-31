import torch
import torch.nn.functional as F
from transformers import (
    LogitsProcessor,
)


class CFGLogits(LogitsProcessor):
    def __init__(
        self,
        guidance_scale,
        uncond,
        model,
        image=None,
        input_type="inputs_ids",
        start_layer=0,
        end_layer=32,
    ):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.image = image
        self.out = None
        self.input_type = input_type
        self.start_layer = start_layer
        self.end_layer = end_layer

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores
        for i in range(self.start_layer, self.end_layer):
            self.model.model.layers[i].self_attn.use_cfg = True

        if self.out is None:
            if self.input_type == "inputs_ids":
                self.out = self.model(self.uncond, use_cache=True)
            elif self.input_type == "inputs_embeds":
                self.out = self.model(inputs_embeds=self.uncond, use_cache=True)
            else:
                print("Neither input_ids nor inputs_embeds is provided.")
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        for i in range(self.start_layer, self.end_layer):
            self.model.model.layers[i].self_attn.use_cfg = False

        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)

        cutoff = torch.log(torch.tensor(0.1)) + scores.max(dim=-1, keepdim=True).values
        out = (
            self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        )
        cd_logits = out.masked_fill(scores < cutoff, -float("inf"))
        return cd_logits
