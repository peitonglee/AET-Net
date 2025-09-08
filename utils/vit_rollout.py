import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def rollout(attentions, discard_ratio, head_fusion, model_name, local=False, img_size=(16, 8)):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for index, attention in enumerate(attentions):
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise("Attention head fusion type Not supported")

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            if local and (index >= (len(model_name) - 3)):
                break
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    mask = mask.reshape(img_size[0], img_size[1]).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    # def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
    #     discard_ratio=0.9, local=False):
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9, local=False, cfg=None):

        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.local = local
        self.model_name = []
        self.img_size = [model.base.patch_embed.num_y, model.base.patch_embed.num_x]
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                self.model_name.append(name)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return rollout(self.attentions, self.discard_ratio, self.head_fusion, self.model_name, self.local, img_size=self.img_size)