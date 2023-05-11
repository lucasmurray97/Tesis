from typing import Optional
import torch
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import  reduce
import cv2
import torchvision
import torchvision.transforms.functional as F
import numpy as np
class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        self.device = torch.device('cpu')
        logits.to(self.device)
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype
            ).to(self.device)
            logits = torch.where(self.mask.to(self.device), logits, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

        

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
    
def generate_mask(state, forbidden_cells, version):
    device = torch.device('cpu')
    masks = []
    for i in range(state.shape[0]):
        masks.append(generate_mask_indiv(state[i], forbidden_cells = forbidden_cells, version=version))
    return torch.stack(masks).to(device)


def generate_mask_indiv(state, forbidden_cells, version):
    if version == 2:
        size = int(np.sqrt(state.shape[1]))
        state = F.resize(state, size, interpolation = torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
    size = state.shape[1]
    mask = torch.ones(size**2)
    flat_state = torch.flatten(state, 1, 2)[0].int().squeeze()
    for i in range(size**2):
        if flat_state[i] == 1:
            mask[i] = 0
    for j in forbidden_cells:
        mask[j] = 0
    return mask.bool()

class Q_Mask:
    def __init__(self, forbidden_cells, version):
        self.device = torch.device('cpu')
        self.forbidden_cells = forbidden_cells
        self.version = version

    def filter_indiv(self, state):
        if self.version == 2:
            state = state[:1]
        flat_state = torch.flatten(state, 0, 2).int().squeeze().bool()
        mask = ~flat_state
        for j in self.forbidden_cells:
            mask[j] = False
        return mask

    def filter(self, state):
        masks = []
        for i in range(state.shape[0]):
            masks.append(self.filter_indiv(state[i]))
        return torch.stack(masks).to(self.device)
    
