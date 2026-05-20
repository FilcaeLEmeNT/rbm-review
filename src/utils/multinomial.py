import numpy as np
import math
import torch
import torch.nn.functional as F

class OneHotTransform:
    def __init__(self, q):
        self.q = q

    def __call__(self, x):
        # x is a 1D tensor of integers: shape (L*L,)
        # Output will be (L*L, q) or with transpose (q, L*L)
        return F.one_hot(x.long(), num_classes=self.q).float()  #.transpose(0, 1)

class DiscretizeTransform:
    def __init__(self, bins):
        self.bins = bins
    
    def __call__(self, x):
        # x: grayscale image, shape e.g. [B, 32, 32] in [0,1]
        x_bin = (x * self.bins).floor().clamp(max=self.bins - 1).long()  # [1024], ints 0,1,2,...
    
        return x_bin  # [nv, ]

def onehot_to_categories(x, data_size, bins=10):
    return x.view(-1, data_size, bins).argmax(dim=-1)

def categories_to_grayscale(x_cat, bins=10):
    x_float = (x_cat.float() + 0.5) / bins  # map bins -> midpoints in [0,1]
    return x_float