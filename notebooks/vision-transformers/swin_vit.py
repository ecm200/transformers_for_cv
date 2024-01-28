# Inspired by:
# https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03
# https://github.com/GSaiDheeraj/Swin-Transformers-from-scratch-for-classification/tree/main
# 
# After the paper: https://arxiv.org/pdf/2103.14030.pdf

#
# %% IMPORTS
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# %% Script Variables
dataset_root_dir = "/home/ecm200/projects/deep_learning/datasets/image_classification/caltech_birds_200/images"
input_image_size = (3,224,224)

# %% Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    Resize,
    RandomCrop,
    CenterCrop,
    ToTensor
)


transform = Compose([
    Resize(size=(input_image_size[1],input_image_size[2])),
    ToTensor()
])

dataset = ImageFolder(
    root=dataset_root_dir,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
# %% Patch Embedding
#
# It first splits an input RGB image into non-overlapping
# patches by a patch splitting module, like ViT. Each patch is
# treated as a “token” and its feature is set as a concatenation
# of the raw pixel RGB values. In our implementation, we use
# a patch size of 4 × 4 and thus the feature dimension of each
# patch is 4 × 4 × 3 = 48. A linear embedding layer is applied on 
# this raw-valued feature to project it to an arbitrary
# dimension.
class SwinEmbedding(nn.Module):

    '''
    input shape -> (b,c,h,w)
    output shape -> (b, (h/4 * w/4), C)
    '''

    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.relu(self.layer_norm(x))
        return x
    
# %% Embeddings
import matplotlib.pylab as plt
imgs, labels = next(iter(dataloader))
X = imgs[0, :, :, :]

plt.figure(figsize=(10,10))
plt.imshow(X.permute([1,2,0]))
plt.show()

swin_embedding = SwinEmbedding()
embeddings = swin_embedding(imgs[0, :, : , :].expand(size=(1,3,224,224)))

plt.figure(figsize=(10,10))
plt.pcolormesh(embeddings.detach().numpy().squeeze())
plt.show()

# %% Patch Merging
# To produce a hierarchical representation, the number of tokens is reduced by patch 
# merging layers as the network gets deeper. The first patch merging layer concatenates 
# the features of each group of 2 × 2 neighboring patches, and applies a linear layer on the 
# 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2×2 = 4 
# (2× downsampling of resolution), and the output dimension is set to 2C.

class PatchMerging(nn.Module):

    '''
    input shape -> (b, (h*w), C)
    output shape -> (b, (h/2 * w/2), C*2)
    '''

    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(4*C, 2*C)
        self.layer_norm = nn.LayerNorm(2*C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1])/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))

# %% SW-MSA (Shifted Window Multi-head Self Attention)
# Compute self-attention within local windows. 
# The windows are arranged to evenly partition
# the image in a non-overlapping manner. Each
# window contains M × M patches, the computational complexity 
# of a global MSA module and a window based one on an image of h × w patches.

class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7, mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        # self.embeddings = RelativeEmbeddings()

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        if self.mask:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4,5)) / math.sqrt(h_dim)
        # att_scores = self.embeddings(att_scores)

        '''
          shape of att_scores = (b, H, h, w, (m1*m2), (m1*m2))
          we simply have to generate our row/column masks and apply them
          to the last row and columns of windows which are [:,:,-1,:] and [:,:,:,-1]
        '''

        if self.mask:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).cuda()
            att_scores[:, :, -1, :] += row_mask
            att_scores[:, :, :, -1] += column_mask

        att = F.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)

        if self.mask:
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1,2))

        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.proj2(x)


# %% Relative Embeddings

# In computing self-attention, we follow [49, 1, 32, 33] by including a relative position bias B ∈ R(M2×M2) to 
# each head in computing similarity: Attention(Q, K, V ) = SoftMax(QKT / √ d + B)V, where Q, K, V ∈ R(M2×d) are 
# the query, key and value matrices; d is the query/key dimension, and M2 is the number of patches in a window. 
# Since the relative position along each axis lies in the range [−M + 1, M −1], we parameterize a smaller-sized 
# bias matrix Bˆ ∈ R (2M−1)×(2M−1), and values in B are taken from Bˆ.

class RelativeEmbeddings(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        B = nn.Parameter(torch.randn(2*window_size-1, 2*window_size-1))
        x = torch.arange(1,window_size+1,1/window_size)
        x = (x[None, :]-x[:, None]).int()
        y = torch.concat([torch.arange(1,window_size+1)] * window_size)
        y = (y[None, :]-y[:, None])
        self.embeddings = nn.Parameter((B[x[:,:], y[:,:]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings

# %% SWIN Transform Block

class SwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.WMSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=mask)
        self.MLP1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        height, width = x.shape[1:3]
        res1 = self.dropout(self.WMSA(self.layer_norm(x)) + x)
        x = self.layer_norm(res1)
        x = self.MLP1(x)
        return self.dropout(x + res1)
    
class AlternatingEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.WSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=False)
        self.SWSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=True)
    
    def forward(self, x):
        return self.SWSA(self.WSA(x))
 
# %% Swin Transformer Model

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = SwinEmbedding()
        self.Embedding = SwinEmbedding()
        self.PatchMerge1 = PatchMerging(96)
        self.PatchMerge2 = PatchMerging(192)
        self.PatchMerge3 = PatchMerging(384)
        self.Stage1 = AlternatingEncoderBlock(96, 3)
        self.Stage2 = AlternatingEncoderBlock(192, 6)
        self.Stage3_1 = AlternatingEncoderBlock(384, 12)
        self.Stage3_2 = AlternatingEncoderBlock(384, 12)
        self.Stage3_3 = AlternatingEncoderBlock(384, 12)
        self.Stage4 = AlternatingEncoderBlock(768, 24)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PatchMerge1(self.Stage1(x))
        x = self.PatchMerge2(self.Stage2(x))
        x = self.Stage3_1(x)
        x = self.Stage3_2(x)
        x = self.Stage3_3(x)
        x = self.PatchMerge3(x)
        x = self.Stage4(x)
        return x
