
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Tuple,Optional,Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear,Embedding,MultiheadAttention
from torch.nn import LayerNorm,Dropout,ReLU,Identity,init
import math
from torch import Tensor
import numpy as np
import argparse



class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self,
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        sequence_first: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear"
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sequence_first = sequence_first
        self.interpolation_mode = interpolation_mode
        self.register_buffer("pos_embed", self.get_weights())

    def get_weights(self) -> Tensor:
        """Build sinusoidal embeddings. Adapted from Fairseq."""
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(self.num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(
            self.num_embeddings, -1
        )
        if self.embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(self.num_embeddings, 1)], dim=1)

        # set embeddings corresponding to padding index to 0
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb.unsqueeze(0).unsqueeze(0)

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        # scale pos embedding
        pos_embed = self.pos_embed

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        if self.sequence_first:
            # Input is of the form [Seq_len, Batch, Embedding_dim]
            return pos_embed.reshape(seq_len, 1, self.embedding_dim)
        else:
            # Input is of the form [Batch, Seq_len, Embedding_dim]
            return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return "{}(num_embeddings={}, embedding_dim={}, padding_idx={}, sequence_first={})".format(
            self.__class__.__name__,
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.sequence_first,
        )