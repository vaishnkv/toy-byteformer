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

from .pos_embedding import SinusoidalPositionalEmbedding
from .windowed_transformer import WindowedTransformerEncoder
from .token_merging import TokenMerging
from torchviz import make_dot



def unfold_tokens(t: Tensor, kernel_size: int) -> Tensor:
    """
    Group tokens from tensor @t using torch.Tensor.unfold, using the given
    kernel size. This amounts to windowing @t using overlapping windows
    of size @kernel_size, with overlap of @kernel_size // 2.

    Args:
        t: A tensor of shape [batch_size, sequence_length, num_channels].
        kernel_size: The kernel size.

    Returns:
        A tensor of shape [batch_size * (sequence_length - kernel_size)
        // (kernel_size // 2) + 1, kernel_size, num_channels].
    """
    t = t.unfold(dimension=1, size=kernel_size, step=kernel_size // 2)
    B, L, C, _ = t.shape
    t = t.reshape(B * L, C, kernel_size)
    t = t.transpose(1, 2)
    return t


class ByteFormer(nn.Module):
    
    
    """
    This class defines the `ByteFormer <https://arxiv.org/abs/2306.00238>`_ architecture.
    """
        
    config : dict ={
        # "embed_dim": 192,
        "embed_dim": 256,
        "n_transformer_layers": 12,
        "n_attn_heads": 8,
        "ffn_dim": 256 * 4,
        # "ffn_dim": 192 * 4,
        "norm_layer": "layer_norm",
        "pos_emb_drop_p": 0.1,
        "attn_dropout": 0.1,
        "ffn_dropout": 0.1,
        "dropout": 0.1,
    }

    def __init__(self, opts,num_classes,downsample_map, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        embed_dim = ByteFormer.config["embed_dim"]
        ffn_dim = ByteFormer.config["ffn_dim"]
        n_transformer_layers = ByteFormer.config["n_transformer_layers"]
        num_heads = ByteFormer.config["n_attn_heads"]
        attn_dropout = ByteFormer.config["attn_dropout"]
        dropout = ByteFormer.config["dropout"]
        ffn_dropout = ByteFormer.config["ffn_dropout"]
        norm_layer = ByteFormer.config["norm_layer"]

        # This is usually 257 in the case of byte inputs (2**8 + 1 mask token).
        vocab_size = 257 # (2**8 + 1 mask token)
        # self.embeddings = embedding.Embedding(
        #     opts, num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=-1
        # )
        
        
        # This is only needed.
        self.embeddings=Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=-1)
        
        '''
        torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)[source]
        '''
        
        # Reinitialize everything except the padding index.
        init.trunc_normal_(self.embeddings.weight[:-1], std=math.sqrt(1.0 / embed_dim))

        self.dummy_input_token_length = 400

        # Add token reduction convolution.
        self.conv_kernel_size = 16
        # '''
        self.token_reduction_net = nn.Conv1d(
            embed_dim,
            ByteFormer.config["embed_dim"],
            kernel_size=self.conv_kernel_size,
            stride=self.conv_kernel_size // 2,
            bias=False,
        )
        # '''

        # Add the positional embeddings.
        self.max_num_tokens = 10000
        # self.max_num_tokens=57086
        
        self.pos_embed = SinusoidalPositionalEmbedding(
            opts=opts,
            num_embeddings=self.max_num_tokens,
            embedding_dim=embed_dim,
            sequence_first=False,
            padding_idx=None,
            interpolation_mode="bilinear",
        )
        
        '''
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        sequence_first: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear"
        '''

        pos_emb_drop_p = 0
        self.emb_dropout = nn.Dropout(p=pos_emb_drop_p)

        # Build the transformer backbone.
        window_sizes = [128]
        window_shifts = [0, 64] * 6
        # downsample = [True, True] + ([False, True] * 4) + [False, False]
        downsample = downsample_map
        if len(window_sizes) == 1:
            window_sizes = window_sizes * n_transformer_layers

        for x in [window_sizes, window_shifts, downsample]:
            if len(x) != n_transformer_layers:
                raise ValueError(
                    f"Invalid argument length {len(x)} != {n_transformer_layers}"
                )

        stochastic_dropout = 0
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]

        blocks = []
        self.downsamplers = nn.ModuleDict()
        
        
        
        for layer_idx in range(n_transformer_layers):
            blocks.append(
                WindowedTransformerEncoder(
                    opts=opts,
                    embed_dim=embed_dim,
                    ffn_latent_dim=ffn_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    ffn_dropout=ffn_dropout,
                    transformer_norm_layer=norm_layer,
                    stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
                    window_size=window_sizes[layer_idx],
                    window_shift=window_shifts[layer_idx],
                )
            )
            if downsample is not None and downsample[layer_idx]:
                self.downsamplers[self.get_downsampler_name(layer_idx)] = (
                    TokenMerging(embed_dim)
                )
        self.transformer = nn.Sequential(*blocks)
        # I Think , this is sufficient 
        self.post_transformer_norm=LayerNorm(embed_dim)

        self.classifier = Linear(embed_dim, num_classes)

    def dummy_input_and_label(self, sample_size: int) -> Dict:
        """
        Get a dummy input and label that could be passed to the model.

        Args:
            sample_size: The batch size to use for the generated inputs.

        Returns:
            A dict with
                {
                    "samples": tensor of shape [sample_size, sequence_length],
                    "targets": tensor of shape [sample_size],
                }
        """
        n_labels = 10
        max_value = 257

        samples = torch.randint(
            0, max_value, [sample_size, self.dummy_input_token_length]
        )
        targets = torch.randint(low=0, high=n_labels, size=(sample_size,)).long()
        return {"samples": samples, "targets": targets}

    def apply_token_reduction_net(
        self, x: Tensor, x_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply the portion of the network used to reduce sequence lengths before
        the transformer backbone.

        Args:
            x: The input token embeddings of shape [batch_size, sequence_length,
                embed_dim].
            x_mask: The input mask of shape [batch_size, sequence_length].

        Returns:
            New versions of @x and @x_mask, downsampled along the sequence
            dimension by the token reduction net.
        """
        B, N, C = x.shape
        if self.token_reduction_net is None:
            return x, x_mask

        x = self.token_reduction_net(x.permute(0, 2, 1)).permute(0, 2, 1)
        if x_mask is not None:
            x_mask = unfold_tokens(
                x_mask.reshape(B, N, 1).float(), self.conv_kernel_size
            )
            # The mask is now [B * N, kernel_size, 1]. It contains values in {0, -inf}.
            x_mask = x_mask.max(dim=1).values.view(x.shape[0], x.shape[1])

            assert x.shape[:2] == x_mask.shape
        return x, x_mask

    def get_backbone_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert input bytes into embeddings to be passed to the network's
        transformer backbone.

        Args:
            x: The input bytes as an integer tensor of shape [batch_size,
                sequence_length]. Integer tensors are expected (rather than byte
                tensors) since -1 is usually used for padding.

        Returns:
            The embeddings of shape [batch_size, new_sequence_length] and a
            mask tensor of shape [batch_size, new_sequence_length]. The mask
            contains 0 at unmasked positions and float(-inf) at masked
            positions.
        """
        mask = torch.zeros_like(x, dtype=torch.float)
        mask[x == -1].fill_(float("-inf"))
        mask = mask.detach().requires_grad_(False)
        x[x == -1] = self.embeddings.padding_idx
        # print(x.shape)
        # print()
        x = self.embeddings(x)
        # print(f"Before token_reduction :X 's shape {x.shape}")
        x, mask = self.apply_token_reduction_net(x, mask)
        # print(f"After token_reduction :X 's shape {x.shape}")
        # assert False
        x = x + self.pos_embed(self.max_num_tokens)[:, : x.shape[1]]

        x = self.emb_dropout(x)
        return x, mask

    def backbone_forward(
        self, x: Tensor, key_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Execute the forward pass of the network's transformer backbone.

        Args:
            x: The input embeddings as a [batch_size, sequence_length, embed_dim] tensor.
            key_padding_mask: The mask tensor of shape [batch_size, sequence_length].

        Returns:
            The outputs of the backbone as a tuple. The first element is the feature
            tensor, and the second element is the updated key_padding_mask.
        """
        B, S, _ = x.shape
        assert key_padding_mask.shape == (B, S)

        for layer_idx, elem in enumerate(self.transformer):
            # print(f"INPUT : X's Shape on Layer {layer_idx} is {x.shape}")
            x = elem(x, key_padding_mask=key_padding_mask)
            if self.get_downsampler(layer_idx) is not None:
                # print(f"X shape at the downsampling layer {layer_idx} is  {x.shape}")
                # print(f"Attention Mask shape is {key_padding_mask.shape}")
                x, key_padding_mask = self.get_downsampler(layer_idx)(
                    x, key_padding_mask
                )
                # print(f"X shape at the downsampling layer {layer_idx} is  {x.shape}")
                # print(f"Attention Mask shape is {key_padding_mask.shape}")
                # assert False
            # print(f"OUTPUT: X's Shape on Layer {layer_idx} is {x.shape}\n")
        x = self.post_transformer_norm(x)
        return x, key_padding_mask

    def get_downsampler_name(self, idx: int) -> str:
        """
        Get the name of the downsampling layer with index @idx.

        Args:
            idx: The index of the downsampling layer.

        Returns:
            A string representing the name of the donwsampling layer.
        """
        return f"downsample_{idx}"

    def get_downsampler(self, idx: int) -> Optional[nn.Module]:
        """
        Get the module that performs downsampling after transformer layer @idx.
        If no downsampling occurs after that layer, return None.

        Args:
            idx: The desired index.

        Returns:
            The downsampling layer, or None.
        """
        name = self.get_downsampler_name(idx)
        if name not in self.downsamplers:
            return None
        return self.downsamplers[name]

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass on input bytes. The tensor is
        stored as an integer tensor of shape [batch_size, sequence_length].
        Integer tensors are used because @x usually contains mask tokens.

        Args:
            x: The input tensor of shape [batch_size, sequence_length].

        Returns:
            The output logits.
        """
        # print(x.shape)
        x, key_padding_mask = self.get_backbone_inputs(x)
        # print(x.shape)
        # print(key_padding_mask.shape)
        # assert False
        x, attn_mask = self.backbone_forward(x, key_padding_mask)
        # make_dot(x.sum(), params=dict(list(self.named_parameters()))).render("graph", format="png")
        # print("Post attention block")
        # print(x.shape)
        # assert False
        attn_mask = attn_mask.view(x.shape[0], x.shape[1], 1)
        x[(attn_mask == float("-inf")).expand(-1, -1, x.shape[-1])] = 0
        norms = (attn_mask == 0).sum(dim=1)
        # print(x.shape)
        x = torch.sum(x, dim=1) / norms
        # print(x.shape)
        # print("Before classifier")
        # print(x.shape)
        # assert False
        x = self.classifier(x)
        # print("Post classifier")
        # print(x.shape)
        # assert False
        return x