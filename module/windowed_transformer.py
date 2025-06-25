
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

from .utils import window_x_and_key_padding_mask,unwindow_x

class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: Command line arguments.
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
    ) -> None:

        super().__init__()

        attn_unit = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True
            )
        '''
        torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
        
        '''

        self.pre_norm_mha = nn.Sequential(
            LayerNorm(embed_dim),
            attn_unit,
            Dropout(p=dropout),
        )

        # act_name = build_activation_layer(opts, num_parameters=1)
        act_name=ReLU()
        self.pre_norm_ffn = nn.Sequential(
            LayerNorm(embed_dim),
            Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            Dropout(p=ffn_dropout),
            Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout),
        )

        self.drop_path = Identity()

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = stochastic_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.stochastic_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x,_ = self.pre_norm_mha[1](
            query=x,
            key=x_prev if x_prev else x,
            value=x_prev if x_prev else x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            *args,
            **kwargs,
        )  # mha
        '''
        query=x_q,
        key=x_kv,
        value=x_kv,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=False
        
        '''
        x = self.drop_path(self.pre_norm_mha[2](x))  # applying stochastic depth
        x = x + res

        # Feed forward network
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x

class WindowedTransformerEncoder(TransformerEncoder):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    with the addition of windowed attention.

    This class first partitions the input sequence into a series of windows (with
    an optional offset to use when defining windows). Then, it calls a
    TransformerEncoder module. Then, it undoes windowing.

    Args:
        opts: Command line arguments.
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0.
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.
        window_size: The size of the window, if using windowed attention. Default: None.
        window_shift: The size of the shift, if using shifted windowed attention. Default: None.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        window_size: Optional[int] = None,
        window_shift: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts,
            embed_dim=embed_dim,
            ffn_latent_dim=ffn_latent_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            transformer_norm_layer=transformer_norm_layer,
            stochastic_dropout=stochastic_dropout,
        )
        
        
        if window_size is None:
            raise ValueError("Please specify window_size")
        if window_shift is None:
            raise ValueError("Please specify window_shift")
        self.window_size: int = window_size
        self.window_shift: int = window_shift

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the outputs of the WindowedTransformerEncoder on an input.

        Args:
            x: The input tensor, of shape [batch_size, sequence_length, embed_dim].
            x_prev: The context input, if using cross-attention. Its shape is
                [batch_size, sequence_length_2, embed_dim].
            key_padding_mask: An optional tensor of masks to be applied to the
                inputs @x. Its shape is [batch_size, sequence_length].
            attn_mask: An optional attention mask. Its shape is [batch_size,
                sequence_length, sequence_length_2]. (If using self-attention,
                the sequence lengths will be equal.)

        Returns:
            The WindowedTransformerEncoder output.
        """
        B, N, C = x.shape
        x, windowed_key_padding_mask, windows_mask = window_x_and_key_padding_mask(
            x, key_padding_mask, self.window_size, self.window_shift
        )
        total_mask = windowed_key_padding_mask.unsqueeze(1) + windows_mask

        if attn_mask is not None:
            total_mask += attn_mask

        # If an entire window is masked out, attention is computed across
        # only -inf values, which gives NaN. We instead set these masks to
        # 0 to avoid this.
        fully_masked_windows = total_mask.max(dim=-1).values == float("-inf")
        total_mask[fully_masked_windows] = 0

        x = super().forward(x, x_prev, attn_mask=attn_mask)

        # Undo windowing.
        x = unwindow_x(x, B, N, C, self.window_shift)
        return x

    def __repr__(self) -> str:
        # Remove closing parentheses from parent __repr__ call.
        ret = super().__repr__()[:-1]
        return f"{ret}, {self.window_size}, {self.window_shift})"