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

from .token_merging import pad_x_and_mask  



def window_partition(t: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition tensor @t into chunks of size @window_size.

    @t's sequence length must be divisible by @window_size.

    Args:
        t: A tensor of shape [batch_size, sequence_length, embed_dim].
        window_size: The desired window size.

    Returns:
        A tensor of shape [batch_size * sequence_length // window_size,
        window_size, embed_dim].
    """
    B, N, C = t.shape

    if not N % window_size == 0:
        raise ValueError(
            f"sequence length {N} must be divisible by window size {window_size}"
        )

    t = t.reshape(B * N // window_size, window_size, C)
    return t

def window_partition_reverse(
    t: torch.Tensor, B: int, num_windows: int, C: int
) -> torch.Tensor:
    """
    Undo the @window_partition operation.

    Args:
        t: The input tensor of shape [batch_size * num_windows, window_size,
            embed_dim].
        B: The batch size.
        num_windows: The number of windows.
        C: The embedding dimension.

    Returns:
        A tensor of shape [batch_size, num_windows * window_size, embed_dim].
    """
    t = t.reshape(B, num_windows * t.shape[1], C)
    return t

def get_windows_shift_mask(
    N: int, window_size: int, window_shift: int, device: torch.device
) -> torch.Tensor:
    """
    Get the mask window required due to window shifting (needed for shifted
    window attention).

    This produces a tensor with mask values for each window. Most windows don't
    require masking, but windows that bleed across the beginning/end of the
    tensor (due to shifting) require it.

    Args:
        N: The sequence length.
        window_size: The window size.
        window_shift: The window shift.
        device: The device on which to create the tensor.

    Returns:
        A tensor of shape [N // window_size, window_size, window_size]
        containing mask values. The values are 0 (unmasked) or float("-inf")
        (masked).
    """
    ret = torch.zeros(N // window_size, window_size, window_size, device=device)
    ret[-1].fill_(float("-inf"))
    ret[-1, : window_size - window_shift, : window_size - window_shift] = 0
    ret[-1, -window_shift:, -window_shift:] = 0
    return ret

def window_x_and_key_padding_mask(
    x: torch.Tensor, key_padding_mask: torch.Tensor, window_size: int, window_shift: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform windowing on @x and @key_padding_mask in preparation for windowed
    attention.

    Args:
        x: The input tensor of shape [batch_size, sequence_length, num_channels].
        key_padding_mask: The mask, as a tensor of shape [batch_size, sequence_length].
        window_size: The window size to be used for windowed attention.
        window_shift: The window shift to be used for windowed attention.

    Returns:
        A tuple containing 3 tensors. The first is the windowed input. The second
        is the windowed mask. The third is the mask needed to perform shifted
        window attention (to avoid the first and last windows from bleeding
        into each other).
    """
    B, N = key_padding_mask.shape
    assert x.shape[:2] == (B, N)

    x, key_padding_mask = pad_x_and_mask(x, key_padding_mask, window_size)

    # Now, perform the windowing.
    if window_shift > 0:
        x = torch.roll(x, shifts=(-window_shift), dims=1)
        key_padding_mask = torch.roll(key_padding_mask, shifts=(-window_shift), dims=1)

    x_windows = window_partition(x, window_size)
    token_mask_windows = key_padding_mask.reshape(
        B * x.shape[1] // window_size, window_size
    )
    window_mask = get_windows_shift_mask(
        x.shape[1], window_size, window_shift, x_windows.device
    ).expand(B, -1, -1, -1)
    window_mask = window_mask.reshape(
        window_mask.shape[0] * window_mask.shape[1],
        window_mask.shape[2],
        window_mask.shape[3],
    )

    return x_windows, token_mask_windows, window_mask

def unwindow_x(x_windows: torch.Tensor, B: int, N: int, C: int, window_shift: int):
    """
    Undoes the operation of @window_x_and_attention on the input tensor @x_windows.

    Args:
        x_windows: The input tensor to unwindow. Its shape is [batch_size *
              padded_sequence_length // window_size, window_size, embed_dim].
        B: The batch size. Referred to as batch_size in this docstring.
        N: The sequence length of the tensor before windowing. Referred to as
            sequence_length in this docstring.
        C: The number of channels. Referred to as embed_dim in this docstring.
        window_shift: The shift applied to the sequence before the windowing
            originally occurred.

    Returns:
        A tensor of shape [batch_size, sequence_length, embed_dim].
    """
    num_windows = x_windows.shape[0] // B
    x = window_partition_reverse(x_windows, B, num_windows, C)

    if window_shift > 0:
        x = torch.roll(x, shifts=window_shift, dims=1)
    x = x[:, :N]

    return x
