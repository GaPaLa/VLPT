from typing import Dict, Optional

import torch as th
from torch import nn
from torch.nn import functional as F

import lib.torch_util as tu
from lib.masked_attention import MaskedAttention
from lib.minecraft_util import store_args
from lib.tree_util import tree_map
#
from torch import float32


def get_module_log_keys_recursive(m: nn.Module):
    """Recursively get all keys that a module and its children want to log."""
    keys = []
    if hasattr(m, "get_log_keys"):
        keys += m.get_log_keys()
    for c in m.children():
        keys += get_module_log_keys_recursive(c)
    return keys






class FanInInitReLULayer(nn.Module):
    """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation
    :param inchan: number of input channels
    :param outchan: number of output channels
    :param layer_args: positional layer args
    :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
    :param init_scale: multiplier on initial weights
    :param batch_norm: use batch norm after the layer (for 2D data)
    :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
        would be equivalent of layernorm for 2D data.
    :param layer_norm: use layernorm after the layer (for 1D data)
    :param layer_kwargs: keyword arguments for the layer
    """

    @store_args
    def __init__(
        self,
        inchan: int,
        outchan: int,
        *layer_args,
        layer_type: str = "conv",
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        dtype=float32,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, dtype=dtype, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan, dtype=dtype)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan, dtype=dtype)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, dtype=dtype, *layer_args, **layer_kwargs)

        # Init Weights (Fan-In)
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        # Init Bias
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x):
        """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

    def get_log_keys(self):
        return [
            f"activation_mean/{self.log_scope}",
            f"activation_std/{self.log_scope}",
        ]







class ResidualRecurrentBlocks(nn.Module):
    @store_args
    def __init__(
        self,
        n_block=4,
        recurrence_type="transformer",
        is_residual=True,
        **block_kwargs,
    ):
        super().__init__()
        init_scale = n_block ** -0.5
        self.blocks = nn.ModuleList(
            [
                ResidualRecurrentBlock(
                    **block_kwargs,
                    recurrence_type='transformer',
                    is_residual=is_residual,
                    init_scale=init_scale,
                    block_number=i,
                )
                for i in range(n_block)
            ]
        )

        
    def forward(self, x, first, state, start_block=0, end_block=4):
        state_out = []
        assert len(state) == len(self.blocks[start_block:end_block]), f"Length of state {len(state)} did not match length of blocks {len(self.blocks[start_block:end_block])}"
        for block, _s_in in zip(self.blocks[start_block:end_block], state):
            x, _s_o = block(x, first, _s_in)
            state_out.append(_s_o)
        return x, state_out

    def initial_state(self, batchsize):
        return [b.r.initial_state(batchsize) for b in self.blocks]








class ResidualRecurrentBlock(nn.Module):
    @store_args
    def __init__(
        self,
        hidsize,
        timesteps,
        init_scale=1,
        recurrence_type="transformer",
        is_residual=True,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
        attention_mask_style="clipped_causal",
        log_scope="resblock",
        block_number=0,
        dtype=float32
    ):
        super().__init__()
        self.log_scope = f"{log_scope}{block_number}"
        
        s = init_scale
        s *= 2 ** -0.5  # second residual
        
        self.mlp0 = FanInInitReLULayer(
            hidsize,
            hidsize * pointwise_ratio,
            init_scale=1,
            layer_type="linear",
            layer_norm=True,
            dtype=dtype,
            log_scope=self.log_scope + "/ptwise_mlp0",
        )
        self.mlp1 = FanInInitReLULayer(
            hidsize * pointwise_ratio,
            hidsize,
            init_scale=s,
            layer_type="linear",
            use_activation=pointwise_use_activation,
            dtype=dtype,
            log_scope=self.log_scope + "/ptwise_mlp1",
        )

        self.pre_r_ln = nn.LayerNorm(hidsize, dtype=dtype)
        
        self.r = MaskedAttention(
            input_size=hidsize,
            timesteps=timesteps,
            memory_size=attention_memory_size,
            heads=attention_heads,
            init_scale=s,
            norm="none",
            log_scope=log_scope + "/sa",
            use_muP_factor=True,
            mask=attention_mask_style,
            dtype=dtype
        )
        
        self.dropout=th.nn.Dropout(p=0.2, inplace=False)


    # define forward for a single transformer block
    def forward(self, x, first, state):
        
        x = self.pre_r_ln(x)

		# Attention
        x, state_out = recurrent_forward(
            self.r,
            x,
            first,
            state,
            reverse_lstm=False,
        )
        #x = self.dropout(x)   #### residual is handled internally instide MaskeedAttentino->xf.py->selfattention(). so dropout is applied inside here, to the output of attention before the residual is added back to the outpu
        
        
        # MLP
        residual = x
        x = self.mlp1(self.mlp0(x))
        x = self.dropout(x)
        
        #residual
        x = x + residual

        return x, state_out


def recurrent_forward(masked_attention, x, first, state, reverse_lstm=False):

    return masked_attention(x, first, state)






def _banded_repeat(x, t):
    """
    Repeats x with a shift.
    For example (ignoring the batch dimension):

    _banded_repeat([A B C D E], 4)
    =
    [D E 0 0 0]
    [C D E 0 0]
    [B C D E 0]
    [A B C D E]
    """
    b, T = x.shape
    x = th.cat([x, x.new_zeros(b, t - 1)], dim=1)
    result = x.unfold(1, T, 1).flip(1)
    return result


def bandify(b_nd, t, T):
    """
    b_nd -> D_ntT, where
        "n" indexes over basis functions
        "d" indexes over time differences
        "t" indexes over output time
        "T" indexes over input time
        only t >= T is nonzero
    B_ntT[n, t, T] = b_nd[n, t - T]
    """
    nbasis, bandsize = b_nd.shape
    b_nd = b_nd[:, th.arange(bandsize - 1, -1, -1)]
    if bandsize >= T:
        b_nT = b_nd[:, -T:]
    else:
        b_nT = th.cat([b_nd.new_zeros(nbasis, T - bandsize), b_nd], dim=1)
    D_tnT = _banded_repeat(b_nT, t)
    return D_tnT


def get_norm(name, d, dtype=th.float32):
    if name == "none":
        return lambda x: x
    elif name == "layer":
        return tu.LayerNorm(d, dtype=dtype)
    else:
        raise NotImplementedError(name)
