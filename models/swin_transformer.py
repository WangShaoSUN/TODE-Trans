""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030
Code/weights from https://github.com/microsoft/Swin-Transformer
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import Parameter
from mindspore.common.initializer import TruncatedNormal, Constant, initializer
import numpy as np
from typing import Optional
import mindspore.numpy as mnp
from mindspore.ops import operations as P

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) for MindSpore.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + ops.uniform(shape, Tensor(0, mindspore.float32), Tensor(1, mindspore.float32))
    random_tensor = ops.floor(random_tensor)  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) for MindSpore.
    """
    def __init__(self, drop_prob: Optional[float] = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    Partition feature map into non-overlapping windows in MindSpore.
    Args:
        x: Tensor of shape (B, H, W, C).
        window_size (int): Window size (M).
    Returns:
        Tensor of shape (num_windows * B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.transpose((0, 1, 3, 2, 4, 5)).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Reverse window partitioning into feature map in MindSpore.
    Args:
        windows: Tensor of shape (num_windows * B, window_size, window_size, C).
        window_size (int): Window size (M).
        H (int): Height of the original image.
        W (int): Width of the original image.
    Returns:
        Tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.transpose((0, 1, 3, 2, 4, 5)).reshape(B, H, W, -1)
    return x

class PatchEmbed(nn.Cell):
    """
    2D Image to Patch Embedding in MindSpore.
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=[96], norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer([embed_dim], begin_norm_axis=-1) if norm_layer else nn.Identity()

    def construct(self, x):
        _, _, H, W = x.shape

        # Padding if H and W are not multiples of patch_size
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            pad_op = ops.Pad(((0, 0), (0, self.patch_size[0] - H % self.patch_size[0]),
                              (0, self.patch_size[1] - W % self.patch_size[1]), (0, 0)))
            x = pad_op(x)

        # Downsample by patch_size
        x = self.proj(x)
        _, _, H, W = x.shape

        # Flatten and transpose
        x = x.view(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Cell):
    """
    Patch Merging Layer for MindSpore.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Cell, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Dense(4 * dim, 2 * dim, has_bias=False)
        self.norm = norm_layer((4 * dim,), begin_norm_axis=-1)

    def construct(self, x, H, W):
        """
        Args:
            x: Tensor of shape (B, H*W, C)
            H: Height of the input
            W: Width of the input
        Returns:
            Merged patches as a tensor.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Padding if H or W is not divisible by 2
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            pad_op = ops.Pad(((0, H % 2), (0, W % 2), (0, 0), (0, 0)))
            x = pad_op(x)
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = ops.concat((x0, x1, x2, x3), -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x.transpose(0, 1, 2)).transpose(0, 1, 2)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

class Mlp(nn.Cell):
    """
    MLP as used in Vision Transformer, MLP-Mixer, and related networks in MindSpore.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)  # MindSpore uses keep_prob
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop2 = nn.Dropout(p=drop)  # MindSpore uses keep_prob

    def construct(self, x):
        """
        Forward pass for the MLP.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Cell):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted windows.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = Parameter(
            Tensor(np.zeros(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)), dtype=mindspore.float32))

        coords_h = mnp.arange(0, self.window_size[0])
        coords_w = mnp.arange(0, self.window_size[1])
        grid_shape = (coords_h.shape[0], coords_w.shape[0])
        coords = mnp.stack(mnp.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, Mh, Mw]

        # Flatten the coordinates
        coords_flatten = P.Reshape()(coords, (2, -1))  # [2, Mh*Mw]

        # Compute relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = P.Transpose()(relative_coords, (1, 2, 0)).astype(mindspore.float32)  # [Mh*Mw, Mh*Mw, 2]

        # Adjust coordinates
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        # Compute relative position index
        relative_position_index = P.ReduceSum()(relative_coords, -1)  # [Mh*Mw, Mh*Mw]
        self.relative_position_index = Tensor(relative_position_index, dtype=mindspore.int32)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, mask: Optional[Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = ops.Split(0, 3)(qkv)

        q = q * self.scale
        attn = ops.matmul(q, k.transpose(0, 1, 2, 4, 3))
        relative_position_bias = ops.reshape(self.relative_position_bias_table[self.relative_position_index.view(-1)], (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(0, 2, 1, 3, 4).view(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Cell):
    """
    Swin Transformer Block for MindSpore.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Cell, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer([dim], begin_norm_axis=-1)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([dim], begin_norm_axis=-1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def roll(self, tensor, shifts, dims):
        """
        Mimics the behavior of torch.roll in MindSpore.
        Args:
            tensor (Tensor): The input tensor.
            shifts (tuple): The number of places by which the elements of the tensor are shifted.
            dims (tuple): The dimensions over which to shift.

        Returns:
            Tensor: The shifted tensor.
        """
        for dim, shift in zip(dims, shifts):
            if shift == 0:
                continue

            if shift < 0:
                shift = tensor.shape[dim] + shift

            indices = tuple(range(shift, tensor.shape[dim])) + tuple(range(shift))
            tensor = ops.gather(tensor, Tensor(indices), dim)
        return tensor

    def construct(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x.transpose(0, 1, 2)).transpose(0, 1, 2)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = ops.Pad(((0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)))(x)
        _, Hp, Wp, _ = x.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
            attn_mask = None
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll(shifted_x, (-self.shift_size, self.shift_size), (1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x.transpose(0, 1, 2)).transpose(0, 1, 2)))

        return x

class BasicLayer(nn.Cell):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim // 2, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # Calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = Tensor(np.zeros((1, Hp, Wp, 1)), x.dtype)
        h_slices = [slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)]
        w_slices = [slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)]
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = P.Reshape()(mask_windows, (-1, self.window_size * self.window_size))
        # Expand mask dimensions
        attn_mask = mask_windows.expand_dims(1) - mask_windows.expand_dims(2)

        # Replace non-zero values with -100.0
        attn_mask = ops.masked_fill(attn_mask, attn_mask != 0, -100.0)

        # Replace zero values with 0.0
        attn_mask = ops.masked_fill(attn_mask, attn_mask == 0, 0.0)

        return attn_mask

    def construct(self, x, H, W):
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        return x, H, W

class SwinTransformer(nn.Cell):
    """
    Swin Transformer in MindSpore.
    """
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super(SwinTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [drop_path_rate * i / sum(depths) for i in range(sum(depths))]

        # Build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer > 0) else None)
            self.layers.append(layer)

        self.norm = norm_layer([self.num_features], begin_norm_axis=-1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(mnp.zeros(cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(mnp.ones(cell.gamma.shape))
                cell.beta.set_data(mnp.zeros(cell.beta.shape))

    def construct(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        res = []
        for layer in self.layers:
            x, H, W = layer(x, H, W)
            B, L, C = x.shape
            res.append(x.view(B, H, W, C).permute(0, 3, 1, 2))
        return res
