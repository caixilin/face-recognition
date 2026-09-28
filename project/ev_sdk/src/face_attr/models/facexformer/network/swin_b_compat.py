"""Torchvision-compatible Swin-B feature extractor for older torchvision."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class _Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def _relative_position_index(window_size):
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    coords = torch.flatten(coords, 1)
    relative_coords = coords[:, :, None] - coords[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    return relative_coords.sum(-1).flatten()


def _shifted_window_attention(x, qkv, proj, relative_bias, window_size, num_heads, shift_size):
    batch, height, width, channels = x.shape
    pad_right = (window_size[1] - width % window_size[1]) % window_size[1]
    pad_bottom = (window_size[0] - height % window_size[0]) % window_size[0]
    x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
    _, padded_height, padded_width, _ = x.shape

    shift_h, shift_w = shift_size
    if window_size[0] >= padded_height:
        shift_h = 0
    if window_size[1] >= padded_width:
        shift_w = 0
    if shift_h or shift_w:
        x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))

    windows_per_image = (padded_height // window_size[0]) * (padded_width // window_size[1])
    x = x.view(
        batch,
        padded_height // window_size[0],
        window_size[0],
        padded_width // window_size[1],
        window_size[1],
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        batch * windows_per_image, window_size[0] * window_size[1], channels
    )

    qkv_values = F.linear(x, qkv.weight, qkv.bias)
    qkv_values = qkv_values.reshape(
        x.size(0), x.size(1), 3, num_heads, channels // num_heads
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv_values.unbind(0)
    query = query * (channels // num_heads) ** -0.5
    attention = query.matmul(key.transpose(-2, -1)) + relative_bias

    if shift_h or shift_w:
        mask = x.new_zeros((padded_height, padded_width))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_h), (-shift_h, None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_w), (-shift_w, None))
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[h_slice[0]:h_slice[1], w_slice[0]:w_slice[1]] = count
                count += 1
        mask = mask.view(
            padded_height // window_size[0],
            window_size[0],
            padded_width // window_size[1],
            window_size[1],
        )
        mask = mask.permute(0, 2, 1, 3).reshape(
            windows_per_image, window_size[0] * window_size[1]
        )
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        attention = attention.view(
            batch, windows_per_image, num_heads, x.size(1), x.size(1)
        ) + mask.unsqueeze(1).unsqueeze(0)
        attention = attention.view(-1, num_heads, x.size(1), x.size(1))

    attention = F.softmax(attention, dim=-1)
    x = attention.matmul(value).transpose(1, 2).reshape(
        x.size(0), x.size(1), channels
    )
    x = F.linear(x, proj.weight, proj.bias)
    x = x.view(
        batch,
        padded_height // window_size[0],
        padded_width // window_size[1],
        window_size[0],
        window_size[1],
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_height, padded_width, channels)
    if shift_h or shift_w:
        x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
    return x[:, :height, :width, :].contiguous()


class _ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        relative_index = _relative_position_index(window_size)
        self.register_buffer("relative_position_index", relative_index)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        token_count = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index]
        bias = bias.view(token_count, token_count, -1).permute(2, 0, 1).contiguous()
        bias = bias.unsqueeze(0)
        return _shifted_window_attention(
            x, self.qkv, self.proj, bias, self.window_size, self.num_heads, self.shift_size
        )


class _SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-5)
        self.norm1 = norm_layer(dim)
        self.attn = _ShiftedWindowAttention(dim, window_size, shift_size, num_heads)
        self.stochastic_depth = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        return x + self.stochastic_depth(self.mlp(self.norm2(x)))


class _PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        height, width = x.shape[-3:-1]
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat((x0, x1, x2, x3), dim=-1)
        return self.reduction(self.norm(x))


class SwinBCompat(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        window_size = (7, 7)
        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers = [
            nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
                _Permute((0, 2, 3, 1)),
                norm_layer(embed_dim),
            )
        ]
        for stage_index, depth in enumerate(depths):
            dim = embed_dim * (2 ** stage_index)
            layers.append(
                nn.Sequential(
                    *[
                        _SwinBlock(
                            dim,
                            num_heads[stage_index],
                            window_size,
                            (0, 0) if block_index % 2 == 0 else (3, 3),
                            norm_layer=norm_layer,
                        )
                        for block_index in range(depth)
                    ]
                )
            )
            if stage_index < len(depths) - 1:
                layers.append(_PatchMerging(dim, norm_layer))

        self.features = nn.Sequential(*layers)
        self.norm = norm_layer(embed_dim * 8)
        self.permute = _Permute((0, 3, 1, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(embed_dim * 8, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.head(x)