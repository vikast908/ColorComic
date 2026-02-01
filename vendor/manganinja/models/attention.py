"""Custom transformer blocks for MangaNinja (from src/models/attention.py).

Uses the custom Attention class that supports encoder_hidden_states_v.
"""

from typing import Optional

import torch
from torch import nn

from .attention_processor import Attention


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and feed-forward.

    Stores ``bank`` for reference attention (populated during RefUNet forward pass,
    read during denoising UNet forward pass).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = norm_type == "ada_norm"
        self.use_ada_layer_norm_zero = norm_type == "ada_norm_zero"

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # Cross-attention
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn,
                              final_dropout=final_dropout)

        # Reference attention bank
        self.bank = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[dict] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        cross_attention_kwargs = cross_attention_kwargs or {}

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class TemporalBasicTransformerBlock(nn.Module):
    """Temporal transformer block (used in video/reference architectures).

    Similar to BasicTransformerBlock but used in ReferenceAttentionControl
    for temporal cross-attention with reference frames.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff_in = FeedForward(dim, dim_out=dim, dropout=dropout,
                                 activation_fn=activation_fn)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int = 1,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Pre-norm + FF
        norm_hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(norm_hidden_states) + hidden_states

        # Self-attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = attn_output + hidden_states

        # Cross-attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = attn_output + hidden_states

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states

        return hidden_states


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU or GELU activation."""

    def __init__(self, dim: int, dim_out: Optional[int] = None,
                 mult: int = 4, dropout: float = 0.0,
                 activation_fn: str = "geglu", final_dropout: bool = False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        if activation_fn == "geglu":
            self.net = nn.ModuleList([
                GEGLU(dim, inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out),
            ])
        elif activation_fn == "gelu":
            self.net = nn.ModuleList([
                nn.Linear(dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out),
            ])
        else:
            raise ValueError(f"Unknown activation_fn: {activation_fn}")

        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    """GELU-gated linear unit."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * torch.nn.functional.gelu(gate)
