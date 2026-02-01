"""ReferenceAttentionControl for MangaNinja (from src/models/mutual_self_attention_multi_scale.py).

Implements the writer/reader pattern for reference-guided attention:
- Writer mode (RefUNet): stores norm_hidden_states in each block's ``bank``
- Reader mode (Denoising UNet): concatenates banked reference features
  with current features, using point embeddings for spatial guidance
"""

import torch
from torch import nn

from .attention import BasicTransformerBlock


class ReferenceAttentionControl:
    """Manages reference attention between a UNet and the denoising pipeline.

    Parameters
    ----------
    unet : nn.Module
        The UNet (either RefUNet for writing or denoising UNet for reading).
    mode : str
        ``"write"`` to populate banks from reference, ``"read"`` to read from banks.
    fusion_blocks : str
        Which blocks to apply attention: ``"midup"`` for mid + up blocks.
    """

    def __init__(self, unet, mode="write", fusion_blocks="midup"):
        self.unet = unet
        self.mode = mode
        self.fusion_blocks = fusion_blocks

        # Collect all BasicTransformerBlock instances in mid + up blocks
        self.attn_modules = []
        self._original_forwards = {}
        self._collect_attn_modules()

        # Point embeddings per scale
        self.point_bank_ref = {}
        self.point_bank_main = {}

    def _collect_attn_modules(self):
        """Find all BasicTransformerBlocks in mid and up blocks."""
        blocks = []
        if self.fusion_blocks in ("midup", "full"):
            if self.unet.mid_block is not None:
                blocks.append(self.unet.mid_block)
            blocks.extend(self.unet.up_blocks)

        for block in blocks:
            for module in block.modules():
                if isinstance(module, BasicTransformerBlock):
                    self.attn_modules.append(module)

    def register(self):
        """Replace forward methods with hacked versions."""
        mode = self.mode
        for i, module in enumerate(self.attn_modules):
            self._original_forwards[i] = module.forward
            if mode == "write":
                module.forward = self._make_writer_forward(module)
            else:
                module.forward = self._make_reader_forward(module, i)

    def unregister(self):
        """Restore original forward methods."""
        for i, module in enumerate(self.attn_modules):
            if i in self._original_forwards:
                module.forward = self._original_forwards[i]
        self._original_forwards.clear()

    def _make_writer_forward(self, module):
        """Create writer forward that stores hidden states in bank."""
        original = module.forward

        def hacked_forward(hidden_states, **kwargs):
            # Store normalized hidden states for reference
            norm_hidden_states = module.norm1(hidden_states)
            module.bank.append(norm_hidden_states.detach().clone())
            # Run normal forward
            return original(hidden_states, **kwargs)

        return hacked_forward

    def _make_reader_forward(self, module, module_idx):
        """Create reader forward that uses banked reference features."""
        original = module.forward
        controller = self

        def hacked_forward(hidden_states, **kwargs):
            if len(module.bank) == 0:
                return original(hidden_states, **kwargs)

            # Get reference features from bank
            bank_features = module.bank[0] if len(module.bank) > 0 else None
            if bank_features is None:
                return original(hidden_states, **kwargs)

            # Get current norm hidden states
            norm_hidden_states = module.norm1(hidden_states)
            batch_size = norm_hidden_states.shape[0]

            # Check for point embeddings at this spatial scale
            seq_len = norm_hidden_states.shape[1]
            point_ref = controller.point_bank_ref.get(seq_len, None)
            point_main = controller.point_bank_main.get(seq_len, None)

            # Expand bank to match batch size
            if bank_features.shape[0] != batch_size:
                bank_features = bank_features.repeat(batch_size, 1, 1)

            # Concatenate reference + current for self-attention
            if point_ref is not None and point_main is not None:
                # With point embeddings: use as encoder_hidden_states_v
                ref_with_points = bank_features + point_ref.to(bank_features.device)
                main_with_points = norm_hidden_states + point_main.to(norm_hidden_states.device)

                # Concatenate along sequence dimension
                augmented_kv = torch.cat([ref_with_points, norm_hidden_states], dim=1)
                augmented_v = torch.cat([bank_features, norm_hidden_states], dim=1)

                # Self-attention with augmented KV
                attn_output = module.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=augmented_kv,
                    encoder_hidden_states_v=augmented_v,
                )
            else:
                # Without point embeddings: simple concatenation
                augmented = torch.cat([bank_features, norm_hidden_states], dim=1)
                attn_output = module.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=augmented,
                )

            hidden_states = attn_output + hidden_states

            # Cross-attention (if present)
            if module.attn2 is not None:
                norm_hidden_states = module.norm2(hidden_states)
                encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
                attn_output = module.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )
                hidden_states = attn_output + hidden_states

            # Feed-forward
            norm_hidden_states = module.norm3(hidden_states)
            hidden_states = module.ff(norm_hidden_states) + hidden_states

            return hidden_states

        return hacked_forward

    def update(self, writer_controller: "ReferenceAttentionControl",
               point_embeddings_ref=None, point_embeddings_main=None):
        """Copy bank data from writer to this reader, with point embeddings.

        Parameters
        ----------
        writer_controller : ReferenceAttentionControl
            The writer controller whose banks to copy from.
        point_embeddings_ref : list, optional
            Multi-scale point embeddings for reference image.
        point_embeddings_main : list, optional
            Multi-scale point embeddings for target image.
        """
        for reader_mod, writer_mod in zip(self.attn_modules, writer_controller.attn_modules):
            reader_mod.bank = list(writer_mod.bank)

        # Store point embeddings indexed by spatial size (seq_len)
        if point_embeddings_ref is not None:
            self.point_bank_ref = {}
            for emb in point_embeddings_ref:
                seq_len = emb.shape[1]
                self.point_bank_ref[seq_len] = emb

        if point_embeddings_main is not None:
            self.point_bank_main = {}
            for emb in point_embeddings_main:
                seq_len = emb.shape[1]
                self.point_bank_main[seq_len] = emb

    def clear(self):
        """Clear all banks."""
        for module in self.attn_modules:
            module.bank.clear()
        self.point_bank_ref.clear()
        self.point_bank_main.clear()
