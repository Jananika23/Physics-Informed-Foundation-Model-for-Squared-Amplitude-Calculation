"""
Physics-informed Transformer model for squared amplitude prediction.

We treat the equation tokens as a source/target language pair:
  src: input_ids (amplitude)
  tgt: target_ids (squared amplitude)

Physics-informed improvement (simple + effective):
1) Index tokens look like "_1", "_2", ... after Task 1.2 normalization.
   - These index tokens have a special embedding contribution.
   - Why: indices repeat and must stay consistent within an equation;
     giving them special capacity helps the model learn structured reuse.

2) Operators vs variables:
   - We classify tokens into coarse types (operator/function/number/index/variable/other).
   - We add a learned "token type embedding" so the model can distinguish
     syntax symbols (operators/functions) from identifiers/parameters.
   - Why: symbolic regression / expression generation relies on correct syntax.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from utils import invert_vocab, load_vocab


INDEX_RE = re.compile(r"^_\d+$")
NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
FUNC_SET = {"sin", "cos", "tan", "log", "exp", "sqrt", "arcsin", "arccos", "arctan"}
OP_SET = {"+", "-", "*", "/", "^", "=", "**"}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learned positions)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PhysicsSeq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer with physics-aware embeddings."""

    def __init__(
        self,
        vocab_path: Path,
        pad_id: int,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        type_vocab_size: int = 7,
    ):
        super().__init__()

        vocab = load_vocab(vocab_path)
        id_to_token = invert_vocab(vocab)

        # Token type mapping (coarse categories).
        # We keep this intentionally small so it is easy to extend.
        self.type_to_id = {
            "op": 0,
            "func": 1,
            "num": 2,
            "index": 3,
            "var": 4,
            "other": 5,
            "pad": 6,
        }
        self.type_vocab_size = type_vocab_size

        # Precompute per-token type IDs and index mask for fast lookup.
        token_type_ids = torch.full((vocab_size,), self.type_to_id["other"], dtype=torch.long)
        is_index_token = torch.zeros((vocab_size,), dtype=torch.bool)

        for token_id in range(vocab_size):
            if token_id == pad_id:
                token_type_ids[token_id] = self.type_to_id["pad"]
                continue
            tok = id_to_token.get(token_id, "")
            if INDEX_RE.match(tok):
                token_type_ids[token_id] = self.type_to_id["index"]
                is_index_token[token_id] = True
            elif tok in OP_SET:
                token_type_ids[token_id] = self.type_to_id["op"]
            elif tok in FUNC_SET:
                token_type_ids[token_id] = self.type_to_id["func"]
            elif NUMBER_RE.match(tok):
                token_type_ids[token_id] = self.type_to_id["num"]
            elif tok:
                # Default: treat identifiers / symbols as "variables".
                token_type_ids[token_id] = self.type_to_id["var"]

        self.register_buffer("token_type_ids", token_type_ids)  # (vocab_size,)
        self.register_buffer("is_index_token", is_index_token)  # (vocab_size,)

        self.pad_id = pad_id
        self.vocab_size = vocab_size

        # "Foundation model" style: scalable Transformer backbone.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.type_embedding = nn.Embedding(self.type_vocab_size, d_model)

        # Special embedding contribution for index tokens.
        # Implementation detail: we keep a separate embedding table, and
        # add it only on positions that correspond to index tokens.
        self.index_special_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we use (seq_len, batch, dim)
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _generate_square_subsequent_mask(tgt_len: int, device: torch.device) -> torch.Tensor:
        # Mask future positions in the decoder.
        # Shape: (tgt_len, tgt_len) with -inf where attention is blocked.
        mask = torch.full((tgt_len, tgt_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        returns: (batch, seq_len, d_model)
        """
        token_emb = self.token_embedding(token_ids)
        type_ids = self.token_type_ids[token_ids]  # (batch, seq_len)
        type_emb = self.type_embedding(type_ids)

        # Add special embedding only for index tokens.
        index_mask = self.is_index_token[token_ids].unsqueeze(-1).to(token_emb.dtype)
        index_emb = self.index_special_embedding(token_ids)

        return token_emb + type_emb + index_emb * index_mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        src_ids: (batch, src_len)
        tgt_in_ids: (batch, tgt_len)
        returns:
          logits: (batch, tgt_len, vocab_size)
        """
        device = src_ids.device
        src_pad_mask = src_ids.eq(self.pad_id)  # (batch, src_len)
        tgt_pad_mask = tgt_in_ids.eq(self.pad_id)  # (batch, tgt_len)

        src_emb = self.embed(src_ids)  # (batch, src_len, d_model)
        tgt_emb = self.embed(tgt_in_ids)  # (batch, tgt_len, d_model)

        # (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)

        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0), device=device)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )  # (tgt_len, batch, d_model)

        out = out.transpose(0, 1)  # (batch, tgt_len, d_model)
        logits = self.output_layer(out)  # (batch, tgt_len, vocab_size)
        return logits

