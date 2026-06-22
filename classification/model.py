"""Model definition: a cross-modal VAE that maps between MRI diffusion
tractography and microscopy (PLI) tractography streamline bundles.

Architecture (a scaled-down version of the "256-pt self-attention over
bundles" approach referenced in generate_and_analyze.py, sized to fit
comfortably on an 8 GB GPU):

    points (P,3) --PointEncoder (self-attn over the P points)--> per-streamline embedding (D)
    streamlines (K,D) --BundleEncoder (self-attn over the K streamlines)--> mu, logvar (latent_dim)
    z (latent_dim) --BundleDecoder (broadcast z, self-attn over K learned slots)--> per-streamline embedding (D)
    embedding (D) --PointDecoder (sinusoidal arc-length queries cross-attend to it)--> points (P,3)

Each modality (MRI, PLI/microscopy) gets its own encoder/decoder pair, but
both encoders write into the *same* latent space. That shared space is what
makes `cross_forward` (encode with one modality's encoder, decode with the
other's decoder) a meaningful translation operator rather than two
unrelated autoencoders.

Change from original: PointDecoder now uses fixed sinusoidal arc-length
embeddings at t = [0, 1/(P-1), ..., 1] as decoder queries instead of
learned positional parameters. The original learned queries had no ordering
signal, so the decoder had to discover point order from scratch -- which it
reliably failed to do, producing zigzag / scribble output and reconstruction
loss stuck above 1.9 on z-scored data (a "predict noise" baseline). The
sinusoidal embeddings give the decoder a monotonic "where along the
streamline am I" signal that makes smooth sequential output learnable from
the first gradient step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    P: int = 128          # points per streamline, after resampling
    K: int = 32           # streamlines per bundle
    d_model: int = 96     # transformer width -- kept modest for an 8 GB GPU
    n_heads: int = 4
    n_layers: int = 2
    latent_dim: int = 48


def _sinusoidal_arc_lengths(P: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Fixed sinusoidal embeddings at t = [0, 1/(P-1), ..., 1].

    Returns (1, P, d_model). The monotonic t encodes "where along the
    streamline am I", giving the decoder an ordering signal that makes
    smooth sequential output far easier to learn than learned queries with
    no positional structure.

    Uses the standard sin/cos frequency encoding from "Attention Is All You
    Need" but with t in [0,1] rather than integer positions, so it
    generalises cleanly to different values of P at inference time.
    """
    t = torch.linspace(0, 1, P, device=device)               # (P,)
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1)
    )                                                          # (half,)
    angles = t[:, None] * freqs[None, :]                      # (P, half)
    pe = torch.cat([angles.sin(), angles.cos()], dim=-1)      # (P, d_model)
    # If d_model is odd, the cat gives d_model-1 columns; pad one zero column.
    if pe.shape[-1] < d_model:
        pe = torch.cat([pe, torch.zeros(P, 1, device=device)], dim=-1)
    return pe.unsqueeze(0)                                     # (1, P, d_model)


def _encoder_stack(d_model: int, n_heads: int, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads, dim_feedforward=4 * d_model,
        batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, n_layers)


class AttentionPool(nn.Module):
    """Pools a (B, N, D) sequence down to (B, D) with one learned query."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class PointEncoder(nn.Module):
    """(N, P, 3) -> (N, D) per-streamline embedding, N = B*K flattened."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(3, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.P, cfg.d_model) * 0.02)
        self.encoder = _encoder_stack(cfg.d_model, cfg.n_heads, cfg.n_layers)
        self.pool = AttentionPool(cfg.d_model, cfg.n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.pool(h)


class BundleEncoder(nn.Module):
    """(B, K, D) -> mu, logvar, each (B, latent_dim)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = _encoder_stack(cfg.d_model, cfg.n_heads, cfg.n_layers)
        self.pool = AttentionPool(cfg.d_model, cfg.n_heads)
        self.to_stats = nn.Linear(cfg.d_model, 2 * cfg.latent_dim)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(emb)
        pooled = self.pool(h)
        mu, logvar = self.to_stats(pooled).chunk(2, dim=-1)
        return mu, logvar


class BundleDecoder(nn.Module):
    """z (B, latent_dim) -> per-streamline embeddings (B, K, D)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.slots = nn.Parameter(torch.randn(1, cfg.K, cfg.d_model) * 0.02)
        self.encoder = _encoder_stack(cfg.d_model, cfg.n_heads, cfg.n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cond = self.z_proj(z).unsqueeze(1)                   # (B, 1, D)
        slots = self.slots.expand(z.shape[0], -1, -1) + cond
        return self.encoder(slots)                           # (B, K, D)


class PointDecoder(nn.Module):
    """Per-streamline embedding (N, D) -> points (N, P, 3), N = B*K flattened.

    Queries are fixed sinusoidal arc-length embeddings, not learned parameters.
    This is the key fix vs. the original: learned queries with no ordering
    structure caused the decoder to produce zigzag / scribble streamlines
    because nothing told it which query should come first along the path.
    Sinusoidal embeddings at t in [0,1] provide that monotonic signal for free.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.P = cfg.P
        self.d_model = cfg.d_model
        # Project the per-streamline embedding into decoder memory space.
        # The original used the embedding directly as a (N,1,D) memory vector;
        # adding an explicit projection gives the decoder a learned interface
        # between the encoder's representation and the positional query space.
        self.mem_proj = nn.Linear(cfg.d_model, cfg.d_model)
        layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.n_heads, dim_feedforward=4 * cfg.d_model,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, 3)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (N, D)  where N = B*K
        memory = self.mem_proj(emb).unsqueeze(1)             # (N, 1, D)
        # Arc-length queries are computed fresh each forward pass so they live
        # on whatever device emb is on, with no stored parameter overhead.
        q = _sinusoidal_arc_lengths(self.P, self.d_model, emb.device)
        q = q.expand(emb.shape[0], -1, -1)                  # (N, P, D)
        h = self.decoder(q, memory)                          # (N, P, D)
        return self.out(h)                                   # (N, P, 3)


class ModalityEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.point_encoder = PointEncoder(cfg)
        self.bundle_encoder = BundleEncoder(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, K, P, 3)
        B, K = x.shape[0], x.shape[1]
        emb = self.point_encoder(x.reshape(B * K, *x.shape[2:])).reshape(B, K, -1)
        return self.bundle_encoder(emb)


class ModalityDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.bundle_decoder = BundleDecoder(cfg)
        self.point_decoder = PointDecoder(cfg)
        self.K, self.P = cfg.K, cfg.P

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        emb = self.bundle_decoder(z)                         # (B, K, D)
        B, K, D = emb.shape
        pts = self.point_decoder(emb.reshape(B * K, D))
        return pts.reshape(B, K, self.P, 3)


class CrossModalStreamlineVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mri_encoder = ModalityEncoder(cfg)
        self.pli_encoder = ModalityEncoder(cfg)
        self.mri_decoder = ModalityDecoder(cfg)
        self.pli_decoder = ModalityDecoder(cfg)

    def _encoder(self, modality: str) -> ModalityEncoder:
        return self.mri_encoder if modality == "mri" else self.pli_encoder

    def _decoder(self, modality: str) -> ModalityDecoder:
        return self.mri_decoder if modality == "mri" else self.pli_decoder

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor, source: str):
        """Within-modality VAE pass: encode and decode in the same modality."""
        mu, logvar = self._encoder(source)(x)
        z = self.reparameterize(mu, logvar)
        recon = self._decoder(source)(z)
        return recon, mu, logvar

    def cross_forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """Translate a bundle from `source` modality into the other modality.

        Uses the posterior mean (no sampling noise) -- this is the operator
        actually used at generation time, e.g. mri -> pli (microscopy).
        """
        target = "pli" if source == "mri" else "mri"
        mu, _ = self._encoder(source)(x)
        return self._decoder(target)(mu)


def build_model_from_checkpoint(path: str | Path, device: str = "cpu") -> CrossModalStreamlineVAE:
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = CrossModalStreamlineVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model