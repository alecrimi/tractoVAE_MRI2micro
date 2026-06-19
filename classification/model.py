"""Model definition: a cross-modal VAE that maps between MRI diffusion
tractography and microscopy (PLI) tractography streamline bundles.

Architecture (a scaled-down version of the "256-pt self-attention over
bundles" approach referenced in generate_and_analyze.py, sized to fit
comfortably on an 8 GB GPU):

    points (P,3) --PointEncoder (self-attn over the P points)--> per-streamline embedding (D)
    streamlines (K,D) --BundleEncoder (self-attn over the K streamlines)--> mu, logvar (latent_dim)
    z (latent_dim) --BundleDecoder (broadcast z, self-attn over K learned slots)--> per-streamline embedding (D)
    embedding (D) --PointDecoder (P learned positional queries cross-attend to it)--> points (P,3)

Each modality (MRI, PLI/microscopy) gets its own encoder/decoder pair, but
both encoders write into the *same* latent space. That shared space is what
makes `cross_forward` (encode with one modality's encoder, decode with the
other's decoder) a meaningful translation operator rather than just two
unrelated autoencoders.
"""
from __future__ import annotations

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
        cond = self.z_proj(z).unsqueeze(1)                  # (B, 1, D)
        slots = self.slots.expand(z.shape[0], -1, -1) + cond
        return self.encoder(slots)                          # (B, K, D)


class PointDecoder(nn.Module):
    """Per-streamline embedding (N, D) -> points (N, P, 3), N = B*K flattened."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, cfg.P, cfg.d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.n_heads, dim_feedforward=4 * cfg.d_model,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, 3)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        memory = emb.unsqueeze(1)                            # (N, 1, D)
        q = self.queries.expand(emb.shape[0], -1, -1)
        h = self.decoder(q, memory)
        return self.out(h)


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
