"""Isolates whether scribbly/noisy generated output is a decoder problem or
a cross-modal latent-alignment problem.

Feeds real PLI streamlines through the PLI encoder/decoder ONLY (no MRI
involved at all -- this is a same-domain VAE reconstruction, using the
posterior mean like cross_forward does) and runs the same diagnostics as
generate.py against the real data.

How to read the result:
  - self-reconstruction smooth (chord/length close to the real PLI's) but
    generated_microscopy.trk is scribbly (chord/length ~0.1)
      -> decoder is fine; the MRI<->PLI latent spaces aren't aligned well.
         Cycle-consistency is a comparatively weak unpaired-alignment
         signal -- try more epochs, a higher --cycle-weight, or (if you
         have any bundle-level correspondence between MRI and PLI bundles)
         a direct mse(z_mri, z_pli) loss, which is much stronger.
  - self-reconstruction is ALSO scribbly
      -> the decoder itself hasn't learned smooth geometry yet. Train
         longer, and/or add the curvature/smoothness loss described in
         train.py (search for `curvature_loss`).

Usage:
  python check_reconstruction.py \
      --checkpoint runs/exp1/final.pt --stats-dir runs/exp1 \
      --pli microscopy_tractography_zscaled.trk --K 32 --P 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import Stats, apply_stats, invert_stats, load_trk, make_bundles
from generate import describe, rms_to_mean
from model import build_model_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--pli", required=True)
    p.add_argument("--P", type=int, default=128)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = build_model_from_checkpoint(args.checkpoint, device=args.device)
    stats_pli = Stats.load(Path(args.stats_dir) / "stats_pli.npz")

    pli_sl, _ = load_trk(args.pli, args.P)
    norm = apply_stats(pli_sl, stats_pli)
    bundles = make_bundles(norm, args.K, shuffle=False)
    n_used = bundles.shape[0] * args.K

    outs = []
    with torch.no_grad():
        for s in range(0, bundles.shape[0], args.chunk):
            x = torch.from_numpy(bundles[s : s + args.chunk]).to(args.device, dtype=torch.float32)
            mu, _ = model._encoder("pli")(x)   # deterministic, same as cross_forward uses
            recon = model._decoder("pli")(mu)
            outs.append(recon.cpu().numpy())
    recon = np.concatenate(outs, axis=0).reshape(n_used, args.P, 3)
    recon = invert_stats(recon, stats_pli)

    describe("real PLI (subset)", pli_sl[:n_used])
    describe("PLI self-reconstruction", recon)

    real_spread = rms_to_mean(pli_sl[:n_used])
    recon_spread = rms_to_mean(recon)
    print(f"\nself-reconstruction diversity ratio = "
          f"{recon_spread.mean() / max(real_spread.mean(), 1e-9):.3f}")
    print("\nCompare the two chord/length ratios above to the ones you saw from "
          "generate.py to tell decoder-capacity issues apart from cross-modal "
          "latent-alignment issues -- see the module docstring for how to read it.")


if __name__ == "__main__":
    main()