"""Generate microscopy (PLI) tractography from MRI tractography using a
trained CrossModalStreamlineVAE checkpoint (see train.py), then run the same
sanity diagnostics as generate_and_analyze.py: spatial envelope, streamline
smoothness, and posterior-collapse signals (across-bundle shape std,
diversity ratio).

Inference runs under torch.no_grad() and is chunked over bundles, same as
generate_and_analyze.py -- this is exactly the case where forward-only
chunking helps, since there's no backward graph to keep around between
chunks (unlike during training; see the note at the top of train.py).

Usage:
  python generate.py \
      --checkpoint runs/exp1/final.pt --stats-dir runs/exp1 \
      --mri dti_MRI_streamlines_Sample1.trk \
      --pli microscopy_tractography_zscaled.trk \
      --out generated_microscopy.trk
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import Stats, apply_stats, invert_stats, load_trk, make_bundles, save_trk
from model import build_model_from_checkpoint


def describe(name: str, sl: np.ndarray) -> np.ndarray:
    pts = sl.reshape(-1, 3)
    bbox = pts.max(0) - pts.min(0)
    seg = np.linalg.norm(np.diff(sl, axis=1), axis=-1).sum(axis=1)  # path length
    chord = np.linalg.norm(sl[:, -1] - sl[:, 0], axis=-1)           # end-to-end
    print(f"\n[{name}] n={sl.shape[0]}")
    print(f"  bbox extent (x,y,z): {bbox.round(2)}")
    print(f"  coord range: min {pts.min(0).round(1)}  max {pts.max(0).round(1)}")
    print(f"  length  mean={seg.mean():.2f} std={seg.std():.2f} "
          f"min={seg.min():.2f} max={seg.max():.2f}")
    print(f"  chord   mean={chord.mean():.2f} "
          f"(chord/length={np.mean(chord / np.maximum(seg, 1e-6)):.3f})")
    return seg


def rms_to_mean(sl: np.ndarray) -> np.ndarray:
    """Per-streamline RMS distance to the *population-average* streamline
    shape (mean taken across streamlines, point-by-point). Low values mean
    every streamline looks like the average one -- a posterior-collapse /
    low-diversity signal, same metric as in generate_and_analyze.py."""
    m = sl.mean(axis=0)  # (P, 3): average path across all streamlines
    return np.sqrt(((sl - m) ** 2).sum(-1).mean(axis=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--mri", required=True, help="MRI tractography to translate, e.g. dti_MRI_streamlines_Sample1.trk")
    p.add_argument("--pli", default=None, help="optional real microscopy .trk for comparison")
    p.add_argument("--out", default="generated_microscopy.trk")
    p.add_argument("--P", type=int, default=128, help="must match the P used in train.py")
    p.add_argument("--K", type=int, default=32, help="must match the K used in train.py")
    p.add_argument("--chunk", type=int, default=4, help="bundles per forward pass")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = build_model_from_checkpoint(args.checkpoint, device=args.device)

    stats_dir = Path(args.stats_dir)
    stats_mri = Stats.load(stats_dir / "stats_mri.npz")
    stats_pli = Stats.load(stats_dir / "stats_pli.npz")

    mri_sl, mri_affine = load_trk(args.mri, args.P)
    norm = apply_stats(mri_sl, stats_mri)
    # shuffle=False keeps order so the output lines up with the input
    # (modulo the remainder dropped when N isn't a multiple of K)
    bundles = make_bundles(norm, args.K, shuffle=False)
    n_used = bundles.shape[0] * args.K

    outs = []
    with torch.no_grad():
        for s in range(0, bundles.shape[0], args.chunk):
            x = torch.from_numpy(bundles[s : s + args.chunk]).to(args.device, dtype=torch.float32)
            outs.append(model.cross_forward(x, source="mri").cpu().numpy())
    gen = np.concatenate(outs, axis=0).reshape(n_used, args.P, 3)
    gen = invert_stats(gen, stats_pli)

    save_trk(gen, args.out, affine=mri_affine)
    print(f"wrote {args.out}: {gen.shape[0]} streamlines x {gen.shape[1]} pts")

    describe("MRI input", mri_sl[:n_used])
    describe("generated microscopy", gen)

    print("\n=== diversity of generated streamlines ===")
    gen_spread = rms_to_mean(gen)
    print(f"  generated: per-streamline RMS-to-mean mean={gen_spread.mean():.3f} "
          f"std={gen_spread.std():.3f}")

    if args.pli:
        pli_sl, _ = load_trk(args.pli, args.P)
        describe("real microscopy", pli_sl)
        pli_spread = rms_to_mean(pli_sl)
        print(f"  real PLI : per-streamline RMS-to-mean mean={pli_spread.mean():.3f} "
              f"std={pli_spread.std():.3f}")
        print(f"  -> diversity ratio gen/real = "
              f"{gen_spread.mean() / max(pli_spread.mean(), 1e-9):.3f}")

    gb = gen.reshape(-1, args.K, args.P, 3).mean(axis=1)  # per-bundle mean shape
    across_bundle = gb.std(axis=0)
    print(f"\n  across-bundle std of per-bundle mean shape: "
          f"mean={across_bundle.mean():.4f} max={across_bundle.max():.4f}")
    print("  (near 0 => every bundle decodes to the same shape; latent ignored / posterior collapse)")
    if args.pli:
        print(f"\n  total point std  generated={gen.reshape(-1,3).std(0).round(3)}  "
              f"real PLI={pli_sl.reshape(-1,3).std(0).round(3)}")


if __name__ == "__main__":
    main()
