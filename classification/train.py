"""Train a cross-modal VAE that translates between MRI diffusion
tractography streamlines and microscopy (PLI) tractography streamlines.

Tuned for a single GeForce GTX 1080 (8 GB VRAM, Pascal architecture):

  - Pascal has no tensor cores and very poor fp16 throughput on consumer
    cards (only the datacenter P100 is fast at fp16). Training here runs in
    plain fp32 -- do not wrap this in torch.cuda.amp.autocast(), it will not
    speed things up and can even slow them down on this GPU.
  - `--chunk` controls a true gradient-accumulation micro-batch: each chunk
    of bundles is forwarded, its (scaled) loss is backpropagated immediately,
    and only after all chunks in a logical batch are done does the optimizer
    step. This is different from -- and more memory-effective for training
    than -- the forward-only chunking used at inference time in
    generate_and_analyze.py, because per-chunk backward() frees that chunk's
    activation graph before the next chunk starts. Forward-only chunking
    followed by a single backward() at the end (as is fine for inference)
    would NOT reduce backward-pass memory, since autograd still has to hold
    every chunk's activations simultaneously.
  - Defaults (K=32 streamlines/bundle, P=128 points/streamline, d_model=96,
    batch of 8 bundles processed in chunks of 2) are deliberately modest.
    Watch `nvidia-smi` during the first few hundred steps and raise --K, --P,
    --batch-bundles, or model width in model.py only if you have headroom.
  - If you still hit OOM: lower --chunk first (down to 1), then --cycle-weight
    0 (drops the 4 extra cross_forward calls per step), then K/P/d_model.

If generated streamlines come out scribbly/zigzagging (low chord/length
ratio, path length much higher than the real data despite a similar
bounding box) rather than smooth and bundle-like, try --smooth-weight
0.01-0.1: plain MSE reconstruction doesn't penalize a wiggly path as long
as it's roughly in the right place on average, so this adds an explicit
curvature penalty that rewards straighter, smoother streamlines.

Every run writes a per-epoch `loss_log.csv` (epoch, recon, kl, cycle,
smooth, score) into --out, regardless of whether --patience is used --
plot `score` against `epoch` afterward to see whether training had already
plateaued well before --epochs ran out, so you know how many epochs you
actually need next time. Pass --patience N to stop automatically once
`score` hasn't improved by more than --min-delta for N epochs in a row (this
also writes a `best.pt` checkpoint at the best epoch, separate from
`last.pt`/`final.pt`).

Data assumption: MRI and microscopy streamline *counts* don't need to match
and individual streamlines are not assumed to correspond 1:1 across
modalities (this is the common case for tractography from different
imaging modalities / specimens). Training therefore uses unpaired
shared-latent VAE objectives: within-modality reconstruction + KL, plus a
cycle-consistency loss (mri -> pli -> mri and pli -> mri -> pli) to keep the
two decoders honest about what the shared latent space means. If you *do*
have bundle-level correspondence between your MRI and microscopy data, add
a direct `mse(z_mri, z_pli)` alignment term for matched bundle indices --
that's a stronger and more sample-efficient signal than cycle-consistency
alone.

Usage:
  python train.py \
      --mri dti_MRI_streamlines_Sample1.trk \
      --pli microscopy_tractography_zscaled.trk \
      --out runs/exp1
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import apply_stats, compute_stats, load_trk, make_bundles
from model import CrossModalStreamlineVAE, ModelConfig


@dataclass
class TrainConfig:
    K: int = 8                   # was 32 -- smaller bundles are more spatially coherent after spatial sort
    P: int = 32                  # was 128 -- PLI streamlines ~20mm; 128pts=0.16mm/step, noise-dominated
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    latent_dim: int = 48
    batch_bundles: int = 8       # logical batch size, in bundles
    chunk: int = 2                # bundles per gradient-accumulation micro-step
    epochs: int = 200
    lr: float = 2e-4
    kl_warmup_steps: int = 400   # ~10 epochs for a ~40 steps/epoch dataset; was 2000 (=54 epochs)
    kl_weight: float = 0.1
    free_bits: float = 0.05      # min nats/dim kept alive -- guards against posterior collapse
    cycle_weight: float = 1.0
    smooth_weight: float = 0.0   # weight on curvature_loss; try 0.01-0.1 if output looks scribbly
    patience: int = 0            # epochs without improvement before early stopping; 0 disables
    min_delta: float = 1e-4      # minimum improvement in epoch score to reset patience
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class BundleDataset(Dataset):
    def __init__(self, bundles: np.ndarray):
        self.bundles = bundles

    def __len__(self) -> int:
        return len(self.bundles)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.bundles[idx]


def kl_term(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    """KL(q(z|x) || N(0,1)) per dim, clamped (free-bits trick) to avoid
    posterior collapse, then summed over dims and averaged over the batch."""
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    kl = torch.clamp(kl, min=free_bits)
    return kl.sum(-1).mean()


def curvature_loss(points: torch.Tensor) -> torch.Tensor:
    """Penalizes the second difference along each streamline's P points.

    Plain MSE on absolute point positions doesn't discourage a decoded
    streamline from zigzagging back and forth (high path length, low
    chord/length ratio -- the "scribble" failure mode) as long as the
    *positions* are roughly in the right place on average. This term
    explicitly rewards smooth, low-curvature paths, which is what real
    tractography streamlines look like. Works on any (..., P, 3) tensor.
    """
    d2 = points[..., 2:, :] - 2 * points[..., 1:-1, :] + points[..., :-2, :]
    return d2.pow(2).sum(-1).mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mri", required=True, help="MRI tractography .trk, e.g. dti_MRI_streamlines_Sample1.trk")
    p.add_argument("--pli", required=True, help="microscopy tractography .trk, e.g. microscopy_tractography_zscaled.trk")
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--P", type=int, default=32)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=48)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-bundles", type=int, default=8)
    p.add_argument("--chunk", type=int, default=2, help="bundles per gradient-accumulation micro-step")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--cycle-weight", type=float, default=1.0)
    p.add_argument("--smooth-weight", type=float, default=0.0,
                    help="weight on curvature/smoothness loss; try 0.01-0.1 if generated streamlines look scribbly")
    p.add_argument("--patience", type=int, default=0,
                    help="stop early after this many epochs with no improvement in epoch score; 0 disables")
    p.add_argument("--min-delta", type=float, default=1e-4,
                    help="minimum improvement in epoch score to count as progress for --patience")
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    cfg = TrainConfig(
        K=args.K, P=args.P, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, latent_dim=args.latent_dim, epochs=args.epochs,
        batch_bundles=args.batch_bundles, chunk=args.chunk, lr=args.lr,
        kl_weight=args.kl_weight, cycle_weight=args.cycle_weight,
        smooth_weight=args.smooth_weight, patience=args.patience, min_delta=args.min_delta,
    )
    if args.device:
        cfg.device = args.device

    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.mri}")
    mri_sl, _ = load_trk(args.mri, cfg.P)
    print(f"  {mri_sl.shape[0]} streamlines, resampled to {cfg.P} pts")
    print(f"loading {args.pli}")
    pli_sl, _ = load_trk(args.pli, cfg.P)
    print(f"  {pli_sl.shape[0]} streamlines, resampled to {cfg.P} pts")

    stats_mri = compute_stats(mri_sl)
    stats_pli = compute_stats(pli_sl)
    stats_mri.save(out_dir / "stats_mri.npz")
    stats_pli.save(out_dir / "stats_pli.npz")

    mri_bundles = make_bundles(apply_stats(mri_sl, stats_mri), cfg.K, seed=cfg.seed)
    pli_bundles = make_bundles(apply_stats(pli_sl, stats_pli), cfg.K, seed=cfg.seed + 1)
    print(f"  {mri_bundles.shape[0]} MRI bundles, {pli_bundles.shape[0]} PLI bundles (K={cfg.K})")

    mri_loader = DataLoader(BundleDataset(mri_bundles), batch_size=cfg.batch_bundles,
                             shuffle=True, drop_last=True)
    pli_loader = DataLoader(BundleDataset(pli_bundles), batch_size=cfg.batch_bundles,
                             shuffle=True, drop_last=True)

    model_cfg = ModelConfig(P=cfg.P, K=cfg.K, d_model=cfg.d_model, n_heads=cfg.n_heads,
                             n_layers=cfg.n_layers, latent_dim=cfg.latent_dim)
    model = CrossModalStreamlineVAE(model_cfg).to(cfg.device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}  device: {cfg.device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Per-epoch loss log -- this is what answers "did it plateau before
    # --epochs ran out?" after the fact, and what --patience uses to decide
    # whether to stop early. `score` = recon + cycle_weight*cycle +
    # smooth_weight*smooth (KL excluded since its weight `beta` itself
    # changes during warmup, which would make the score non-comparable
    # across early epochs).
    log_path = out_dir / "loss_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "recon", "kl", "cycle", "smooth", "score"])

    best_score = float("inf")
    patience_counter = 0

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        pli_iter = iter(pli_loader)
        epoch_recon = epoch_kl = epoch_cycle = epoch_smooth = 0.0
        epoch_macro_steps = 0
        for mri_batch in mri_loader:
            try:
                pli_batch = next(pli_iter)
            except StopIteration:
                pli_iter = iter(pli_loader)
                pli_batch = next(pli_iter)

            mri_batch = mri_batch.to(cfg.device, dtype=torch.float32)
            pli_batch = pli_batch.to(cfg.device, dtype=torch.float32)
            n = mri_batch.shape[0]
            beta = cfg.kl_weight * min(1.0, step / max(cfg.kl_warmup_steps, 1))

            opt.zero_grad(set_to_none=True)
            log_recon = log_kl = log_cycle = log_smooth = 0.0
            for s in range(0, n, cfg.chunk):
                mc = mri_batch[s : s + cfg.chunk]
                pc = pli_batch[s : s + cfg.chunk]
                frac = mc.shape[0] / n

                recon_mri, mu_m, lv_m = model(mc, "mri")
                recon_pli, mu_p, lv_p = model(pc, "pli")
                recon_loss = F.mse_loss(recon_mri, mc) + F.mse_loss(recon_pli, pc)
                kl_loss = kl_term(mu_m, lv_m, cfg.free_bits) + kl_term(mu_p, lv_p, cfg.free_bits)
                smooth_loss = curvature_loss(recon_mri) + curvature_loss(recon_pli)

                cycle_loss = torch.zeros((), device=cfg.device)
                if cfg.cycle_weight > 0:
                    translated_pli = model.cross_forward(mc, "mri")
                    cycled_mri = model.cross_forward(translated_pli, "pli")
                    translated_mri = model.cross_forward(pc, "pli")
                    cycled_pli = model.cross_forward(translated_mri, "mri")
                    cycle_loss = F.mse_loss(cycled_mri, mc) + F.mse_loss(cycled_pli, pc)
                    if cfg.smooth_weight > 0:
                        # penalize scribbling in the cross-modal outputs too,
                        # not just the within-modality reconstructions above
                        smooth_loss = smooth_loss + curvature_loss(translated_pli) \
                                                  + curvature_loss(translated_mri)

                loss = (recon_loss + beta * kl_loss + cfg.cycle_weight * cycle_loss
                        + cfg.smooth_weight * smooth_loss) * frac
                loss.backward()  # accumulates into .grad; frees this chunk's graph

                log_recon += recon_loss.item() * frac
                log_kl += kl_loss.item() * frac
                log_cycle += cycle_loss.item() * frac
                log_smooth += smooth_loss.item() * frac

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch:4d} step {step:6d}  recon={log_recon:.4f} "
                      f"kl={log_kl:.4f} (beta={beta:.3f}) cycle={log_cycle:.4f} "
                      f"smooth={log_smooth:.4f}")
            step += 1
            epoch_recon += log_recon
            epoch_kl += log_kl
            epoch_cycle += log_cycle
            epoch_smooth += log_smooth
            epoch_macro_steps += 1

        avg_recon = epoch_recon / epoch_macro_steps
        avg_kl = epoch_kl / epoch_macro_steps
        avg_cycle = epoch_cycle / epoch_macro_steps
        avg_smooth = epoch_smooth / epoch_macro_steps
        score = avg_recon + cfg.cycle_weight * avg_cycle + cfg.smooth_weight * avg_smooth
        log_writer.writerow([epoch, avg_recon, avg_kl, avg_cycle, avg_smooth, score])
        log_file.flush()
        print(f"epoch {epoch:4d} done  avg_recon={avg_recon:.4f} avg_kl={avg_kl:.4f} "
              f"avg_cycle={avg_cycle:.4f} avg_smooth={avg_smooth:.4f}  score={score:.4f}")

        torch.save({"model": model.state_dict(), "cfg": asdict(model_cfg)}, out_dir / "last.pt")

        if cfg.patience > 0:
            if score < best_score - cfg.min_delta:
                best_score = score
                patience_counter = 0
                torch.save({"model": model.state_dict(), "cfg": asdict(model_cfg)}, out_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"no improvement in score for {cfg.patience} epochs "
                          f"(best={best_score:.4f}) -- stopping early at epoch {epoch}")
                    break

    log_file.close()
    torch.save({"model": model.state_dict(), "cfg": asdict(model_cfg)}, out_dir / "final.pt")
    print(f"done. checkpoint + stats written to {out_dir}")
    print(f"per-epoch loss history: {log_path} (plot `score` vs `epoch` to see where it plateaus)")


if __name__ == "__main__":
    main()
