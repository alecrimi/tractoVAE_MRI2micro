"""I/O and preprocessing utilities for streamline tractography data.

Handles loading raw .trk files (MRI diffusion tractography and microscopy/PLI
tractography), resampling every streamline to a fixed number of points, and
z-score normalization/denormalization so both modalities live on a
comparable numeric scale before being fed to the VAE.

Assumes .trk format for both modalities (TrackVis/MRtrix-style trk, readable
by nibabel). If your microscopy file (e.g. microscopy_tractography_zscaled.trk)
is actually some other format despite the .trk-style name -- a custom text
format, .vtk, .tck -- swap out `load_trk`'s body accordingly; everything
downstream only needs an (N, P, 3) float32 array back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, TrkFile


def _resample_one(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a single (n_i, 3) streamline to `n_points` along arc length."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == n_points:
        return points.astype(np.float32)
    if len(points) < 2:
        return np.repeat(points[:1], n_points, axis=0).astype(np.float32)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total == 0:  # degenerate: all points identical
        return np.repeat(points[:1], n_points, axis=0).astype(np.float32)
    target = np.linspace(0.0, total, n_points)
    out = np.empty((n_points, 3), dtype=np.float32)
    for d in range(3):
        out[:, d] = np.interp(target, arc, points[:, d])
    return out


def resample_streamlines(streamlines: list[np.ndarray], n_points: int) -> np.ndarray:
    """Resample a list of variable-length streamlines to a fixed point count.

    Returns an (N, n_points, 3) float32 array.
    """
    return np.stack([_resample_one(s, n_points) for s in streamlines], axis=0)


def load_trk(path: str | Path, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a .trk file and resample every streamline to `n_points`.

    Returns (streamlines[N, n_points, 3] float32, affine[4, 4]).
    """
    trk_obj = nib.streamlines.load(str(path))
    raw = [np.asarray(s, dtype=np.float32) for s in trk_obj.streamlines]
    affine = getattr(trk_obj, "affine", None)
    if affine is None:
        affine = trk_obj.tractogram.affine_to_rasmm
    return resample_streamlines(raw, n_points), np.asarray(affine, dtype=np.float32)


def save_trk(streamlines: np.ndarray, path: str | Path, affine: np.ndarray | None = None) -> None:
    """Save an (N, P, 3) array of streamlines to a .trk file."""
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    tractogram = Tractogram([s for s in streamlines], affine_to_rasmm=affine)
    TrkFile(tractogram).save(str(path))


@dataclass
class Stats:
    mean: np.ndarray  # (3,)
    std: np.ndarray   # (3,)

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: str | Path) -> "Stats":
        d = np.load(path)
        return Stats(mean=d["mean"], std=d["std"])


def compute_stats(streamlines: np.ndarray) -> Stats:
    pts = streamlines.reshape(-1, 3)
    return Stats(mean=pts.mean(0).astype(np.float32), std=(pts.std(0) + 1e-6).astype(np.float32))


def apply_stats(streamlines: np.ndarray, stats: Stats) -> np.ndarray:
    return ((streamlines - stats.mean) / stats.std).astype(np.float32)


def invert_stats(streamlines: np.ndarray, stats: Stats) -> np.ndarray:
    return (streamlines * stats.std + stats.mean).astype(np.float32)


def make_bundles(streamlines: np.ndarray, K: int, shuffle: bool = True, seed: int = 0) -> np.ndarray:
    """Group N streamlines into bundles of K streamlines each (drops remainder).

    Returns (n_bundles, K, P, 3). When `shuffle` is True each bundle is a
    random subset of the source tractogram rather than a spatially
    contiguous chunk -- use this for training. For generation, pass
    shuffle=False to keep streamlines in their original order so the output
    file lines up with the input one-to-one (modulo the dropped remainder).
    """
    n = streamlines.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_bundles = n // K
    if n_bundles == 0:
        raise ValueError(f"need at least K={K} streamlines, got {n}")
    idx = idx[: n_bundles * K]
    return streamlines[idx].reshape(n_bundles, K, *streamlines.shape[1:])
