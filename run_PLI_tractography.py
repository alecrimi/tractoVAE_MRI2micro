import numpy as np
import nibabel as nib
from dipy.tracking.streamline import Streamlines
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import gc
import os

# ==========================================================
# PARAMETERS
# ==========================================================
image_path   = "registered_stack_fixed.nii.gz"
output_trk   = "microscopy_tractography.trk"
MAX_ANGLE    = 60.0   # PLI at 25µm needs wider angle — rapid direction changes
MIN_STEPS    = 10     # shorter minimum — fine scale data
MAX_STEPS    = 2000
BATCH_SIZE   = 5000
TARGET_SL    = 3000
TARGET_PER_Z = TARGET_SL // 13

# Smoothing sigma in voxels — smooth direction field before tracking
# This is the key fix: raw PLI at 25µm is too noisy for direct tracking
# Smooth over ~200µm = 8 voxels to get bundle-level directions
SMOOTH_SIGMA = 2.0

# ==========================================================
# LOAD
# ==========================================================
print("Loading header...")
nifti_img  = nib.load(image_path)
affine     = nifti_img.affine.astype(np.float32)
zooms      = nifti_img.header.get_zooms()[:3]
shape      = tuple(nifti_img.header.get_data_shape()[:3])
data_proxy = nifti_img.dataobj

print(f"Shape  : {shape}")
print(f"Zooms  : {zooms} mm")
step_size  = float(min(zooms[:2])) * 0.5
cos_max    = np.cos(np.deg2rad(MAX_ANGLE))
print(f"Step   : {step_size:.4f} mm")
print(f"Angle  : {MAX_ANGLE}°")
print(f"Smooth : {SMOOTH_SIGMA} voxels = "
      f"{SMOOTH_SIGMA*zooms[0]*1000:.0f} µm")

# ==========================================================
# PER-SLICE TRACTOGRAPHY
# ==========================================================
from scipy.ndimage import gaussian_filter

all_streamlines = []

for z in range(shape[2]):
    print(f"\n{'='*50}")
    print(f"Z={z} ({z+1}/{shape[2]})", flush=True)

    sl_data = np.array(data_proxy[:, :, z], dtype=np.float32)
    tissue  = sl_data > 0.0
    n_tis   = tissue.sum()
    print(f"  Tissue: {n_tis:,} ({100*n_tis/tissue.size:.1f}%)")
    if n_tis == 0:
        continue

    # ----------------------------------------------------------
    # SMOOTH the angle field before computing directions
    # Raw PLI has sub-voxel crossings — smooth to bundle scale
    # Use circular smoothing: smooth cos(2phi) and sin(2phi)
    # separately, then reconstruct angle
    # ----------------------------------------------------------
    phi_raw = sl_data * np.pi           # (X, Y) in [0, π]

    cos2 = np.cos(2 * phi_raw)
    sin2 = np.sin(2 * phi_raw)

    # Zero background before smoothing to avoid edge bleeding
    cos2[~tissue] = 0.0
    sin2[~tissue] = 0.0

    cos2_s = gaussian_filter(cos2, sigma=SMOOTH_SIGMA)
    sin2_s = gaussian_filter(sin2, sigma=SMOOTH_SIGMA)

    # Reconstruct smoothed angle: phi_smooth = atan2(sin2, cos2) / 2
    phi_smooth = np.arctan2(sin2_s, cos2_s) / 2.0  # in [-π/2, π/2]
    phi_smooth += np.pi / 2 
    
    # Coherence map — use as confidence/stopping mask
    coherence = np.sqrt(cos2_s**2 + sin2_s**2)     # [0, 1]
    coherence[~tissue] = 0.0

    # Direction vectors from smoothed angle
    dirx = np.cos(phi_smooth)
    diry = np.sin(phi_smooth)
    
    #diry = -np.sin(phi_smooth)
    dirx[~tissue] = 0.0
    diry[~tissue] = 0.0

    # Coherence threshold for stopping — stop where directions
    # are locally incoherent (crossings, edges)
    coh_threshold = 0.1
    valid_tissue  = tissue & (coherence > coh_threshold)

    print(f"  Valid after coherence filter: "
          f"{valid_tissue.sum():,} "
          f"({100*valid_tissue.sum()/n_tis:.1f}% of tissue)")

    del cos2, sin2, cos2_s, sin2_s, phi_raw, phi_smooth, coherence

    # ----------------------------------------------------------
    # SEEDS
    # ----------------------------------------------------------
    seed_coords = np.argwhere(valid_tissue)
    n_seeds     = min(TARGET_PER_Z * 3, len(seed_coords))
    rng         = np.random.default_rng(42 + z)
    chosen      = rng.choice(len(seed_coords), size=n_seeds,
                             replace=False)
    seed_xy     = seed_coords[chosen]

    # mm coordinates — Z fixed for this slice
    seed_mm        = np.zeros((n_seeds, 3), dtype=np.float32)
    seed_mm[:, 0]  = seed_xy[:, 0] * float(zooms[0])
    seed_mm[:, 1]  = seed_xy[:, 1] * float(zooms[1])
    seed_mm[:, 2]  = z * float(zooms[2])

    print(f"  Seeds: {n_seeds:,}", flush=True)
    del seed_coords, chosen, seed_xy

    # ----------------------------------------------------------
    # LOOKUP FUNCTIONS for this slice
    # ----------------------------------------------------------
    x_max = shape[0] - 1
    y_max = shape[1] - 1

    def lookup_2d(pts_mm):
        xi = np.clip(np.round(pts_mm[:, 0] / zooms[0]).astype(np.int32),
                     0, x_max)
        yi = np.clip(np.round(pts_mm[:, 1] / zooms[1]).astype(np.int32),
                     0, y_max)
        dx = dirx[xi, yi]
        dy = diry[xi, yi]
        dz = np.zeros(len(pts_mm), dtype=np.float32)
        return np.stack([dx, dy, dz], axis=1)

    def in_tissue_2d(pts_mm):
        xi = np.clip(np.round(pts_mm[:, 0] / zooms[0]).astype(np.int32),
                     0, x_max)
        yi = np.clip(np.round(pts_mm[:, 1] / zooms[1]).astype(np.int32),
                     0, y_max)
        return valid_tissue[xi, yi]

    # ----------------------------------------------------------
    # TRACK ONE DIRECTION
    # ----------------------------------------------------------
    def track_one_dir_2d(batch, sign=1.0):
        n         = len(batch)
        active    = np.ones(n, dtype=bool)
        positions = batch.copy()
        dirs      = lookup_2d(positions)

        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        valid = norms[:, 0] > 1e-6
        dirs[valid]  /= norms[valid]
        dirs[~valid]  = 0.0
        active       &= valid
        dirs         *= sign

        streams = [[] for _ in range(n)]
        for i in range(n):
            if active[i]:
                streams[i].append(positions[i].copy())

        for step in range(MAX_STEPS):
            if not active.any():
                break

            act_idx  = np.where(active)[0]
            next_pos = positions[act_idx] + step_size * dirs[act_idx]

            new_dirs = lookup_2d(next_pos)
            norms    = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
            valid_a  = norms[:, 0] > 1e-6
            new_dirs[valid_a]  /= norms[valid_a]
            new_dirs[~valid_a]  = 0.0

            dot  = np.einsum('ij,ij->i', new_dirs, dirs[act_idx])
            flip = dot < 0
            new_dirs[flip] = -new_dirs[flip]
            dot[flip]      = -dot[flip]

            in_t = in_tissue_2d(next_pos)
            keep = (dot >= cos_max) & in_t & valid_a

            cont_idx = act_idx[keep]
            active[act_idx[~keep]] = False

            positions[cont_idx] = next_pos[keep]
            dirs[cont_idx]      = new_dirs[keep]

            for gi in cont_idx:
                streams[gi].append(positions[gi].copy())

            del next_pos, new_dirs, norms, valid_a, dot, flip, in_t, keep

        return [
            np.array(s, dtype=np.float32)
            if len(s) >= MIN_STEPS // 2 else None
            for s in streams
        ]

    # ----------------------------------------------------------
    # BATCH LOOP FOR THIS SLICE
    # ----------------------------------------------------------
    slice_streamlines = []
    n_batches = (n_seeds + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        if len(slice_streamlines) >= TARGET_PER_Z:
            break

        b0    = b * BATCH_SIZE
        b1    = min(b0 + BATCH_SIZE, n_seeds)
        batch = seed_mm[b0:b1]

        fwd = track_one_dir_2d(batch,  1.0)
        bwd = track_one_dir_2d(batch, -1.0)

        for i in range(len(batch)):
            f  = fwd[i]
            bk = bwd[i]
            flen = len(f)  if f  is not None else 0
            blen = len(bk) if bk is not None else 0
            if flen + blen < MIN_STEPS:
                continue
            parts = []
            if bk is not None and len(bk) > 1:
                parts.append(bk[::-1])
            if f is not None and len(f) > 0:
                parts.append(f)
            if parts:
                slice_streamlines.append(
                    np.concatenate(parts, axis=0).astype(np.float32))
            if len(slice_streamlines) >= TARGET_PER_Z:
                break

        pct = 100 * b1 / n_seeds
        print(f"  batch {b+1}/{n_batches} ({pct:.0f}%)  "
              f"sl: {len(slice_streamlines):,}", flush=True)
        del fwd, bwd
        gc.collect()

    print(f"  → {len(slice_streamlines):,} streamlines for Z={z}")
    all_streamlines.extend(slice_streamlines)

    del sl_data, tissue, valid_tissue, dirx, diry
    del seed_mm, slice_streamlines
    gc.collect()

print(f"\nTotal: {len(all_streamlines):,} streamlines")

# ==========================================================
# SAVE
# ==========================================================
if all_streamlines:
    print("Saving...")
    sl_obj = Streamlines(all_streamlines)
    sft    = StatefulTractogram(sl_obj, nifti_img, Space.RASMM)

    sft.to_vox()
    valid = [s for s in sft.streamlines
             if s.min() >= 0
             and s[:, 0].max() < shape[0]
             and s[:, 1].max() < shape[1]
             and s[:, 2].max() < shape[2]]
    print(f"Valid : {len(valid):,} / {len(all_streamlines):,}")

    sft_clean = StatefulTractogram(Streamlines(valid),
                                   nifti_img, Space.VOX)
    save_trk(sft_clean, output_trk, bbox_valid_check=False)
    print(f"Saved → {output_trk}")

    lengths = np.array([len(s) for s in sft_clean.streamlines])
    print(f"Count  : {len(sft_clean):,}")
    print(f"Length : min={lengths.min()}  max={lengths.max()}  "
          f"mean={lengths.mean():.1f} steps  "
          f"({lengths.mean()*step_size:.2f} mm mean)")
    print(f"Size   : {os.path.getsize(output_trk)/1e9:.2f} GB")
else:
    print("No streamlines — check parameters")
