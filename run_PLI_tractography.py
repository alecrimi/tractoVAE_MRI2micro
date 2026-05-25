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
image_path = "registered_stack.nii.gz"
output_trk = "microscopy_tractography.trk"
peak_path = "peak_dirs.bin"
SEED_STRIDE = 20
MAX_ANGLE = 30.0
MIN_STEPS = 20
MAX_STEPS = 2000
BATCH_SIZE = 10000
TARGET_SL = 50000

# ==========================================================
# LOAD
# ==========================================================
print("Loading header...")
nifti_img = nib.load(image_path)
affine = nifti_img.affine.astype(np.float32)
inv_affine = np.linalg.inv(affine).astype(np.float32)
zooms = nifti_img.header.get_zooms()[:3]
shape = tuple(nifti_img.header.get_data_shape()[:3])
data_proxy = nifti_img.dataobj
shape_max = np.array(shape, dtype=np.int32) - 1

print(f"Shape : {shape}, Zooms : {zooms} mm")

# ==========================================================
# TISSUE MASK
# ==========================================================
print("\nBuilding tissue mask...")
tissue_mask = np.zeros(shape, dtype=np.uint8)
for z in range(shape[2]):
    sl = np.array(data_proxy[:, :, z], dtype=np.float32)
    tissue_mask[:, :, z] = (sl > 0.001).astype(np.uint8)
print(f"Tissue : {tissue_mask.sum():,} ({100*tissue_mask.sum()/tissue_mask.size:.1f}%)")

# ==========================================================
# ORIENTATION FIELD
# ==========================================================
CHUNK_SIZE = 50
n_chunks = (shape[0] + CHUNK_SIZE - 1) // CHUNK_SIZE

if os.path.exists(peak_path):
    print(f"\nFound {peak_path} — skipping")
    peak_dirs = np.memmap(peak_path, dtype=np.float32,
                          mode='r', shape=shape+(3,))
else:
    print(f"\nComputing orientation in {n_chunks} chunks...")
    peak_dirs = np.memmap(peak_path, dtype=np.float32,
                          mode='w+', shape=shape+(3,))
    for x0 in range(0, shape[0], CHUNK_SIZE):
        x1 = min(x0 + CHUNK_SIZE, shape[0])
        print(f"  X {x0}:{x1}", flush=True)
        phi_norm = np.array(data_proxy[x0:x1], dtype=np.float32)
        if phi_norm.ndim == 4:
            phi_norm = phi_norm[..., 0]
        phi = phi_norm * np.pi
        orient = np.stack([np.cos(phi), np.sin(phi),
                           np.zeros_like(phi)], axis=-1).astype(np.float32)
        orient[~tissue_mask[x0:x1].astype(bool)] = 0.0
        peak_dirs[x0:x1] = orient
        del phi_norm, phi, orient
        gc.collect()
    peak_dirs.flush()
    del peak_dirs
    gc.collect()
    peak_dirs = np.memmap(peak_path, dtype=np.float32,
                          mode='r', shape=shape+(3,))
    print("Done.")

# ==========================================================
# SEEDS
# ==========================================================
print(f"\nGenerating seeds (random stratified, target {TARGET_SL})...")

total_voxels = shape[0] * shape[1] * shape[2]
tissue_frac = 0.588
target_seeds = TARGET_SL * 3
stride = max(1, int((total_voxels * tissue_frac / target_seeds) ** (1/3)))
stride = max(stride, 4)

print(f"Using stride={stride}...")

xs = np.arange(0, shape[0], stride)
ys = np.arange(0, shape[1], stride)
zs = np.arange(0, shape[2], 1)

seed_list = []
for x in xs:
    yy, zz = np.meshgrid(ys, zs, indexing='ij')
    coords = np.stack([
        np.full(yy.size, x, dtype=np.int32),
        yy.ravel().astype(np.int32),
        zz.ravel().astype(np.int32)], axis=1)
    in_tissue = tissue_mask[coords[:,0],
                            coords[:,1],
                            coords[:,2]].astype(bool)
    seed_list.append(coords[in_tissue])
    del coords, yy, zz, in_tissue

seed_vox = np.concatenate(seed_list, axis=0).astype(np.float32)
del seed_list
gc.collect()

rng = np.random.default_rng(42)
if len(seed_vox) > target_seeds:
    idx = rng.choice(len(seed_vox), size=target_seeds, replace=False)
    seed_vox = seed_vox[idx]
    del idx

seeds_mm = (affine[:3,:3] @ seed_vox.T).T + affine[:3, 3]
print(f"Seeds : {len(seeds_mm):,}")
del seed_vox
gc.collect()

# ==========================================================
# VECTORIZED TRACKER
# ==========================================================
cos_max = np.cos(np.deg2rad(MAX_ANGLE))
step_size = float(min(zooms[:2])) * 0.5

def mm_to_vox_batch(pts):
    vox = (inv_affine[:3,:3] @ pts.T).T + inv_affine[:3, 3]
    return np.clip(np.round(vox).astype(np.int32), 0, shape_max)

def lookup_batch(pts):
    idx = mm_to_vox_batch(pts)
    return peak_dirs[idx[:,0], idx[:,1], idx[:,2]].copy()

def tissue_batch(pts):
    idx = mm_to_vox_batch(pts)
    return tissue_mask[idx[:,0], idx[:,1], idx[:,2]].astype(bool)

def track_one_dir(batch, sign=1.0):
    n = len(batch)
    active = np.ones(n, dtype=bool)
    positions = batch.copy()
    dirs = lookup_batch(positions)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    valid = norms[:,0] > 1e-6
    dirs[valid] /= norms[valid]
    dirs[~valid] = 0.0
    active &= valid
    dirs *= sign

    traj = np.zeros((n, MAX_STEPS, 3), dtype=np.float32)
    traj[:, 0] = positions

    for step in range(1, MAX_STEPS):
        if not active.any():
            traj[active==False, step:] = traj[active==False, step-1:step]
            break

        next_pos = positions.copy()
        next_pos[active] += step_size * dirs[active]

        new_dirs = lookup_batch(next_pos)
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        valid = norms[:,0] > 1e-6
        new_dirs[valid] /= norms[valid]
        new_dirs[~valid] = 0.0

        dot = np.einsum('ij,ij->i', new_dirs, dirs)
        flip = (dot < 0) & active
        new_dirs[flip] = -new_dirs[flip]
        dot[flip] = -dot[flip]

        stop = ((dot < cos_max) | ~tissue_batch(next_pos) | ~valid) & active
        active &= ~stop

        traj[~active, step] = traj[~active, step-1]
        positions[active] = next_pos[active]
        dirs[active] = new_dirs[active]
        traj[active, step] = positions[active]

    return traj

def trim(pts):
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    stopped = np.where(diffs < 1e-7)[0]
    return stopped[0] + 1 if len(stopped) else len(pts)

def collect_streamlines_targeted(seeds_all, target):
    streamlines = []
    n_total = len(seeds_all)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        if len(streamlines) >= target:
            print(f"  Target {target} reached — stopping early")
            break

        b0 = b * BATCH_SIZE
        b1 = min(b0 + BATCH_SIZE, n_total)
        batch = seeds_all[b0:b1].astype(np.float32)

        pct = 100 * b1 / n_total
        print(f"  batch {b+1}/{n_batches} ({pct:.1f}%) "
              f"streamlines so far: {len(streamlines):,}", flush=True)

        fwd = track_one_dir(batch, 1.0)
        bwd = track_one_dir(batch, -1.0)

        for i in range(len(batch)):
            flen = trim(fwd[i])
            blen = trim(bwd[i])

            if flen + blen < MIN_STEPS:
                continue

            sl = np.concatenate([
                bwd[i, 1:blen][::-1],
                fwd[i, :flen]
            ], axis=0).astype(np.float32)

            streamlines.append(sl)

            if len(streamlines) >= target:
                break

        del fwd, bwd
        gc.collect()

    return streamlines

# ==========================================================
# RUN
# ==========================================================
print(f"\nTracking (target: {TARGET_SL} streamlines)...")
print(f"  step_size = {step_size:.5f} mm")
print(f"  max_angle = {MAX_ANGLE}°")
print(f"  min_steps = {MIN_STEPS}")

streamlines = collect_streamlines_targeted(seeds_mm, TARGET_SL)
print(f"\nFinal streamline count: {len(streamlines):,}")

# ==========================================================
# SAVE
# ==========================================================
if streamlines:
    print("Saving...")
    sl_obj = Streamlines(streamlines)
    sft = StatefulTractogram(sl_obj, nifti_img, Space.RASMM)

    # Fallback — manual bbox filter
    sft.to_vox()
    valid = []
    for s in sft.streamlines:
        if (s.min() >= 0 and s[:,0].max() < shape[0] and
                s[:,1].max() < shape[1] and s[:,2].max() < shape[2]):
            valid.append(s)
    print(f"Valid after filter: {len(valid):,} / {len(streamlines):,}")
    sft_clean = StatefulTractogram(Streamlines(valid), nifti_img, Space.VOX)

    save_trk(sft_clean, output_trk, bbox_valid_check=False)
    print(f"Saved → {output_trk}")

    lengths = np.array([len(s) for s in sft_clean.streamlines])
    print(f"Count  : {len(sft_clean):,}")
    print(f"Length : min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.1f} steps "
          f"({lengths.mean()*step_size:.2f} mm mean)")
    print(f"Size   : {os.path.getsize(output_trk)/1e9:.2f} GB")