import nibabel as nib
import numpy as np
from scipy.ndimage import (gaussian_filter,
                            gaussian_gradient_magnitude,
                            uniform_filter)
from skimage.morphology import (binary_erosion, binary_dilation,
                                remove_small_objects, disk)
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

from scipy.ndimage import binary_fill_holes
from skimage.segmentation import flood_fill

import matplotlib.pyplot as plt
from pathlib import Path

# ── NEW: tifffile for loading a single TIF ────────────────────────────────────
from tifffile import imread as tif_imread


# ── 0. Load ───────────────────────────────────────────────────────────────────

# ── ORIGINAL (NIfTI stack) — commented out ────────────────────────────────────
nii   = nib.load("registered_stack.nii.gz")
phi   = nii.get_fdata().squeeze()   # shape (X, Y) or (X, Y, 1)
# # phi is in radians — values typically in (-π/2, π/2] for PLI in-plane angle

'''
# ── NEW: Load a single TIF file ───────────────────────────────────────────────
TIF_PATH = "In_plane_01.tif"         # ← updated file path

phi_raw = tif_imread(TIF_PATH)       # shape may be (Y, X), (Y, X, C), etc.
phi_raw = np.squeeze(phi_raw)        # drop singleton dims (e.g. single channel)

# If the TIF has multiple channels, take the first (or average as appropriate)
if phi_raw.ndim == 3:
    phi_raw = phi_raw[..., 0]        # take channel 0; adjust if needed
phi_raw_f = phi_raw.astype(np.float64)

# ── DIAGNOSTIC: check angle value range before any conversion ─────────────────

print(f"phi range: {phi_raw_f.min():.3f} to {phi_raw_f.max():.3f}")
print(f"phi mean:  {phi_raw_f.mean():.3f},  std: {phi_raw_f.std():.3f}")
# Expected if radians:  roughly -1.57 to +1.57  (i.e. -π/2 to +π/2)
# Expected if degrees:  roughly -90  to +90   or  0 to 180

# ── ORIGINAL (assumed radians, no conversion) — commented out ─────────────────
# phi = phi_raw.astype(np.float64)

# ── NEW: convert degrees → radians (no-op if already in radians) ─────────────
phi = np.deg2rad(phi_raw_f)          # -90..+90 deg → -π/2..+π/2 rad
'''
# ── QUICK TEST: downsample by factor N to speed up debugging ─────────────────
DOWNSAMPLE = 1   # set to 1 to disable; try 4 or 8 for a fast sanity check
if DOWNSAMPLE > 1:
    phi = phi[::DOWNSAMPLE, ::DOWNSAMPLE]

# Dummy affine / header for NIfTI saves later (identity 4×4)
_affine = nii.affine # np.eye(4)
_header = nii.header #None   # nib.Nifti1Image accepts None — it will create a default header


# ── 1. Convert angle to a VECTOR field (avoids phase-wrap artifacts) ──────────
# Working directly on phi causes artefactual high gradients wherever phi
# wraps from +π/2 to -π/2.  Convert to unit vectors first.
# For in-plane PLI the angle has π-periodicity (headless vectors):
#   v = (cos(phi), sin(phi))   but  phi and phi+π are the same fibre direction
# Use the double-angle trick to make the field 2π-periodic and continuous:
cos2 = np.cos(2 * phi)    # ranges in [-1, 1], 2π-periodic, no wrap artefact
sin2 = np.sin(2 * phi)


# ── 2. Local orientation coherence (structure tensor coherence index) ─────────
# For each pixel compute the local coherence C in a small window:
#   C = sqrt( <cos2φ>² + <sin2φ>² ) / 1   ∈ [0, 1]
#   C ≈ 1  → all pixels in window point the same way  → coherent WM
#   C ≈ 0  → random orientations in window             → noise / GM
# ── ORIGINAL fixed sigma — commented out ────────────────────────────────────
# sigma_local = 3.0   # px — tune to your pixel size (Mollink ~4 µm/px → sigma≈3–5)
# ── NEW: scale sigma to physical size regardless of downsample factor ────────
sigma_local = 3.0 * DOWNSAMPLE   # keeps smoothing window ~same physical size

cos2_smooth = gaussian_filter(cos2, sigma=sigma_local)
sin2_smooth = gaussian_filter(sin2, sigma=sigma_local)

coherence = np.sqrt(cos2_smooth**2 + sin2_smooth**2)
# coherence ∈ [0, 1];  high = good WM signal


# ── 3. Background mask from raw signal magnitude ──────────────────────────────
# Background voxels have phi≈0 (or are exactly 0 if the image was zero-padded).
# First exclude hard-zero pixels (image border padding) from Otsu calculation,
# then threshold coherence only within non-zero tissue area.

# ── ORIGINAL (no zero-exclusion) — commented out ─────────────────────────────
# thresh_coherence = threshold_otsu(coherence)
# initial_mask = coherence > thresh_coherence

# ── NEW: exclude exact-zero pixels (padding) before Otsu ─────────────────────
nonzero_mask  = phi != 0.0                          # True = real tissue pixels
thresh_coherence = threshold_otsu(coherence[nonzero_mask])
initial_mask  = nonzero_mask & (coherence > thresh_coherence)


# ── 4. Gradient magnitude mask (removes tears / folds / GM border mess) ───────
# Even inside the tissue mask, pixels at tears have very high local gradient.
# Compute gradient on the VECTOR components (not raw phi) to avoid wrap issues.
grad_cos = gaussian_gradient_magnitude(cos2, sigma=1.5)
grad_sin = gaussian_gradient_magnitude(sin2, sigma=1.5)
grad_mag  = np.sqrt(grad_cos**2 + grad_sin**2)

# Exclude top N% of gradient magnitude pixels (tissue tears, GM speckle)
# Start with 85th percentile; lower to 80 if too much noise remains
# ── ORIGINAL fixed percentile — commented out ─────────────────────────────────────
# noise_thresh  = np.percentile(grad_mag[initial_mask], 87)
# ── NEW: looser threshold at low res (borders have high gradient) ──────────────
noise_thresh  = np.percentile(grad_mag[initial_mask], 95 if DOWNSAMPLE > 1 else 87)
coherent_mask = initial_mask & (grad_mag < noise_thresh)

# This is to avoid that we remove too much from the inside
# Pad by 1 so the flood-fill seed is guaranteed outside the tissue
padded = np.pad(coherent_mask, 1, constant_values=0)
outside = flood_fill(padded.astype(np.uint8), (0, 0), 1) == 1
# Remove padding
outside = outside[1:-1, 1:-1]
# Everything that is NOT outside and NOT already in mask = enclosed holes = GM
coherent_mask = coherent_mask | ~outside


# ── 5. Morphological cleanup ──────────────────────────────────────────────────
# (a) Erode: disconnect thin bridges between CC and peripheral blobs
#     Scale disk radius to your resolution:
#     if pixel size ≈ 4 µm  → disk(4);  if ≈ 64 µm (downsampled) → disk(2)
# ── ORIGINAL fixed radius — commented out ────────────────────────────────────
# r_erode = 4   # tune to your pixel size
# ── NEW: scale erosion radius to downsample factor ───────────────────────────
#r_erode = max(1, 4 // DOWNSAMPLE)    # e.g. DOWNSAMPLE=8 → r_erode=1
#coherent_mask = binary_erosion(coherent_mask, disk(r_erode))

# (b) Remove small isolated blobs
# ── ORIGINAL fixed blob size — commented out ─────────────────────────────────
# min_blob_px = 2000   # tune: at 4 µm/px this ≈ 0.032 mm²
# ── NEW: scale min blob size to downsample factor (area scales as factor²) ───
#min_blob_px = max(50, 2000 // DOWNSAMPLE**2)
#coherent_mask = remove_small_objects(coherent_mask.astype(bool),     min_size=min_blob_px)

# (c) Keep ONLY the largest connected component → the CC body
labeled  = label(coherent_mask)
regions  = regionprops(labeled)
if not regions:
    raise ValueError("Mask is empty — relax coherence threshold or gradient threshold")
largest      = max(regions, key=lambda r: r.area)
coherent_mask = (labeled == largest.label)

# (d) Gentle re-dilation to recover pixels lost at the CC border during erosion
# ── ORIGINAL fixed radius — commented out ───────────────────────────────────
# r_dilate = 2   # always less than r_erode
# ── NEW: scale dilation radius to downsample factor ─────────────────────────
# ── NEW: stronger dilation at test res to recover eroded borders ─────────────
# ── ORIGINAL — commented out: r_dilate = max(1, 2 // DOWNSAMPLE) ────────────
r_dilate = 3 if DOWNSAMPLE > 1 else 2   # larger at test res to recover borders
coherent_mask = binary_dilation(coherent_mask, disk(r_dilate))


# ── 6. Apply mask ─────────────────────────────────────────────────────────────
phi_masked = phi * coherent_mask.astype(float)
# Zeros outside mask — downstream code should ignore zero-mask pixels


# ── 7. Save outputs ───────────────────────────────────────────────────────────
out_dir = Path("preprocessed")
out_dir.mkdir(exist_ok=True)

# ── ORIGINAL _save (NIfTI, uses nii.affine / nii.header) — commented out ──────
# def _save(data, fname, dtype=np.float32):
#     img = nib.Nifti1Image(data.astype(dtype), nii.affine, nii.header)
#     nib.save(img, out_dir / fname)

# ── NEW _save: works without an original NIfTI header ─────────────────────────
def _save(data, fname, dtype=np.float32):
    img = nib.Nifti1Image(data.astype(dtype), _affine, _header)
    nib.save(img, out_dir / fname)

_save(phi_masked,              "phi_masked.nii.gz")
_save(coherence,               "coherence_map.nii.gz")
_save(coherent_mask.astype(np.uint8), "wm_mask.nii.gz")
_save(grad_mag,                "gradient_magnitude.nii.gz")  # QC


# ── 8. QC figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(phi,            cmap='hsv',  origin='lower', vmin=-np.pi/2, vmax=np.pi/2)
axes[0,0].set_title('Raw φ (orientation angle)')

axes[0,1].imshow(coherence,      cmap='hot',  origin='lower', vmin=0, vmax=1)
axes[0,1].set_title(f'Coherence (Otsu thresh={thresh_coherence:.2f})')

axes[0,2].imshow(grad_mag,       cmap='hot',  origin='lower')
axes[0,2].set_title('Gradient magnitude (cos2φ, sin2φ)')

axes[1,0].imshow(initial_mask,   cmap='gray', origin='lower')
axes[1,0].set_title('Initial mask (coherence > Otsu)')

axes[1,1].imshow(coherent_mask,  cmap='gray', origin='lower')
axes[1,1].set_title('Final WM mask (after morphology)')

axes[1,2].imshow(phi_masked,     cmap='hsv',  origin='lower', vmin=-np.pi/2, vmax=np.pi/2)
axes[1,2].set_title('φ masked — ready for structure tensor')

for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig(out_dir / "preprocessing_qc.png", dpi=150)
plt.show()

print(f"WM mask covers {coherent_mask.sum()} pixels "
      f"({100*coherent_mask.mean():.1f}% of image)")
