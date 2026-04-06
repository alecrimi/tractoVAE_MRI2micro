import nibabel as nib
import numpy as np
from scipy.ndimage import (gaussian_filter,
                            gaussian_gradient_magnitude,
                            uniform_filter)
from skimage.morphology import (binary_erosion, binary_dilation,
                                binary_closing,
                                remove_small_objects, disk)
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.segmentation import flood_fill
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread as tif_imread, imwrite as tif_imwrite


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — only edit this block
# ══════════════════════════════════════════════════════════════════════════════

MODE       = "tif_folder"     # "nifti"      →  single registered_stack.nii.gz
                              # "tif"        →  single TIF slice
                              # "tif_folder" →  all TIFs in a folder

NIFTI_PATH  = "registered_stack.nii.gz"
TIF_PATH    = "In_plane_01.tif"
TIF_FOLDER  = "In_plane"          # input folder containing *.tif files
OUT_FOLDER  = "In_plane_masked"   # output folder for masked TIFs

DOWNSAMPLE  = 1    # 1 = full resolution; 4 or 8 for fast debug runs

# ── Multi-region mask settings ────────────────────────────────────────────────
# Regions whose area >= REGION_AREA_FRAC * total_mask_area are kept.
# Lower this (e.g. 0.02) if thin cortical strips are being dropped.
# Raise it (e.g. 0.08) if noisy edge fragments still sneak in.
REGION_AREA_FRAC = 0.05

# Disk radius for binary_closing that bridges gaps between kept regions.
# Increase if WM/GM splits leave visible holes; set to 0 to disable.
CLOSING_RADIUS   = 5   # pixels at full resolution (halved when downsampled)

# ══════════════════════════════════════════════════════════════════════════════


def load_phi(mode, nifti_path=None, tif_path=None):
    """Load and return (phi_rad, affine, header) for a single file."""
    if mode == "nifti":
        nii       = nib.load(nifti_path)
        phi_raw_f = nii.get_fdata().squeeze().astype(np.float64)
        if phi_raw_f.ndim == 3:
            phi_raw_f = phi_raw_f[..., 0]
        return phi_raw_f, nii.affine, nii.header   # already in radians

    elif mode in ("tif", "tif_folder"):
        phi_raw = tif_imread(tif_path)
        phi_raw = np.squeeze(phi_raw)
        if phi_raw.ndim == 3:
            phi_raw = phi_raw[..., 0]
        phi_raw_f = phi_raw.astype(np.float64)
        return np.deg2rad(phi_raw_f), np.eye(4), None   # degrees → radians

    else:
        raise ValueError(f"Unknown MODE '{mode}'. Use 'nifti', 'tif', or 'tif_folder'.")


def process_phi(phi, downsample=1):
    """
    Given a 2-D phi array (radians), return:
      coherent_mask, phi_masked, coherence, grad_mag, initial_mask, thresh_coherence

    Fixes vs original:
      - Flood-fill exterior only seeds from corners that are truly background (==0),
        preventing the fill leaking through slanted / non-rectangular FOV edges.
      - All connected regions above REGION_AREA_FRAC of total mask area are kept
        (not just the single largest-and-central one), so tissue that splits due
        to low-coherence GM bands is not discarded.
      - binary_closing bridges narrow gaps between surviving regions before the
        final nonzero_mask intersection clips back to the true FOV.
    """
    if downsample > 1:
        phi = phi[::downsample, ::downsample]

    # ── 1. Vector field ───────────────────────────────────────────────────────
    cos2 = np.cos(2 * phi)
    sin2 = np.sin(2 * phi)

    # ── 2. Coherence ──────────────────────────────────────────────────────────
    sigma_local = 3.0 * max(1, downsample)
    cos2_smooth = gaussian_filter(cos2, sigma=sigma_local)
    sin2_smooth = gaussian_filter(sin2, sigma=sigma_local)
    coherence   = np.sqrt(cos2_smooth**2 + sin2_smooth**2)

    # ── 3. Initial tissue mask ────────────────────────────────────────────────
    nonzero_mask     = phi != 0.0
    thresh_coherence = threshold_otsu(coherence[nonzero_mask])
    initial_mask     = nonzero_mask & (coherence > thresh_coherence)

    # ── 4. Gradient magnitude ─────────────────────────────────────────────────
    grad_cos = gaussian_gradient_magnitude(cos2, sigma=1.5)
    grad_sin = gaussian_gradient_magnitude(sin2, sigma=1.5)
    grad_mag  = np.sqrt(grad_cos**2 + grad_sin**2)

    noise_pct     = 95 if downsample > 1 else 87
    noise_thresh  = np.percentile(grad_mag[initial_mask], noise_pct)
    coherent_mask = initial_mask & (grad_mag < noise_thresh)

    # ── 5. Fill GM interior via exterior flood-fill ───────────────────────────
    # Only seed from corners that are genuinely background (value == 0).
    # This prevents the exterior label from leaking through slanted FOV edges
    # into tissue, which was causing half the slice to be lost.
    padded = np.pad(coherent_mask.astype(np.uint8), 1, constant_values=0)
    filled = padded.copy()
    H, W   = padded.shape
    corner_seeds = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    for seed in corner_seeds:
        if filled[seed] == 0:          # only flood from true background corners
            filled = flood_fill(filled, seed, 2)
    outside       = (filled == 2)[1:-1, 1:-1]
    coherent_mask = coherent_mask | ~outside

    # ── 6. Keep all sufficiently large regions (multi-region aware) ───────────
    # The original code kept only the single largest-central region, which
    # discarded the lower half of the tissue when coherence was uneven.
    # Now we keep every region whose area exceeds REGION_AREA_FRAC of the
    # total mask, plus always the single largest as a safety fallback.
    labeled  = label(coherent_mask)
    regions  = regionprops(labeled)
    if not regions:
        raise ValueError("Mask is empty — relax coherence or gradient threshold")

    total_area   = coherent_mask.sum()
    area_thresh  = REGION_AREA_FRAC * total_area
    largest_area = max(r.area for r in regions)

    keep_labels = {
        r.label for r in regions
        if r.area >= area_thresh or r.area == largest_area
    }
    coherent_mask = np.isin(labeled, list(keep_labels))

    # ── 7. Morphological closing to bridge gaps between kept regions ──────────
    # Gaps between WM and GM that coherence thresholding breaks apart are
    # closed here so the final mask is one contiguous tissue region.
    r_close = max(1, CLOSING_RADIUS // max(1, downsample))
    if r_close > 0:
        coherent_mask = binary_closing(coherent_mask, disk(r_close))

    # Never expand beyond the true (non-zero) FOV
    coherent_mask = coherent_mask & nonzero_mask

    # ── 8. Apply mask ─────────────────────────────────────────────────────────
    phi_masked = phi * coherent_mask.astype(float)

    return coherent_mask, phi_masked, coherence, grad_mag, initial_mask, thresh_coherence


def save_qc_figure(phi, coherence, thresh_coherence, grad_mag,
                   initial_mask, coherent_mask, phi_masked, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(phi,           cmap='hsv',  origin='lower', vmin=-np.pi/2, vmax=np.pi/2)
    axes[0, 0].set_title('Raw φ (orientation angle)')
    axes[0, 1].imshow(coherence,     cmap='hot',  origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Coherence (Otsu thresh={thresh_coherence:.2f})')
    axes[0, 2].imshow(grad_mag,      cmap='hot',  origin='lower')
    axes[0, 2].set_title('Gradient magnitude (cos2φ, sin2φ)')
    axes[1, 0].imshow(initial_mask,  cmap='gray', origin='lower')
    axes[1, 0].set_title('Initial mask (coherence > Otsu)')
    axes[1, 1].imshow(coherent_mask, cmap='gray', origin='lower')
    axes[1, 1].set_title('Final tissue mask (WM + GM)')
    axes[1, 2].imshow(phi_masked,    cmap='hsv',  origin='lower', vmin=-np.pi/2, vmax=np.pi/2)
    axes[1, 2].set_title('φ masked — ready for structure tensor')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if MODE == "tif_folder":
    # ── Batch: process every TIF in TIF_FOLDER ────────────────────────────────
    in_dir  = Path(TIF_FOLDER)
    out_dir = Path(OUT_FOLDER)
    out_dir.mkdir(exist_ok=True)
    qc_dir  = out_dir / "qc"
    qc_dir.mkdir(exist_ok=True)

    tif_files = sorted(in_dir.glob("*.tif")) + sorted(in_dir.glob("*.tiff"))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in '{in_dir}'")

    print(f"Found {len(tif_files)} TIF file(s) in '{in_dir}'")

    for tif_path in tif_files:
        print(f"\nProcessing: {tif_path.name} ...", end=" ", flush=True)

        phi, _affine, _header = load_phi("tif_folder", tif_path=tif_path)
        print(f"phi range: {phi.min():.3f} to {phi.max():.3f}", end=" | ", flush=True)

        try:
            coherent_mask, phi_masked, coherence, grad_mag, initial_mask, thresh_coherence = \
                process_phi(phi, downsample=DOWNSAMPLE)
        except ValueError as e:
            print(f"SKIPPED ({e})")
            continue

        # Save masked phi as TIF (float32, degrees — convert back from radians)
        out_tif = out_dir / tif_path.name
        tif_imwrite(out_tif, np.rad2deg(phi_masked).astype(np.float32))

        # Save QC figure per file
        save_qc_figure(
            phi, coherence, thresh_coherence, grad_mag,
            initial_mask, coherent_mask, phi_masked,
            qc_dir / (tif_path.stem + "_qc.png")
        )

        print(f"mask {coherent_mask.mean()*100:.1f}% → saved {out_tif.name}")

    print(f"\nDone. Masked TIFs saved to '{out_dir}/'")
    print(f"QC figures saved to '{qc_dir}/'")


else:
    # ── Single file: nifti or tif ─────────────────────────────────────────────
    phi, _affine, _header = load_phi(MODE, nifti_path=NIFTI_PATH, tif_path=TIF_PATH)
    print(f"[{MODE}] phi range: {phi.min():.3f} to {phi.max():.3f}  "
          f"(expected ~-1.57 to +1.57 for radians)")

    if DOWNSAMPLE > 1:
        print(f"Downsampled by {DOWNSAMPLE}×")

    coherent_mask, phi_masked, coherence, grad_mag, initial_mask, thresh_coherence = \
        process_phi(phi, downsample=DOWNSAMPLE)

    out_dir = Path("preprocessed")
    out_dir.mkdir(exist_ok=True)

    def _save_nifti(data, fname, dtype=np.float32):
        img = nib.Nifti1Image(data.astype(dtype), _affine, _header)
        nib.save(img, out_dir / fname)

    _save_nifti(phi_masked,                     "phi_masked.nii.gz")
    _save_nifti(coherence,                      "coherence_map.nii.gz")
    _save_nifti(coherent_mask.astype(np.uint8), "wm_mask.nii.gz")
    _save_nifti(grad_mag,                       "gradient_magnitude.nii.gz")

    save_qc_figure(phi, coherence, thresh_coherence, grad_mag,
                   initial_mask, coherent_mask, phi_masked,
                   out_dir / "preprocessing_qc.png")
    plt.show()

    print(f"Tissue mask covers {coherent_mask.sum()} pixels "
          f"({100*coherent_mask.mean():.1f}% of image)")
    print(f"Outputs saved to '{out_dir}/'")
