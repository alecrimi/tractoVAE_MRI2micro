"""
PLI-to-MRI Registration Pipeline
==================================
# Minimal — rigid registration only
python pli_mri_registration.py  path/to/phi.nii.gz  path/to/FA.nii.gz

# With deformable SyN (better for large tissue distortion)
python pli_mri_registration.py  phi.nii.gz  FA.nii.gz  --use-syn

# Tune masking for your pixel size (e.g. 64 µm downsampled data)
python pli_mri_registration.py  phi.nii.gz  FA.nii.gz \
    --sigma-local 2.0  --r-erode 2  --min-blob-px 500

# Low RAM machine — reduce chunk size
python pli_mri_registration.py  phi.nii.gz  FA.nii.gz  --chunk-size 5

# Re-run only the full-res warp after fixing registration
python pli_mri_registration.py  phi.nii.gz  FA.nii.gz \
    --skip-preprocessing  --skip-registration


Registers Digital Anatomist PLI in-plane orientation maps (.nii.gz)
to MRI space (FA map or b0 volume) using a multi-resolution strategy:

  1. Preprocess & mask PLI (coherence-based WM mask)
  2. Downsample PLI proxy to MRI resolution
  3. Reslice MRI to PLI section plane
  4. Rigid registration on downsampled proxy (ANTs / dipy fallback)
  5. QC checkerboard overlay
  6. Apply transform back to full-res PLI (chunked, memory-safe)
  7. Save all outputs with consistent naming

Dependencies:
    pip install nibabel nilearn antspyx scipy scikit-image matplotlib numpy
    # antspyx includes ANTs binaries — no separate ANTs install needed
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (binary_dilation, binary_erosion,
                                 disk, remove_small_objects)

# ── Optional imports (graceful fallback messages) ────────────────────────────
try:
    import ants
    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False
    print("[WARNING] antspyx not found. Install with: pip install antspyx")
    print("          Falling back to dipy affine registration.")

try:
    from nilearn.image import resample_img
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    print("[WARNING] nilearn not found. Install with: pip install nilearn")
    print("          Reslicing will use nibabel-only fallback.")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PLI PREPROCESSING — coherence-based WM mask
# ═══════════════════════════════════════════════════════════════════════════════

def compute_coherence(phi: np.ndarray, sigma_local: float = 3.0) -> np.ndarray:
    """
    Local orientation coherence from the in-plane angle map phi (radians).

    Uses the double-angle trick to avoid phase-wrap artefacts at ±π/2:
        C = sqrt( <cos(2φ)>² + <sin(2φ)>² )   ∈ [0, 1]
    C ≈ 1  →  all pixels in the local window share the same direction  →  WM
    C ≈ 0  →  random orientations in the window                         →  GM / noise
    """
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    cos2_s = gaussian_filter(cos2, sigma=sigma_local)
    sin2_s = gaussian_filter(sin2, sigma=sigma_local)
    return np.sqrt(cos2_s ** 2 + sin2_s ** 2)


def build_wm_mask(
    phi: np.ndarray,
    sigma_local: float = 3.0,
    grad_percentile: float = 87.0,
    r_erode: int = 4,
    r_dilate: int = 2,
    min_blob_px: int = 2000,
    single_component: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a white-matter mask from the orientation angle map alone.

    Returns
    -------
    mask       : bool array, same shape as phi
    coherence  : float array ∈ [0,1], useful for QC
    """
    coherence = compute_coherence(phi, sigma_local)

    # --- Step A: coherence threshold (Otsu on coherence image) ---------------
    thresh = threshold_otsu(coherence)
    mask = coherence > thresh

    # --- Step B: gradient magnitude filter (removes tears / GM edge noise) ---
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    grad = np.sqrt(
        gaussian_gradient_magnitude(cos2, sigma=1.5) ** 2 +
        gaussian_gradient_magnitude(sin2, sigma=1.5) ** 2
    )
    noise_thresh = np.percentile(grad[mask], grad_percentile)
    mask = mask & (grad < noise_thresh)

    # --- Step C: morphological cleanup ----------------------------------------
    mask = binary_erosion(mask, disk(r_erode))
    mask = remove_small_objects(mask.astype(bool), min_size=min_blob_px)

    if single_component:
        labeled  = label(mask)
        regions  = regionprops(labeled)
        if not regions:
            raise RuntimeError(
                "WM mask is empty after morphological cleanup.\n"
                "Try reducing grad_percentile, r_erode, or min_blob_px."
            )
        largest = max(regions, key=lambda r: r.area)
        mask = labeled == largest.label

    mask = binary_dilation(mask, disk(r_dilate))
    return mask.astype(bool), coherence


def preprocess_pli(
    pli_path: str | Path,
    out_dir: Path,
    sigma_local: float = 3.0,
    grad_percentile: float = 87.0,
    r_erode: int = 4,
    r_dilate: int = 2,
    min_blob_px: int = 2000,
) -> tuple[Path, Path]:
    """
    Load PLI orientation map, build WM mask slice-by-slice (handles 2D and 3D),
    apply mask and save.

    Returns
    -------
    phi_masked_path : Path to masked orientation map
    mask_path       : Path to binary WM mask
    """
    print(f"\n[1/6] Preprocessing PLI: {pli_path}")
    nii  = nib.load(str(pli_path))
    data = nii.get_fdata().squeeze()

    # Handle 2D (single slice) vs 3D (stack of slices)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    phi_out  = np.zeros_like(data, dtype=np.float32)
    mask_out = np.zeros_like(data, dtype=np.uint8)
    coh_out  = np.zeros_like(data, dtype=np.float32)

    n_slices = data.shape[2]
    for z in range(n_slices):
        phi_slice = data[:, :, z].astype(np.float64)
        if phi_slice.max() == phi_slice.min():
            print(f"  Slice {z}: empty/uniform — skipping")
            continue
        try:
            mask_slice, coh_slice = build_wm_mask(
                phi_slice,
                sigma_local=sigma_local,
                grad_percentile=grad_percentile,
                r_erode=r_erode,
                r_dilate=r_dilate,
                min_blob_px=min_blob_px,
            )
        except RuntimeError as e:
            print(f"  Slice {z}: mask failed ({e}) — using full slice")
            mask_slice = np.ones_like(phi_slice, dtype=bool)
            coh_slice  = np.zeros_like(phi_slice)

        phi_out[:, :, z]  = phi_slice * mask_slice
        mask_out[:, :, z] = mask_slice.astype(np.uint8)
        coh_out[:, :, z]  = coh_slice

        if z % 10 == 0 or n_slices <= 10:
            pct = 100 * mask_slice.sum() / mask_slice.size
            print(f"  Slice {z+1}/{n_slices}  WM coverage: {pct:.1f}%")

    # Restore 2D header shape if input was 2D
    if nii.get_fdata().squeeze().ndim == 2:
        phi_out  = phi_out.squeeze()
        mask_out = mask_out.squeeze()
        coh_out  = coh_out.squeeze()

    def _save(arr, fname, dtype=np.float32):
        img = nib.Nifti1Image(arr.astype(dtype), nii.affine, nii.header)
        p   = out_dir / fname
        nib.save(img, str(p))
        print(f"  Saved: {p}")
        return p

    phi_path  = _save(phi_out,               "phi_masked.nii.gz")
    mask_path = _save(mask_out,              "wm_mask.nii.gz", dtype=np.uint8)
    _save(coh_out,                           "coherence_map.nii.gz")

    # QC figure for first (or only) slice
    _qc_preprocessing(
        phi_slice=data[:, :, 0],
        mask_slice=mask_out[:, :, 0] if mask_out.ndim == 3 else mask_out,
        coh_slice=coh_out[:, :, 0]   if coh_out.ndim  == 3 else coh_out,
        out_path=out_dir / "qc_preprocessing.png",
    )
    return phi_path, mask_path


def _qc_preprocessing(phi_slice, mask_slice, coh_slice, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(phi_slice,            cmap="hsv",  origin="lower",
                   vmin=-np.pi/2, vmax=np.pi/2)
    axes[0].set_title("Raw φ (orientation angle)")
    axes[1].imshow(coh_slice,            cmap="hot",  origin="lower", vmin=0, vmax=1)
    axes[1].set_title("Local coherence")
    axes[2].imshow(phi_slice * mask_slice, cmap="hsv", origin="lower",
                   vmin=-np.pi/2, vmax=np.pi/2)
    axes[2].set_title("φ after WM mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  QC figure: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DOWNSAMPLE PLI TO MRI RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def downsample_pli_to_mri(
    pli_path: Path,
    mri_ref_path: Path,
    out_dir: Path,
) -> Path:
    """
    Resample the masked PLI to the MRI voxel grid.
    Uses nilearn if available, otherwise falls back to nibabel + scipy.
    Result is the small proxy used for registration (~5–20 MB).
    """
    print(f"\n[2/6] Downsampling PLI to MRI resolution")
    pli_nii = nib.load(str(pli_path))
    mri_nii = nib.load(str(mri_ref_path))
    out_path = out_dir / "pli_proxy.nii.gz"

    if HAS_NILEARN:
        proxy = resample_img(
            pli_nii,
            target_affine=mri_nii.affine,
            target_shape=mri_nii.shape[:3],
            interpolation="nearest",
        )
    else:
        # Nibabel-only fallback: nearest-neighbour voxel lookup
        proxy = _resample_nibabel(pli_nii, mri_nii.affine, mri_nii.shape[:3])

    nib.save(proxy, str(out_path))
    orig_mb  = pli_nii.get_fdata().nbytes / 1e6
    proxy_mb = proxy.get_fdata().nbytes / 1e6
    print(f"  PLI {pli_nii.shape} → proxy {proxy.shape}")
    print(f"  Size reduction: {orig_mb:.0f} MB → {proxy_mb:.1f} MB")
    print(f"  Saved: {out_path}")
    return out_path


def _resample_nibabel(src_nii, tgt_affine, tgt_shape):
    """Nearest-neighbour resampling without nilearn."""
    from scipy.ndimage import map_coordinates
    src_data = src_nii.get_fdata().squeeze()
    src_inv  = np.linalg.inv(src_nii.affine)

    # Build world coordinates of all target voxels
    i, j, k = np.meshgrid(
        np.arange(tgt_shape[0]),
        np.arange(tgt_shape[1]),
        np.arange(tgt_shape[2]),
        indexing="ij",
    )
    vox_tgt = np.stack([i.ravel(), j.ravel(), k.ravel(),
                         np.ones(i.size)], axis=0)   # (4, N)
    world   = tgt_affine @ vox_tgt                   # (4, N)
    vox_src = src_inv @ world                         # (4, N) in source voxels

    coords  = vox_src[:3]
    resampled = map_coordinates(src_data, coords, order=0,  # order=0 = nearest
                                  mode="constant", cval=0.0)
    resampled = resampled.reshape(tgt_shape)
    return nib.Nifti1Image(resampled.astype(np.float32), tgt_affine)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  RESLICE MRI TO PLI SECTION PLANE
# ═══════════════════════════════════════════════════════════════════════════════

def reslice_mri_to_pli_plane(
    mri_path: Path,
    pli_proxy_path: Path,
    out_dir: Path,
) -> Path:
    """
    Reslice the MRI volume along the PLI section plane so that both images
    share the same 2D grid before registration.
    """
    print(f"\n[3/6] Reslicing MRI to PLI section plane")
    mri_nii   = nib.load(str(mri_path))
    proxy_nii = nib.load(str(pli_proxy_path))
    out_path  = out_dir / "mri_resliced.nii.gz"

    if HAS_NILEARN:
        resliced = resample_img(
            mri_nii,
            target_affine=proxy_nii.affine,
            target_shape=proxy_nii.shape[:3],
            interpolation="continuous",
        )
    else:
        resliced = _resample_nibabel(mri_nii, proxy_nii.affine, proxy_nii.shape[:3])

    nib.save(resliced, str(out_path))
    print(f"  MRI resliced to shape {resliced.shape}")
    print(f"  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  RIGID REGISTRATION  (ANTs preferred, dipy fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def register_pli_to_mri(
    fixed_path: Path,       # MRI resliced to PLI plane
    moving_path: Path,      # PLI proxy (downsampled)
    out_dir: Path,
    use_syn: bool = False,  # True = add deformable SyN after rigid
) -> tuple[Path, Path]:
    """
    Rigid (+ optional SyN) registration of PLI proxy onto MRI reference.

    Returns
    -------
    transform_path    : Path to saved transform (.mat for rigid, list for SyN)
    warped_proxy_path : Path to registered PLI proxy (for QC)
    """
    print(f"\n[4/6] Registering PLI → MRI")
    transform_path    = out_dir / "pli_to_mri_rigid.mat"
    warped_proxy_path = out_dir / "pli_proxy_registered.nii.gz"

    if HAS_ANTS:
        _register_ants(fixed_path, moving_path, out_dir,
                       transform_path, warped_proxy_path, use_syn)
    else:
        _register_dipy(fixed_path, moving_path, out_dir,
                       transform_path, warped_proxy_path)

    return transform_path, warped_proxy_path


def _register_ants(fixed_path, moving_path, out_dir,
                   transform_path, warped_proxy_path, use_syn):
    fixed  = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    # Normalise intensities for MI metric
    fixed  = ants.iMath(fixed,  "Normalize")
    moving = ants.iMath(moving, "Normalize")

    transform_type = "SyNRA" if use_syn else "Rigid"
    print(f"  ANTs registration type: {transform_type}")
    print(f"  Fixed  shape: {fixed.shape}  spacing: {fixed.spacing}")
    print(f"  Moving shape: {moving.shape} spacing: {moving.spacing}")

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        aff_metric="mattes",          # Mattes mutual information — cross-modality
        aff_sampling=32,
        grad_step=0.1,
        flow_sigma=3,
        total_sigma=0,
        syn_metric="mattes",
        syn_sampling=32,
        reg_iterations=(1000, 500, 250, 100),
        verbose=False,
    )

    # Save forward transform
    import shutil
    shutil.copy(reg["fwdtransforms"][0], str(transform_path))
    if use_syn and len(reg["fwdtransforms"]) > 1:
        # SyN produces [warp_field, affine] — save both
        syn_warp_path = out_dir / "pli_to_mri_syn_warp.nii.gz"
        shutil.copy(reg["fwdtransforms"][1], str(syn_warp_path))
        print(f"  SyN warp field saved: {syn_warp_path}")

    ants.image_write(reg["warpedmovout"], str(warped_proxy_path))
    print(f"  Transform saved: {transform_path}")
    print(f"  Warped proxy saved: {warped_proxy_path}")

    # Report overlap metric
    mi = ants.image_mutual_information(fixed, reg["warpedmovout"])
    print(f"  Mutual information after registration: {mi:.4f}")


def _register_dipy(fixed_path, moving_path, out_dir,
                   transform_path, warped_proxy_path):
    """
    Dipy affine registration fallback (no ANTs required).
    Less robust for large deformations but works for moderate misalignment.
    """
    try:
        from dipy.align.imaffine import (AffineRegistration,
                                          MutualInformationMetric,
                                          AffineMap)
        from dipy.align.transforms import RigidTransform3D
    except ImportError:
        raise ImportError(
            "Neither antspyx nor dipy is installed.\n"
            "Install one of:\n"
            "  pip install antspyx\n"
            "  pip install dipy"
        )

    print("  Using dipy affine registration (antspyx fallback)")
    fixed_nii  = nib.load(str(fixed_path))
    moving_nii = nib.load(str(moving_path))

    fixed_data  = fixed_nii.get_fdata().squeeze().astype(np.float64)
    moving_data = moving_nii.get_fdata().squeeze().astype(np.float64)

    # Normalise
    fixed_data  = (fixed_data  - fixed_data.min())  / (fixed_data.ptp()  + 1e-8)
    moving_data = (moving_data - moving_data.min()) / (moving_data.ptp() + 1e-8)

    metric    = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    affreg    = AffineRegistration(metric=metric,
                                    level_iters=[1000, 100, 10],
                                    sigmas=[3.0, 1.0, 0.0],
                                    factors=[4, 2, 1])
    transform = RigidTransform3D()
    affine_map = affreg.optimize(
        fixed_data,  moving_data,
        transform,   None,
        fixed_nii.affine, moving_nii.affine,
    )

    # Save transform as .npy (4×4 matrix)
    np.save(str(transform_path).replace(".mat", ".npy"), affine_map.affine)
    transform_path = Path(str(transform_path).replace(".mat", ".npy"))

    # Apply and save warped moving image
    warped = affine_map.transform(moving_data)
    warped_nii = nib.Nifti1Image(warped.astype(np.float32),
                                   fixed_nii.affine, fixed_nii.header)
    nib.save(warped_nii, str(warped_proxy_path))
    print(f"  Transform saved: {transform_path}")
    print(f"  Warped proxy saved: {warped_proxy_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  QC — CHECKERBOARD OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def qc_registration(
    fixed_path: Path,
    warped_proxy_path: Path,
    out_dir: Path,
    n_blocks: int = 8,
):
    """
    Save a checkerboard overlay and a side-by-side comparison figure.
    Green contour overlay is added for edge-alignment inspection.
    """
    print(f"\n[5/6] QC registration overlay")
    fixed_data  = nib.load(str(fixed_path)).get_fdata().squeeze()
    warped_data = nib.load(str(warped_proxy_path)).get_fdata().squeeze()

    # Handle 3D by taking the middle slice if needed
    if fixed_data.ndim == 3:
        z = fixed_data.shape[2] // 2
        fixed_data  = fixed_data[:, :, z]
        warped_data = warped_data[:, :, z]

    # Normalise both to [0, 1]
    def _norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    f = _norm(fixed_data)
    w = _norm(warped_data)

    # Checkerboard blend
    h, ww = f.shape
    block = max(h, ww) // n_blocks
    checker = np.zeros((h, ww), dtype=bool)
    for i in range(0, h, block):
        for j in range(0, ww, block):
            if (i // block + j // block) % 2 == 0:
                checker[i:i+block, j:j+block] = True
    blend = np.where(checker, f, w)

    # RGB overlay (fixed=gray, warped=warm tint)
    overlay = np.stack([
        np.where(checker, f, w * 0.6),
        np.where(checker, f, w * 0.4),
        np.where(checker, f, w * 0.1),
    ], axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(f,       cmap="gray",   origin="lower"); axes[0].set_title("MRI (fixed)")
    axes[1].imshow(w,       cmap="hot",    origin="lower"); axes[1].set_title("PLI registered")
    axes[2].imshow(blend,   cmap="gray",   origin="lower"); axes[2].set_title("Checkerboard")
    axes[3].imshow(overlay,                origin="lower"); axes[3].set_title("Colour overlay")
    for ax in axes:
        ax.axis("off")

    out_path = out_dir / "qc_registration.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  QC figure: {out_path}")
    print("  >>> Inspect qc_registration.png before proceeding to Step 6 <<<")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  APPLY TRANSFORM TO FULL-RESOLUTION PLI  (chunked, memory-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_transform_full_res(
    pli_masked_path: Path,
    wm_mask_path: Path,
    mri_ref_path: Path,
    transform_path: Path,
    out_dir: Path,
    chunk_size: int = 20,           # slices per chunk — tune to your RAM
) -> tuple[Path, Path]:
    """
    Apply the registration transform to the full-resolution PLI (3.5 GB).

    Processes `chunk_size` slices at a time to stay within RAM limits.
    A machine with 16 GB RAM can safely use chunk_size=20 for typical PLI.

    Returns
    -------
    phi_reg_path  : registered orientation map in MRI space
    mask_reg_path : registered WM mask in MRI space
    """
    print(f"\n[6/6] Applying transform to full-resolution PLI")
    print(f"  Input : {pli_masked_path}")
    print(f"  RAM strategy: {chunk_size} slices per chunk")

    pli_nii  = nib.load(str(pli_masked_path))
    mask_nii = nib.load(str(wm_mask_path))
    mri_nii  = nib.load(str(mri_ref_path))
    ref_shape = mri_nii.shape[:3]

    phi_out  = np.zeros(ref_shape, dtype=np.float32)
    mask_out = np.zeros(ref_shape, dtype=np.uint8)

    n_slices  = pli_nii.shape[2] if pli_nii.ndim >= 3 else 1
    transform = str(transform_path)

    if HAS_ANTS:
        ref_ants = ants.image_read(str(mri_ref_path))
        # Determine list of transforms (rigid .mat, or [warp, affine] for SyN)
        syn_warp = out_dir / "pli_to_mri_syn_warp.nii.gz"
        if syn_warp.exists():
            transform_list = [str(syn_warp), transform]
        else:
            transform_list = [transform]

        for start in range(0, n_slices, chunk_size):
            end = min(start + chunk_size, n_slices)
            print(f"  Chunk slices {start}–{end-1} / {n_slices-1}")

            for z in range(start, end):
                # Lazy-load one slice at a time from the dataobj proxy
                phi_slice  = np.array(pli_nii.dataobj[:, :, z]).astype(np.float32)
                mask_slice = np.array(mask_nii.dataobj[:, :, z]).astype(np.float32)

                # Wrap as 3D single-slice NIfTI for ANTs
                sl_affine = pli_nii.affine.copy()
                # Offset the affine origin to this slice's z position
                sl_affine[:3, 3] += z * pli_nii.affine[:3, 2]

                def _warp_slice(data_slice, interp):
                    sl_nii  = nib.Nifti1Image(
                        data_slice[:, :, np.newaxis].astype(np.float32),
                        sl_affine
                    )
                    sl_ants = ants.from_nibabel(sl_nii)
                    warped  = ants.apply_transforms(
                        fixed=ref_ants,
                        moving=sl_ants,
                        transformlist=transform_list,
                        interpolator=interp,
                    )
                    return warped.numpy().squeeze()

                phi_out[:, :, z]  = _warp_slice(phi_slice,  "nearestNeighbor")
                mask_out[:, :, z] = _warp_slice(mask_slice, "nearestNeighbor").astype(np.uint8)

    else:
        # Dipy / nibabel fallback — load full transform matrix
        print("  Using nibabel/scipy resampling (antspyx not found)")
        tmat_path = str(transform_path).replace(".mat", ".npy")
        if not Path(tmat_path).exists():
            raise FileNotFoundError(
                f"Transform file not found: {tmat_path}\n"
                "Run registration step first."
            )
        transform_matrix = np.load(tmat_path)

        # Build composite affine: PLI→world→MRI_voxel
        pli_to_mri_vox = np.linalg.inv(mri_nii.affine) @ transform_matrix @ pli_nii.affine

        from scipy.ndimage import map_coordinates
        for start in range(0, n_slices, chunk_size):
            end = min(start + chunk_size, n_slices)
            print(f"  Chunk slices {start}–{end-1} / {n_slices-1}")
            chunk_phi  = np.array(pli_nii.dataobj[:, :, start:end]).astype(np.float32)
            chunk_mask = np.array(mask_nii.dataobj[:, :, start:end]).astype(np.float32)

            # Map MRI voxel coordinates back to PLI source voxels
            i, j, k = np.meshgrid(
                np.arange(ref_shape[0]),
                np.arange(ref_shape[1]),
                np.arange(start, end),
                indexing="ij",
            )
            vox = np.stack([i.ravel(), j.ravel(), k.ravel(),
                             np.ones(i.size)], axis=0)
            src = pli_to_mri_vox @ vox
            coords = src[:3]

            phi_chunk = map_coordinates(
                np.array(pli_nii.dataobj), coords, order=0, mode="constant"
            ).reshape(ref_shape[0], ref_shape[1], end - start)
            mask_chunk = map_coordinates(
                np.array(mask_nii.dataobj), coords, order=0, mode="constant"
            ).reshape(ref_shape[0], ref_shape[1], end - start)

            phi_out[:, :, start:end]  = phi_chunk.astype(np.float32)
            mask_out[:, :, start:end] = mask_chunk.astype(np.uint8)

    # Save outputs
    phi_reg_path  = out_dir / "phi_registered_to_mri.nii.gz"
    mask_reg_path = out_dir / "wm_mask_registered_to_mri.nii.gz"

    nib.save(nib.Nifti1Image(phi_out,  mri_nii.affine, mri_nii.header), str(phi_reg_path))
    nib.save(nib.Nifti1Image(mask_out, mri_nii.affine, mri_nii.header), str(mask_reg_path))

    print(f"\n  Registered PLI: {phi_reg_path}")
    print(f"  Registered mask: {mask_reg_path}")
    return phi_reg_path, mask_reg_path


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  FINAL SUMMARY QC
# ═══════════════════════════════════════════════════════════════════════════════

def final_qc(
    phi_reg_path: Path,
    mri_ref_path: Path,
    out_dir: Path,
):
    """
    Overlay the registered PLI orientation on the MRI reference for a final
    sanity check that anatomical structures align.
    """
    print(f"\n[QC] Final overlay: registered PLI on MRI")
    phi  = nib.load(str(phi_reg_path)).get_fdata().squeeze()
    mri  = nib.load(str(mri_ref_path)).get_fdata().squeeze()

    # Use middle slice if 3D
    if phi.ndim == 3:
        z   = phi.shape[2] // 2
        phi = phi[:, :, z]
        mri = mri[:, :, z]

    mri_norm = (mri - mri.min()) / (mri.ptp() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mri_norm, cmap="gray",  origin="lower");                    axes[0].set_title("MRI reference")
    axes[1].imshow(phi,      cmap="hsv",   origin="lower",
                   vmin=-np.pi/2, vmax=np.pi/2);                              axes[1].set_title("Registered PLI φ")
    axes[2].imshow(mri_norm, cmap="gray",  origin="lower", alpha=0.6)
    axes[2].imshow(phi,      cmap="hsv",   origin="lower",
                   vmin=-np.pi/2, vmax=np.pi/2, alpha=0.5);                  axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    out_path = out_dir / "qc_final_overlay.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Final QC figure: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pli_path: str,
    mri_path: str,
    out_dir: str = "registration_output",
    # Masking params
    sigma_local: float = 3.0,
    grad_percentile: float = 87.0,
    r_erode: int = 4,
    r_dilate: int = 2,
    min_blob_px: int = 2000,
    # Registration params
    use_syn: bool = False,
    # Memory params
    chunk_size: int = 20,
    # Control flow
    skip_preprocessing: bool = False,
    skip_registration: bool = False,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")

    # ── Step 1: Preprocess PLI ------------------------------------------------
    if skip_preprocessing:
        phi_path  = out / "phi_masked.nii.gz"
        mask_path = out / "wm_mask.nii.gz"
        if not phi_path.exists():
            raise FileNotFoundError(f"skip_preprocessing=True but {phi_path} not found")
        print(f"[1/6] Skipping preprocessing — using {phi_path}")
    else:
        phi_path, mask_path = preprocess_pli(
            pli_path, out,
            sigma_local=sigma_local,
            grad_percentile=grad_percentile,
            r_erode=r_erode,
            r_dilate=r_dilate,
            min_blob_px=min_blob_px,
        )

    # ── Step 2: Downsample proxy ----------------------------------------------
    proxy_path = downsample_pli_to_mri(phi_path, mri_path, out)

    # ── Step 3: Reslice MRI ---------------------------------------------------
    mri_resliced_path = reslice_mri_to_pli_plane(mri_path, proxy_path, out)

    # ── Step 4: Register ------------------------------------------------------
    if skip_registration:
        transform_path    = out / "pli_to_mri_rigid.mat"
        warped_proxy_path = out / "pli_proxy_registered.nii.gz"
        if not transform_path.exists():
            transform_path = out / "pli_to_mri_rigid.npy"
        if not transform_path.exists():
            raise FileNotFoundError(
                f"skip_registration=True but no transform found in {out}"
            )
        print(f"[4/6] Skipping registration — using {transform_path}")
    else:
        transform_path, warped_proxy_path = register_pli_to_mri(
            mri_resliced_path, proxy_path, out, use_syn=use_syn
        )

    # ── Step 5: QC -----------------------------------------------------------
    qc_registration(mri_resliced_path, warped_proxy_path, out)

    # ── Step 6: Apply to full-res --------------------------------------------
    phi_reg_path, mask_reg_path = apply_transform_full_res(
        phi_path, mask_path, mri_path, transform_path, out,
        chunk_size=chunk_size,
    )

    # ── Final QC -------------------------------------------------------------
    final_qc(phi_reg_path, mri_path, out)

    # ── Summary --------------------------------------------------------------
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"  phi_registered_to_mri.nii.gz  → {phi_reg_path}")
    print(f"  wm_mask_registered_to_mri.nii.gz → {mask_reg_path}")
    print(f"  transform                      → {transform_path}")
    print(f"  QC figures                     → {out}/qc_*.png")
    print("\nNext steps:")
    print("  1. Inspect qc_registration.png and qc_final_overlay.png")
    print("  2. If alignment is poor: re-run with use_syn=True for deformable reg")
    print("  3. Run MRI tractography seeded from wm_mask_registered_to_mri.nii.gz")
    print("  4. Extract PLI streamlines from phi_registered_to_mri.nii.gz")
    print("  5. Both tractographies are now in MRI world coordinates → LSTM encoder")

    return {
        "phi_registered":  phi_reg_path,
        "mask_registered": mask_reg_path,
        "transform":       transform_path,
        "output_dir":      out,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="PLI in-plane orientation → MRI registration pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pli",  help="Path to PLI in-plane orientation .nii.gz (3.5 GB)")
    p.add_argument("mri",  help="Path to MRI reference volume .nii.gz (FA map or b0)")
    p.add_argument("-o", "--out-dir", default="registration_output",
                   help="Output directory")
    # Masking
    p.add_argument("--sigma-local",      type=float, default=3.0,
                   help="Coherence window sigma (px) — increase for noisier images")
    p.add_argument("--grad-percentile",  type=float, default=87.0,
                   help="Gradient magnitude exclusion percentile (80–92)")
    p.add_argument("--r-erode",          type=int,   default=4,
                   help="Erosion disk radius (px) — scale to your pixel size")
    p.add_argument("--r-dilate",         type=int,   default=2,
                   help="Re-dilation disk radius (px) — always < r_erode")
    p.add_argument("--min-blob-px",      type=int,   default=2000,
                   help="Minimum connected component size to keep (px²)")
    # Registration
    p.add_argument("--use-syn",          action="store_true",
                   help="Add SyN deformable registration after rigid (slower)")
    # Memory
    p.add_argument("--chunk-size",       type=int,   default=20,
                   help="Slices per chunk when applying transform to full-res PLI")
    # Skip flags
    p.add_argument("--skip-preprocessing",  action="store_true",
                   help="Skip preprocessing (reuse existing phi_masked.nii.gz)")
    p.add_argument("--skip-registration",   action="store_true",
                   help="Skip registration (reuse existing transform)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        pli_path=args.pli,
        mri_path=args.mri,
        out_dir=args.out_dir,
        sigma_local=args.sigma_local,
        grad_percentile=args.grad_percentile,
        r_erode=args.r_erode,
        r_dilate=args.r_dilate,
        min_blob_px=args.min_blob_px,
        use_syn=args.use_syn,
        chunk_size=args.chunk_size,
        skip_preprocessing=args.skip_preprocessing,
        skip_registration=args.skip_registration,
    )
