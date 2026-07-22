#!/usr/bin/env bash
# ============================================================================
# register_ants.sh (v2 -- diagonal-misregistration fix)
#
# Registers a microscopy volume (moving image) onto an MRI volume
# (fixed image) using ANTs, and saves the resulting linear transform
# ("projection matrix") to disk.
#
# CHANGES FROM PREVIOUS VERSION:
#   1. Writes out the center-of-mass-only initialization as its own
#      warped image (init_check_Warped.nii.gz) BEFORE any optimization
#      runs, so you can immediately tell in ITK-SNAP whether a bad
#      diagonal orientation is already present at init (header/direction
#      mismatch) or only appears after Rigid/Similarity (optimizer
#      exploiting the underconstrained tilt axes on a thin 13-section
#      slab).
#   2. Coarsest shrink factor lowered from 8 to 4, and the top
#      convergence stage shortened, so the optimizer has less room to
#      wander into a totally different rotation before the finer levels
#      lock in. This directly targets the "thin sparse slab tips into a
#      diagonal that MI still scores well" failure mode.
#   3. Rigid gradient step lowered slightly (0.1 -> 0.05) so the first
#      stage moves more cautiously instead of jumping into a bad basin
#      early.
#
# Usage:
#   ./register_ants.sh
#
# Requires ANTs to be installed and on your PATH (antsRegistration,
# antsApplyTransforms available).
# ============================================================================
set -eu

# ---- Inputs -----------------------------------------------------------
FIXED="b0_fixed_trueorient.nii.gz"                    # MRI  -> reference / target space (header-corrected, see fix_b0_header.py)
MOVING_RAW="registered_stack_fixed_final.nii.gz"      # microscopy, WRONG per-axis spacing (see fix_microscopy_spacing.py)
MOVING="registered_stack_fixed_final_spacingfixed.nii.gz"  # microscopy after empirical spacing correction

# ---- Output prefix ------------------------------------------------------
OUT_PREFIX="micro2mri_"
# This will produce (among others):
#   ${OUT_PREFIX}0GenericAffine.mat   <-- the "projection matrix" you need
#   ${OUT_PREFIX}Warped.nii.gz        <-- moving image resampled into fixed space

mkdir -p ants_out
cd ants_out
cp ../"${FIXED}" .
cp ../"${MOVING_RAW}" .

# ----------------------------------------------------------------------
# EMPIRICAL SPACING FIX: registered_stack_fixed_final.nii.gz's X/Y
# spacing was confirmed wrong (anisotropic mismatch vs the MRI FOV --
# see check_tissue_bbox.py results: tissue fills canvas edge-to-edge at
# 159.9x261.9mm vs the MRI's actual 51.2x38.4mm FOV, same anatomy).
# This is not a real calibration fix -- see fix_microscopy_spacing.py's
# docstring -- but it's the best available correction until the
# upstream script that set this spacing is found and fixed properly.
# ----------------------------------------------------------------------
echo "== Applying empirical X/Y spacing correction to moving image =="
python3 ../fix_microscopy_spacing.py "${MOVING_RAW}" "${MOVING}"

# ----------------------------------------------------------------------
# Downsample microscopy for registration (physical extent preserved, so
# the resulting transform is valid at full resolution too).
# ----------------------------------------------------------------------
MOVING_DS_SPACING="0.15"   # mm, in-plane
MOVING_DS="${MOVING%.nii.gz}_ds.nii.gz"

echo "== Downsampling ${MOVING} to ${MOVING_DS_SPACING}mm in-plane for registration =="
python3 ../downsample_moving.py "${MOVING}" "${MOVING_DS}" "${MOVING_DS_SPACING}"

# ----------------------------------------------------------------------
# STEP 1 -- header/orientation sanity check.
# Print direction cosines for both images so a mismatched RAS/LPS
# convention or an axis swap is visible before we even run ANTs.
# ----------------------------------------------------------------------
echo "== Fixed image header =="
PrintHeader "${FIXED}" 2
echo "== Moving (downsampled) image header =="
PrintHeader "${MOVING_DS}" 2

# ----------------------------------------------------------------------
# STEP 2 -- header/orientation sanity check, done directly with Python
# (SimpleITK) instead of antsRegistration, so it's guaranteed fast and
# has no dependency on ANTs' internal optimizer stack. This computes a
# translation-only, center-of-mass alignment and writes it out for
# visual inspection. If it looks diagonal relative to the fixed image,
# the direction-cosine headers disagree -- fix that before trusting any
# antsRegistration output below.
# ----------------------------------------------------------------------
echo "== Computing quick center-of-mass check (Python, should take seconds) =="
python3 - "${FIXED}" "${MOVING_DS}" <<'PYEOF'
import sys
import SimpleITK as sitk

fixed_path, moving_path = sys.argv[1], sys.argv[2]
fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

init = sitk.CenteredTransformInitializer(
    fixed, moving, sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.MOMENTS)

resampled = sitk.Resample(moving, fixed, init, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
sitk.WriteImage(resampled, "init_check_Warped.nii.gz")
print("Wrote init_check_Warped.nii.gz")
PYEOF

echo "== Wrote ants_out/init_check_Warped.nii.gz -- inspect this in ITK-SNAP first. =="
echo "   If it is already diagonal/rotated relative to ${FIXED}, stop here:"
echo "   the direction-cosine headers disagree and need fixing before"
echo "   the full registration below will produce anything meaningful."
echo ""
echo "== Proceeding with full Rigid + Similarity registration =="

# ----------------------------------------------------------------------
# STEP 3 -- full registration.
#
# DIAGNOSIS (confirmed by comparing init_check_Warped.nii.gz against
# the final Warped/check_resampled images): the translation-only,
# moments-based init already sits correctly on the brain, axis-aligned,
# no diagonal. The diagonal only appears after Rigid/Similarity
# optimize rotation. So this is NOT a header/direction mismatch -- it's
# the optimizer exploiting the data-starved z-axis (13 sparse sections,
# 3.6mm thick) to tip the slab into a diagonal orientation that MI
# still scores well on, because there's almost no real signal
# constraining rotation around the in-plane tilt axes.
#
# FIX: constrain what the optimizer is allowed to do, rather than
# letting it search full 3-DOF rotation freely from scratch:
#   1. Build a fixed-image brain mask (Otsu + dilation) so MI can't
#      lock onto some other bright structure far from the true tissue
#      location.
#   2. Stage 1: Translation only (0 rotational DOF) to refine the
#      already-good moments-based position/offset.
#   3. Stage 2: Rigid, but with a much smaller gradient step and far
#      fewer iterations than before, starting from the good translation
#      -- so it can only nudge into a *small* rotation correction, not
#      wander into a completely different orientation.
#   4. Stage 3: Similarity, same tight constraints, to pick up the one
#      legitimate uniform scale factor (histology shrinkage).
# ----------------------------------------------------------------------
echo "== Building fixed-image mask (Otsu + dilation) to constrain the search =="
ThresholdImage 3 "${FIXED}" fixed_mask.nii.gz Otsu 1
ImageMath 3 fixed_mask.nii.gz MD fixed_mask.nii.gz 3   # dilate 3vox so we don't clip real tissue edges

echo "== Stage 1: translation-only refinement =="
stdbuf -oL -eL antsRegistration \
    --dimensionality 3 \
    --output ["${OUT_PREFIX}translation_","${OUT_PREFIX}translation_Warped.nii.gz"] \
    --interpolation Linear \
    --winsorize-image-intensities [0.005,0.995] \
    --use-histogram-matching 0 \
    --initial-moving-transform ["${FIXED}","${MOVING_DS}",1] \
    --masks ["fixed_mask.nii.gz","NA"] \
    --transform Translation[0.1] \
    --metric MI["${FIXED}","${MOVING_DS}",1,32,Regular,0.25] \
    --convergence [500x250x100,1e-6,10] \
    --shrink-factors 4x2x1 \
    --smoothing-sigmas 2x1x0vox

echo "== Stage 2+3: small-step, well-converged Rigid then Similarity correction =="
stdbuf -oL -eL antsRegistration \
    --dimensionality 3 \
    --output ["${OUT_PREFIX}","${OUT_PREFIX}Warped.nii.gz"] \
    --interpolation Linear \
    --winsorize-image-intensities [0.005,0.995] \
    --use-histogram-matching 0 \
    --initial-moving-transform "${OUT_PREFIX}translation_0GenericAffine.mat" \
    --masks ["fixed_mask.nii.gz","NA"] \
    --transform Rigid[0.01] \
    --metric MI["${FIXED}","${MOVING_DS}",1,32,Regular,0.25] \
    --convergence [300x150x50,1e-6,10] \
    --shrink-factors 4x2x1 \
    --smoothing-sigmas 2x1x0vox \
    --transform Similarity[0.01] \
    --metric MI["${FIXED}","${MOVING_DS}",1,32,Regular,0.25] \
    --convergence [300x150x50,1e-6,10] \
    --shrink-factors 4x2x1 \
    --smoothing-sigmas 2x1x0vox

echo "== Fitted transform parameters (sanity check) =="
python3 - "${OUT_PREFIX}0GenericAffine.mat" <<'PYEOF'
import sys
import SimpleITK as sitk
import numpy as np

t = sitk.ReadTransform(sys.argv[1])
params = t.GetParameters()
print(f"Raw parameters: {params}")

# For a Similarity3DTransform: [scale, rot_x, rot_y, rot_z(versor part varies), ...]
# Print the matrix and derive an approximate isotropic scale from its determinant,
# which works regardless of the exact transform type ANTs wrote out.
try:
    matrix = np.array(t.GetParameters()[:9] if len(t.GetParameters()) >= 12 else None)
except Exception:
    matrix = None

# Robust approach: read as AffineTransform to get the 3x3 matrix directly.
t2 = sitk.ReadTransform(sys.argv[1])
try:
    aff = sitk.AffineTransform(t2)
    M = np.array(aff.GetMatrix()).reshape(3, 3)
    scale_approx = np.linalg.det(M) ** (1.0 / 3.0)
    print(f"3x3 matrix:\n{M}")
    print(f"Approx isotropic scale (det^(1/3)): {scale_approx:.4f}")
    print("Sanity range for histology shrinkage: ~0.75-1.0 is plausible;")
    print(">1.3 or <0.6 suggests the fit may still be unstable.")
except Exception as e:
    print(f"Could not convert to AffineTransform for a matrix printout: {e}")
PYEOF

# NOTE: Rigid, then Similarity (rigid + ONE uniform scale factor, no
# shear, no per-axis stretch) -- not full Affine. Histological tissue
# shrinkage relative to in vivo MRI is a real, well-documented effect
# (commonly ~10-25%), so allowing *some* scaling is legitimate. Full
# 12-DOF Affine on a 3.6mm-thick sparse slab was observed to invent
# large shear to exploit the data-starved z axis, so it is deliberately
# not used here.
#
# After this runs, print the fitted scale factor (see the python
# snippet used earlier with sitk.ReadTransform) and sanity check:
# values roughly in 0.75-1.0 are consistent with known histology
# shrinkage; something like 1.6+ or <0.5 means the fit is still
# unstable and should not be trusted.

echo "== Done. Affine transform saved as: ants_out/${OUT_PREFIX}0GenericAffine.mat =="

# ---- Sanity check: resample FULL-RESOLUTION microscopy into MRI space -
antsApplyTransforms \
    -d 3 \
    -i "${MOVING}" \
    -r "${FIXED}" \
    -t "${OUT_PREFIX}0GenericAffine.mat" \
    -o "${OUT_PREFIX}check_resampled.nii.gz" \
    -n Linear

echo "== Sanity-check image written to: ants_out/${OUT_PREFIX}check_resampled.nii.gz =="
echo "   Open init_check_Warped.nii.gz, ${OUT_PREFIX}Warped.nii.gz and"
echo "   ${OUT_PREFIX}check_resampled.nii.gz alongside ${FIXED} in ITK-SNAP"
echo "   to see exactly which stage introduces the diagonal orientation."