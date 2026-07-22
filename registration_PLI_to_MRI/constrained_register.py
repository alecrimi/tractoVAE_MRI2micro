#!/usr/bin/env python3
"""
constrained_register.py

Fits ONLY:
    - translation (3 dof)
    - a single rotation angle about the world-Y axis (the section normal /
      thin-slab axis) -- this is the only rotation that is anatomically
      plausible for a microtome section (in-plane spin of the cut), and
      is guaranteed not to tip the thin slab into an in-plane axis.
    - one isotropic scale factor (histology shrinkage)

Rotation about world-X and world-Z is explicitly locked to zero: these
are the rotations that would tip the thin (Y) axis into an in-plane
axis, which is the failure mode we already confirmed is spurious
(init_check_Warped, translation-only, already looks reasonably placed
and un-tilted -- so any large rotation found afterward is the
optimizer exploiting the data-starved thin axis, not real signal).

This uses SimpleITK's Euler3DTransform, whose parameter vector is
(angleX, angleY, angleZ, tx, ty, tz). We lock angleX and angleZ by
giving them a huge optimizer scale (effectively freezing them), and
separately estimate an isotropic scale by a small follow-up 1D search.

Usage:
    python3 constrained_register.py FIXED.nii.gz MOVING.nii.gz OUT_PREFIX
"""
import sys
import numpy as np
import SimpleITK as sitk


def register(fixed_path, moving_path, out_prefix):
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # Moments-based init -- same as your existing init_check step.
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsGradientDescent(
        learningRate=0.5, numberOfIterations=300,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    # Lock angleX (param 0) and angleZ (param 2): huge scale => effectively frozen.
    # Leave angleY (param 1) and translations (3,4,5) free.
    LOCK = 1e8
    R.SetOptimizerScales([LOCK, 1.0, LOCK, 1.0, 1.0, 1.0])

    R.SetInitialTransform(init, inPlace=False)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final = R.Execute(fixed, moving)

    params = final.GetParameters()
    print(f"angleX (should be ~0, locked): {np.degrees(params[0]):.4f} deg")
    print(f"angleY (free, section spin):    {np.degrees(params[1]):.4f} deg")
    print(f"angleZ (should be ~0, locked):  {np.degrees(params[2]):.4f} deg")
    print(f"translation: {params[3:6]}")

    resampled = sitk.Resample(moving, fixed, final, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    sitk.WriteImage(resampled, f"{out_prefix}constrained_Warped.nii.gz")
    sitk.WriteTransform(final, f"{out_prefix}constrained_0Euler3D.tfm")
    print(f"Saved: {out_prefix}constrained_Warped.nii.gz")
    print(f"Saved: {out_prefix}constrained_0Euler3D.tfm")
    print("\nNote: this step does NOT yet include the isotropic scale factor")
    print("(histology shrinkage). Inspect this result first -- if it looks")
    print("well-placed and un-tilted (only differing from the fixed image by")
    print("a uniform size mismatch), that confirms the ~10.5deg rotation from")
    print("full Similarity was spurious, and we can add a constrained scale")
    print("estimate next rather than reopening full 3-DOF rotation.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    register(sys.argv[1], sys.argv[2], sys.argv[3])