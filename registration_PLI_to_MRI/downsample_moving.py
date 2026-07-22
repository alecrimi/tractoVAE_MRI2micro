#!/usr/bin/env python3
"""
downsample_moving.py (fixed)

Downsamples a (very large) microscopy volume for ANTs registration, and
casts it to float32 to cut memory further. The physical (mm) extent of
the image is preserved, so the affine transform ANTs computes on the
downsampled version is directly valid at full resolution too -- you
only downsample to make the registration computation feasible, not
because you need the transform itself at low res.

CHANGES FROM ORIGINAL:
  1. In-plane (x/y) and through-stack (z) target spacing are now
     independent. For a sparse-section stack (e.g. 13 real physical
     sections), you only have that many real z samples -- resampling
     z to a finer isotropic grid (e.g. 0.15mm) doesn't recover missing
     anatomy, it fabricates synthetic slices between real data points
     that ANTs will then treat as real signal. By default this script
     now leaves z at its native spacing and only coarsens x/y.
  2. Added a memory/sanity guard: prints the requested output voxel
     count and refuses to run if it would need an unreasonable amount
     of RAM, so a bad spacing value (wrong axis, wrong units, etc.)
     fails fast with a clear message instead of crashing deep inside
     ITK's allocator.

Usage:
    python3 downsample_moving.py <input.nii.gz> <output.nii.gz> \\
        [target_xy_spacing_mm] [target_z_spacing_mm]

    If target_z_spacing_mm is omitted, the native z spacing from the
    input header is kept as-is (recommended for sparse section stacks).
    Pass target_z_spacing_mm explicitly only if you deliberately want
    to resample z too (e.g. it's a dense/isotropic acquisition, not a
    sparse section stack).

Example (typical sparse-section case -- only touch x/y):
    python3 downsample_moving.py registered_stack_fixed_final.nii.gz \\
                                  registered_stack_fixed_ds.nii.gz 0.15

Example (explicitly also resample z, e.g. dense acquisition):
    python3 downsample_moving.py in.nii.gz out.nii.gz 0.15 0.15
"""
import sys
import SimpleITK as sitk

MAX_VOXELS = 500_000_000  # ~2GB at float32; adjust if your machine has more headroom


def downsample(in_path: str, out_path: str, target_xy: float = 0.15,
                target_z: float = None):
    img = sitk.ReadImage(in_path)
    img = sitk.Cast(img, sitk.sitkFloat32)  # halves memory vs double

    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()

    z_spacing = orig_spacing[2] if target_z is None else target_z
    new_spacing = [target_xy, target_xy, z_spacing]

    new_size = [
        max(1, int(round(orig_size[i] * (orig_spacing[i] / new_spacing[i]))))
        for i in range(3)
    ]
    voxel_count = new_size[0] * new_size[1] * new_size[2]

    print(f"Input:  size={orig_size}, spacing={orig_spacing}")
    print(f"Requested output: size={new_size}, spacing={new_spacing}")
    print(f"Requested voxel count: {voxel_count:,} "
          f"(~{voxel_count * 4 / 1e9:.2f} GB at float32)")

    if target_z is None:
        print(f"z-axis spacing left at native value ({orig_spacing[2]}mm) -- "
              f"not resampling through a sparse stack.")

    if voxel_count > MAX_VOXELS:
        print(f"\nABORTING: requested voxel count exceeds the safety limit "
              f"({MAX_VOXELS:,}). This usually means a spacing value is "
              f"wrong (wrong axis, wrong units, or a stray large/small "
              f"number) rather than a genuine memory requirement. Double "
              f"check orig_spacing above before overriding MAX_VOXELS.")
        sys.exit(1)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(img)

    sitk.WriteImage(resampled, out_path)
    print(f"Saved downsampled volume to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    target_xy = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    target_z = float(sys.argv[4]) if len(sys.argv) > 4 else None
    downsample(in_path, out_path, target_xy, target_z)