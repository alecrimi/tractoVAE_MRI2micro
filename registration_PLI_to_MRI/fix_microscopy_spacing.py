#!/usr/bin/env python3
"""
fix_microscopy_spacing.py

EMPIRICAL spacing correction, not a true calibration fix. We don't have
access to the upstream script/metadata that originally set this
microscopy volume's X/Y voxel spacing (0.025mm), and we've confirmed:

  - Fixed (MRI) FOV:    51.2mm x 38.4mm x ~4.0mm
  - Moving (microscopy) canvas at current (wrong) spacing: 159.9mm x 261.9mm x 3.9mm
  - Tissue fills the moving canvas edge-to-edge (not an empty-background
    issue -- see check_tissue_bbox.py output)
  - Same anatomy in both (confirmed by user)
  - Z already matches closely (3.9mm vs 4.0mm) -- left unchanged

So as a stopgap, this script rescales ONLY the X/Y spacing so the
moving image's physical extent matches the fixed image's known FOV
exactly, and recomputes the origin so the volume's world-space CENTER
is preserved (same center point as before, not reset to world zero).
Z spacing/size are untouched.

THIS IS NOT A SUBSTITUTE FOR REAL CALIBRATION. The correction factor
came out different per axis (X needs /3.1, Y needs /6.8), which is
unusual for an isotropic-pixel microscope sensor and suggests the real
upstream bug may not be a simple spacing constant. Once registration
looks reasonable using this corrected file, cross-check with a real
landmark measurement (a known distance visible in both MRI and
microscopy) before trusting this for anything beyond initial QC.

Usage:
    python3 fix_microscopy_spacing.py <input.nii.gz> <output.nii.gz> \\
        [target_x_extent_mm] [target_y_extent_mm]

    Defaults to 51.2 and 38.4 (this dataset's known MRI FOV in X/Y).
"""
import sys
import numpy as np
import SimpleITK as sitk


def fix_spacing(in_path, out_path, target_x_extent=51.2, target_y_extent=38.4):
    img = sitk.ReadImage(in_path)
    size = img.GetSize()
    old_spacing = img.GetSpacing()
    old_origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)

    new_spacing_x = target_x_extent / size[0]
    new_spacing_y = target_y_extent / size[1]
    new_spacing = (new_spacing_x, new_spacing_y, old_spacing[2])

    print(f"Old spacing: {old_spacing}")
    print(f"New spacing: {new_spacing}")
    print(f"Old physical extent (x,y,z): "
          f"{[size[i]*old_spacing[i] for i in range(3)]}")
    print(f"New physical extent (x,y,z): "
          f"{[size[i]*new_spacing[i] for i in range(3)]}")

    center_vox = (np.array(size) - 1) / 2.0

    old_center_world = old_origin + direction @ (np.array(old_spacing) * center_vox)
    new_origin = old_center_world - direction @ (np.array(new_spacing) * center_vox)

    print(f"World-space center preserved at: {old_center_world}")

    img.SetSpacing(new_spacing)
    img.SetOrigin(tuple(new_origin))
    sitk.WriteImage(img, out_path)
    print(f"Saved: {out_path}")
    print("\nReminder: this is an empirical stopgap correction, not a")
    print("verified calibration. Cross-check with a real landmark")
    print("distance in both images once registration looks reasonable.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    target_x = float(sys.argv[3]) if len(sys.argv) > 3 else 51.2
    target_y = float(sys.argv[4]) if len(sys.argv) > 4 else 38.4
    fix_spacing(in_path, out_path, target_x, target_y)