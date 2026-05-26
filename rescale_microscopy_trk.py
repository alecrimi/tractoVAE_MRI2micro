from dipy.io.streamline import load_trk, save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.streamline import Streamlines
import nibabel as nib
import numpy as np

# ==========================================================
# PARAMETERS — only thing to tune
# ==========================================================
Z_SCALE = 50.0    # multiply Z by this for display
                  # 50 = makes 0.768mm look like 38mm
                  # adjust until it looks good in TrackVis

# ==========================================================
# LOAD
# ==========================================================
trk       = load_trk("microscopy_tractography.trk", "same")
micro_img = nib.load("registered_stack_fixed.nii.gz")
mri_img   = nib.load("data.nii.gz")

# Read Z info directly from headers — no hardcoding
micro_z_vox  = float(micro_img.header.get_zooms()[2])
micro_z_slices = micro_img.header.get_data_shape()[2]
micro_z_mm   = micro_z_vox * micro_z_slices

mri_z_vox    = float(mri_img.header.get_zooms()[2])
mri_z_slices = mri_img.header.get_data_shape()[2]
mri_z_mm     = mri_z_vox * mri_z_slices

auto_scale   = mri_z_mm / micro_z_mm

print(f"Micro Z : {micro_z_slices} slices × {micro_z_vox:.4f} mm "
      f"= {micro_z_mm:.3f} mm")
print(f"MRI Z   : {mri_z_slices} slices × {mri_z_vox:.4f} mm "
      f"= {mri_z_mm:.3f} mm")
print(f"Auto scale (match MRI): {auto_scale:.1f}x")
print(f"Using Z_SCALE          : {Z_SCALE:.1f}x")

trk.to_rasmm()

# ==========================================================
# RESCALE Z
# ==========================================================
scaled = []
for s in trk.streamlines:
    s2        = s.copy()
    s2[:, 2] *= Z_SCALE
    scaled.append(s2)

# Update affine Z spacing to match
new_affine        = micro_img.affine.copy()
new_affine[2, 2] *= Z_SCALE

ref = nib.Nifti1Image(
    np.zeros(micro_img.shape, dtype=np.float32),
    new_affine, micro_img.header)

sft = StatefulTractogram(Streamlines(scaled), ref, Space.RASMM)
save_trk(sft, "microscopy_tractography_zscaled.trk",
         bbox_valid_check=False)

print(f"\nNew Z extent : {micro_z_mm * Z_SCALE:.1f} mm")
print(f"Saved → microscopy_tractography_zscaled.trk")