#!/usr/bin/env python3

import numpy as np
import SimpleITK as sitk

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space


# ============================================================
# INPUTS
# ============================================================

trk_file = "microscopy_tractography_zscaled.trk"

reference_file = "ants_out/registered_stack_fixed_final_spacingfixed.nii.gz"
# This should be the image used as reference when the tractography
# was generated (i.e. still in MICROSCOPY space).

transform_file = "ants_out/micro2mri_0GenericAffine.mat"

output_file = "tractography_in_mri_space.trk"


# ============================================================
# Load tractography
# ============================================================

print("Loading tractography...")

sft = load_tractogram(
    trk_file,
    "same",
    bbox_valid_check=False
)

print(f"{len(sft.streamlines)} streamlines loaded")


# ============================================================
# Load ANTs transform
# ============================================================

print("Loading ANTs transform...")

transform = sitk.ReadTransform(transform_file)

print(transform)


# ============================================================
# Apply transform
# ============================================================

print("Transforming streamlines...")

new_streamlines = []

for sl in sft.streamlines:

    pts = np.asarray(sl)

    new_pts = np.zeros_like(pts)

    for i in range(len(pts)):
        new_pts[i] = np.asarray(
    transform.TransformPoint(pts[i].astype(np.float64).tolist()),
    dtype=np.float32
)

    new_streamlines.append(new_pts)


# ============================================================
# Build new tractogram
# ============================================================

new_sft = StatefulTractogram.from_sft(
    new_streamlines,
    sft
)

# preserve metadata
new_sft.data_per_streamline = sft.data_per_streamline
new_sft.data_per_point = sft.data_per_point


# ============================================================
# Save
# ============================================================

save_tractogram(
    new_sft,
    output_file,
    bbox_valid_check=False
)

print(f"Saved: {output_file}")