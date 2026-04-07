import ants
import nibabel as nib
import numpy as np

# =========================
# CONFIG
# =========================
microscopy_path = "registered_stack.nii.gz"
mri_path = "data.nii.gz"

COARSE_FACTOR = 12   # aggressive for safety

# =========================
# LOAD MRI (small anyway)
# =========================
print("Loading MRI...")
nii_mri = nib.load(mri_path)

mri_data = np.array(nii_mri.dataobj[..., 0], dtype=np.float32)

mri_data = np.nan_to_num(mri_data)
fixed = ants.from_numpy(mri_data)
fixed.set_spacing((1,1,1))

del mri_data

# =========================
# LOAD MICROSCOPY (LAZY + STRIDED)
# =========================
print("Loading microscopy lazily...")

nii_micro = nib.load(microscopy_path)
proxy = nii_micro.dataobj   # 🚨 this is lazy, NOT loaded

# 🚨 KEY: downsample BEFORE loading into RAM
print("Subsampling microscopy on disk...")

micro_small = np.array(
    proxy[::COARSE_FACTOR, ::COARSE_FACTOR, ::COARSE_FACTOR],
    dtype=np.float32
)

# Clean data
micro_small = np.nan_to_num(micro_small)

p1, p99 = np.percentile(micro_small, (1, 99))
micro_small = np.clip(micro_small, p1, p99)

moving_coarse = ants.from_numpy(micro_small)
moving_coarse.set_spacing((1,1,1))

del micro_small  # free immediately

# =========================
# ALSO DOWNSAMPLE MRI TO MATCH
# =========================
fixed_coarse = ants.resample_image(
    fixed,
    moving_coarse.shape,
    use_voxels=True,
    interp_type=1
)

# =========================
# NORMALIZE
# =========================
fixed_coarse = ants.iMath(fixed_coarse, "Normalize")
moving_coarse = ants.iMath(moving_coarse, "Normalize")

# =========================
# REGISTRATION (COARSE ONLY)
# =========================
print("Registration...")

reg = ants.registration(
    fixed=fixed_coarse,
    moving=moving_coarse,
    type_of_transform="Affine",
    aff_metric="MI",
    reg_iterations=(100, 50, 25),
)

# =========================
# ⚠️ APPLY TO FULL RES (STREAMING)
# =========================
print("Reloading full microscopy for final transform...")

# ⚠️ This is still heavy — do ONLY affine first
nii_micro = nib.load(microscopy_path)
full_data = np.array(nii_micro.dataobj, dtype=np.float32)

full_data = np.nan_to_num(full_data)

moving_full = ants.from_numpy(full_data)
moving_full.set_spacing((1,1,1))

final = ants.apply_transforms(
    fixed=fixed,
    moving=moving_full,
    transformlist=reg['fwdtransforms'],
    interpolator='linear'
)

ants.image_write(final, "registered.nii.gz")

print("DONE")
