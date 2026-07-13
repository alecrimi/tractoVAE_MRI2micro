import numpy as np
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.data import get_sphere
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import nibabel as nib
import tifffile as tiff
import os
 
# ==========================================================
# PARAMETERS
# ==========================================================
image_path = "test.tif"
output_trk = "microscopy_tractography.trk"
structure_sigma = 1.0
seed_density = 1 # Seeds per voxel
fa_threshold = 0.1  # Lowered threshold to get more seeds

# ==========================================================
# LOAD IMAGE
# ==========================================================
data = tiff.imread(image_path)
print(f"Loaded image stack with shape {data.shape}")

# If 2D RGB, make it 3D grayscale stack
if data.ndim == 3 and data.shape[-1] == 3:
    data_gray = np.mean(data, axis=-1)
elif data.ndim == 4 and data.shape[-1] == 3:
    # 3D RGB stack (Z, Y, X, RGB)
    data_gray = np.mean(data, axis=-1)
else:
    data_gray = data.astype(float)

# Create an identity affine (voxel size = 1)
affine = np.eye(4)

# ==========================================================
# COMPUTE STRUCTURE TENSOR & EIGENVALUES
# ==========================================================
# Pass the structure tensor components as a tuple
structure_components = structure_tensor(data_gray, sigma=structure_sigma)
eigenvals = structure_tensor_eigenvalues(structure_components)

print(f"Eigenvalues shape: {eigenvals.shape}")
print(f"Number of eigenvalue arrays: {len(eigenvals)}")

for i, ev in enumerate(eigenvals):
    print(f"  Eigenvalue {i}: min={ev.min():.6f}, max={ev.max():.6f}, mean={ev.mean():.6f}")
    print(f"    NaN count: {np.isnan(ev).sum()}, Inf count: {np.isinf(ev).sum()}")


# Orientation proxy from gradients
gradients = np.stack(np.gradient(data_gray), axis=-1)
norm = np.linalg.norm(gradients, axis=-1, keepdims=True)
norm[norm == 0] = 1
main_orientation = gradients / norm

# ==========================================================
# FA-LIKE METRIC (FIXED: eigenvals are sorted LARGEST to SMALLEST)
# ==========================================================
lambda1 = eigenvals[0]  # Largest eigenvalue
lambda2 = eigenvals[1]  # Middle eigenvalue  
lambda3 = eigenvals[2]  # Smallest eigenvalue

# FA formula: measures anisotropy
# FA = sqrt(1/2) * sqrt((λ1-λ2)² + (λ2-λ3)² + (λ1-λ3)²) / sqrt(λ1² + λ2² + λ3²)
numerator = np.sqrt((lambda1 - lambda2)**2 + (lambda2 - lambda3)**2 + (lambda1 - lambda3)**2)
denominator = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)

FA = np.zeros_like(lambda1)
valid = denominator > 0
FA[valid] = np.sqrt(0.5) * numerator[valid] / denominator[valid]
FA = np.clip(FA, 0, 1)

print(f"\nFA range: {FA.min():.4f} to {FA.max():.4f}")
print(f"FA mean: {FA.mean():.4f}")

mask = FA > fa_threshold
print(f"\nUsing threshold {fa_threshold}:")
print(f"Mask coverage: {mask.sum()} / {mask.size} voxels ({100*mask.sum()/mask.size:.2f}%)")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask shape: {mask.shape}")


stopping_criterion = ThresholdStoppingCriterion(FA, fa_threshold)

# ==========================================================
# SEED GENERATION
# ==========================================================
seeds = utils.seeds_from_mask(mask, density=seed_density, affine=affine)
print(f"Number of seeds: {len(seeds)}")
print("\nGenerating seeds...")
print(f"Affine matrix:\n{affine}")


if len(seeds) == 0:
    print("ERROR: No seeds generated. Try lowering fa_threshold.")
    exit(1)

# ==========================================================
# DIRECTION GETTER - Create PMF from orientations
# ==========================================================
sphere = get_sphere(name='symmetric724')

# Convert orientation vectors to PMF (probability mass function)
# PMF should have shape (*image_shape, n_directions)
pmf = np.zeros(data_gray.shape + (len(sphere.vertices),))

# For each voxel, find the sphere direction closest to the gradient orientation
for idx in np.ndindex(data_gray.shape):
    vec = main_orientation[idx]
    # Compute dot product with all sphere directions
    dots = np.abs(sphere.vertices @ vec)
    # Set the closest direction to 1
    pmf[idx + (np.argmax(dots),)] = 1.0

dg = DeterministicMaximumDirectionGetter.from_pmf(
    pmf,
    max_angle=30.0,
    sphere=sphere
)

# ==========================================================
# LOCAL TRACKING
# ==========================================================
streamlines_generator = LocalTracking(
    dg,
    stopping_criterion,
    seeds,
    affine,
    step_size=0.5,
    return_all=False
)

streamlines = Streamlines(streamlines_generator)
print(f"Generated {len(streamlines)} streamlines")

# ==========================================================
# SAVE OUTPUT 
# ==========================================================
if len(streamlines) > 0:

    # Create a NIfTI image object as reference
    nifti_img = nib.Nifti1Image(data_gray, affine)
 
    # Create a StatefulTractogram object (required for newer DIPY versions)
    sft = StatefulTractogram(streamlines, nifti_img, Space.VOX)
    save_trk(sft, output_trk)
    print(f"Tractography saved as {output_trk}")
else:
    print("No streamlines generated.")
