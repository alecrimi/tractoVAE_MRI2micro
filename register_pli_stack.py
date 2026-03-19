"""
PLI In-Plane Slice Stack Registration
======================================
Registers serial 2D PLI in-plane TIFF slices into a coherent 3D volume
using ANTs SyNRA (rigid + affine + deformable SyN).

Workflow:
  1. Load all slices from in_plane/
  2. Pick the middle slice as the global reference (avoids drift)
  3. Register every other slice to that reference (SyNRA)
  4. Save the registered stack as a 3D NIfTI (.nii.gz) ready for
     3D structure tensor analysis

Requirements:
    pip install antspyx tifffile nibabel numpy

Usage:
    python register_pli_stack.py
    python register_pli_stack.py --input_dir in_plane --output registered_stack.nii.gz --voxel_size_um 25
"""

import argparse
import sys
from pathlib import Path

import os
import numpy as np
import tifffile
import nibabel as nib

try:
    import ants
except ImportError:
    sys.exit(
        "antspyx is not installed. Run:  pip install antspyx"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Register PLI in-plane slices with ANTs SyNRA")
    p.add_argument("--input_dir",    default="in_plane",
                   help="Folder containing *.tif PLI slices (default: in_plane)")
    p.add_argument("--output",       default="registered_stack.nii.gz",
                   help="Output 3D NIfTI file (default: registered_stack.nii.gz)")
    p.add_argument("--voxel_size_um", type=float, default=25.0,
                   help="In-plane pixel size in micrometres (default: 25). "
                        "Slice thickness (z) is assumed equal.")
    p.add_argument("--transform",    default="SyNRA",
                   choices=["Translation", "Rigid", "Affine", "SyNRA", "SyN"],
                   help="ANTs transform type (default: SyNRA = rigid+affine+deformable)")
    p.add_argument("--metric",       default="GC",
                   choices=["GC", "MI", "CC", "Demons"],
                   help="Registration metric (default: GC = global cross-correlation, "
                        "good for same-modality PLI slices)")
    p.add_argument("--threads",      type=int, default=1,
                   help="Number of ITK threads (default: 1)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_slice_as_float32(path: Path) -> np.ndarray:
    """Load a TIFF as a 2D float32 array normalised to [0, 1]."""
    arr = tifffile.imread(str(path)).astype(np.float32)

    # If RGB/RGBA, convert to greyscale luminance for registration.
    # The full colour data is preserved in the original files.
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]

    # Normalise to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def numpy_to_ants(arr: np.ndarray, spacing_mm: float) -> ants.ANTsImage:
    """Wrap a 2D numpy array as an ANTsImage with correct pixel spacing."""
    img = ants.from_numpy(arr)
    img.set_spacing([spacing_mm, spacing_mm])
    return img


def ants_to_numpy(img: ants.ANTsImage) -> np.ndarray:
    return img.numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    tiff_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiff_files:
        sys.exit(f"No TIFF files found in {input_dir}")

    print(f"Found {len(tiff_files)} slices in '{input_dir}'")

    spacing_mm = args.voxel_size_um / 1000.0          # µm → mm
    ref_idx    = len(tiff_files) // 2                  # middle slice as reference

    print(f"Loading slices …")
    raw_slices = [load_slice_as_float32(f) for f in tiff_files]

    # Sanity-check: all slices must have the same shape
    shapes = {a.shape for a in raw_slices}
    if len(shapes) > 1:
        print(f"  WARNING: slices have mixed shapes: {shapes}")
        print("  They will be resampled to the reference shape during registration.")

    reference_np  = raw_slices[ref_idx]
    reference_img = numpy_to_ants(reference_np, spacing_mm)

    print(f"Reference slice: index {ref_idx} ({tiff_files[ref_idx].name})")
    print(f"Transform type : {args.transform}")
    print(f"Metric         : {args.metric}")
    print(f"Voxel size     : {args.voxel_size_um} µm ({spacing_mm:.4f} mm)\n")

    registered = np.zeros(
        (reference_np.shape[0], reference_np.shape[1], len(tiff_files)),
        dtype=np.float32
    )

    for i, (path, raw) in enumerate(zip(tiff_files, raw_slices)):
        if i == ref_idx:
            print(f"  [{i+1:>3}/{len(tiff_files)}] {path.name}  ← reference (no registration)")
            registered[..., i] = reference_np
            continue

        moving_img = numpy_to_ants(raw, spacing_mm)

        result = ants.registration(
            fixed   = reference_img,
            moving  = moving_img,
            type_of_transform = args.transform,
            aff_metric        = args.metric,
            syn_metric        = args.metric,
            reg_iterations    = (100, 70, 40, 20),   # multi-scale pyramid
            verbose           = False,
        )

        warped = ants_to_numpy(result["warpedmovout"])
        registered[..., i] = warped

        print(f"  [{i+1:>3}/{len(tiff_files)}] {path.name}  ✓")

    # Save as NIfTI with correct voxel spacing
    # Axes: (rows, cols, slices)  →  voxel size: (x, y, z)
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    nifti  = nib.Nifti1Image(registered, affine)

    nifti.header.set_xyzt_units(xyz=2)   # 2 = mm
    nifti.header["descrip"] = b"PLI in-plane registered stack"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti, str(output_path))

    print(f"\nRegistered 3D stack saved → {output_path}")
    print(f"Volume shape : {registered.shape}  (rows × cols × slices)")
    print(f"Voxel size   : {spacing_mm:.4f} × {spacing_mm:.4f} × {spacing_mm:.4f} mm")
    print("\nNext step: compute 3D structure tensor on this volume.")


if __name__ == "__main__":
    main()   
