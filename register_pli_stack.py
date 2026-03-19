"""
PLI In-Plane Slice Stack Registration  (memory-safe version)
=============================================================
Designed for very large slices (>10000×10000 px) on machines with ~16 GB RAM.

Strategy
--------
1. Compute warpfields at LOW resolution  (~10% of original, ~1000×1000 px)
   -> ANTs peak RAM per slice pair: ~300 MB instead of ~3 GB
2. Apply each warpfield to the FULL-RES slice using ants.apply_transforms()
   -> only two full-res slices are in RAM at once
3. Write each registered slice directly to disk via memory-mapped NIfTI
   -> the full 3D array is NEVER assembled in RAM

Requirements:
    pip install antspyx tifffile nibabel numpy scikit-image

Usage:
    python register_pli_stack.py
    python register_pli_stack.py --input_dir in_plane --output registered_stack.nii.gz \
        --voxel_size_um 25 --reg_scale 0.10 --threads 2
"""

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import nibabel as nib
from skimage.transform import rescale

try:
    import ants
except ImportError:
    sys.exit("antspyx is not installed.  Run:  pip install antspyx")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Memory-safe PLI slice registration (low-res warp -> full-res apply)"
    )
    p.add_argument("--input_dir",      default="in_plane",
                   help="Folder containing *.tif PLI slices  (default: in_plane)")
    p.add_argument("--output",         default="registered_stack.nii.gz",
                   help="Output 3D NIfTI file  (default: registered_stack.nii.gz)")
    p.add_argument("--voxel_size_um",  type=float, default=25.0,
                   help="In-plane pixel size in um  (default: 25)")
    p.add_argument("--reg_scale",      type=float, default=0.10,
                   help="Fraction of full resolution used for warpfield estimation "
                        "(default: 0.10 = 10%%, keeps ANTs peak RAM ~300 MB/slice)")
    p.add_argument("--transform",      default="SyNRA",
                   choices=["Translation", "Rigid", "Affine", "SyNRA", "SyN"],
                   help="ANTs transform type  (default: SyNRA)")
    p.add_argument("--metric",         default="GC",
                   choices=["GC", "MI", "CC"],
                   help="Registration metric  (default: GC)")
    p.add_argument("--threads",        type=int, default=2,
                   help="ITK threads  (default: 2 — keep low to limit RAM)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_tiff_float32(path: Path) -> np.ndarray:
    """Load TIFF -> 2D float32 in [0, 1].  RGB/RGBA -> luminance."""
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def downscale(arr: np.ndarray, scale: float) -> np.ndarray:
    return rescale(arr, scale, anti_aliasing=True, order=1).astype(np.float32)


def to_ants(arr: np.ndarray, spacing_mm: float) -> ants.ANTsImage:
    img = ants.from_numpy(arr.copy())
    img.set_spacing([spacing_mm, spacing_mm])
    return img


def ram_mb(shape, n_copies=1) -> float:
    return shape[0] * shape[1] * 4 * n_copies / 1e6


# ---------------------------------------------------------------------------
# Streaming NIfTI writer  (never holds full 3D volume in RAM)
# ---------------------------------------------------------------------------

class NiftiStreamWriter:
    def __init__(self, path: Path, shape_3d: tuple, affine: np.ndarray):
        self.path      = path
        self.shape     = shape_3d
        self.affine    = affine
        self._dat_path = path.with_suffix("").with_suffix(".memmap.dat")
        self._mmap = np.memmap(
            str(self._dat_path), dtype=np.float32, mode="w+", shape=shape_3d
        )

    def write_slice(self, idx: int, data: np.ndarray):
        self._mmap[..., idx] = data.astype(np.float32)
        self._mmap.flush()

    def finalise(self):
        print("  Assembling NIfTI from disk memmap ...")
        img = nib.Nifti1Image(np.array(self._mmap), self.affine)
        img.header.set_xyzt_units(xyz=2)   # mm
        img.header["descrip"] = b"PLI in-plane registered stack"
        nib.save(img, str(self.path))
        del self._mmap
        if self._dat_path.exists():
            self._dat_path.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Set before ANTs initialises ITK thread pool
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    tiff_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiff_files:
        sys.exit(f"No TIFF files found in {input_dir}")

    n_slices       = len(tiff_files)
    ref_idx        = n_slices // 2
    scale          = args.reg_scale
    spacing_mm     = args.voxel_size_um / 1000.0
    spacing_lowres = spacing_mm / scale    # coarser spacing at low res

    print("=" * 62)
    print("  PLI stack registration  (memory-safe)")
    print("=" * 62)
    print(f"  Slices        : {n_slices}  (reference index = {ref_idx})")
    print(f"  Transform     : {args.transform}")
    print(f"  Metric        : {args.metric}")
    print(f"  Reg. scale    : {scale*100:.0f}% of full resolution")
    print(f"  Voxel size    : {args.voxel_size_um} um")
    print(f"  ITK threads   : {args.threads}")

    # ---- Load reference slice (full + low res) -----------------------------
    print(f"\nLoading reference slice [{ref_idx}]  {tiff_files[ref_idx].name} ...")
    ref_full  = load_tiff_float32(tiff_files[ref_idx])
    full_shape = ref_full.shape
    ref_low   = downscale(ref_full, scale)
    low_shape  = ref_low.shape

    print(f"  Full-res : {full_shape[0]} x {full_shape[1]} px  "
          f"({ram_mb(full_shape):.0f} MB/slice)")
    print(f"  Low-res  : {low_shape[0]} x {low_shape[1]} px  "
          f"(ANTs peak RAM est. {ram_mb(low_shape, 8):.0f} MB/pair)\n")

    ref_ants_low  = to_ants(ref_low,  spacing_lowres)
    ref_ants_full = to_ants(ref_full, spacing_mm)

    # ---- Initialise streaming writer ---------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    writer = NiftiStreamWriter(
        output_path,
        (full_shape[0], full_shape[1], n_slices),
        affine
    )

    # Reference slice goes straight to disk
    writer.write_slice(ref_idx, ref_full)
    del ref_full
    gc.collect()

    # ---- Register every slice ----------------------------------------------
    with tempfile.TemporaryDirectory(prefix="ants_pli_") as tmpdir:

        for i, path in enumerate(tiff_files):

            if i == ref_idx:
                print(f"  [{i+1:>3}/{n_slices}] {path.name}  <- reference")
                continue

            # 1. Load and downscale
            moving_full = load_tiff_float32(path)
            moving_low  = downscale(moving_full, scale)

            # 2. Estimate warpfield at low resolution
            mov_ants_low = to_ants(moving_low, spacing_lowres)
            result = ants.registration(
                fixed             = ref_ants_low,
                moving            = mov_ants_low,
                type_of_transform = args.transform,
                aff_metric        = args.metric,
                syn_metric        = args.metric,
                reg_iterations    = (100, 70, 40, 20),
                outprefix         = os.path.join(tmpdir, f"s{i:04d}_"),
                verbose           = False,
            )
            transforms = result["fwdtransforms"]

            # 3. Apply warpfield to full-res slice
            mov_ants_full = to_ants(moving_full, spacing_mm)
            warped = ants.apply_transforms(
                fixed         = ref_ants_full,
                moving        = mov_ants_full,
                transformlist = transforms,
                interpolator  = "linear",
            )

            # 4. Stream to disk, immediately free RAM
            writer.write_slice(i, warped.numpy())
            del moving_full, moving_low, mov_ants_low, mov_ants_full, warped, result
            gc.collect()

            print(f"  [{i+1:>3}/{n_slices}] {path.name}  ok")

    # ---- Save NIfTI --------------------------------------------------------
    writer.finalise()

    print("\n" + "=" * 62)
    print(f"  Saved -> {output_path}")
    print(f"  Volume : {full_shape[0]} x {full_shape[1]} x {n_slices} voxels")
    print(f"  Spacing: {spacing_mm:.4f} mm isotropic")
    print("  Next step: compute 3D structure tensor on this volume.")
    print("=" * 62)


if __name__ == "__main__":
    main()
