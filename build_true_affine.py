#!/usr/bin/env python3
"""
build_true_affine.py (fixed)

Changes from the original:
  1. Explicitly stamps sform_code=1 and qform_code=1 on the saved header.
     Without this, some viewers (ITK-SNAP included) can silently ignore
     your custom affine and fall back to a default axial/identity
     interpretation if qform_code ends up 0 ("unknown") -- which is what
     was causing both volumes to appear in the Axial pane regardless of
     the axcodes you passed.
  2. Prints a verification block after building the affine, confirming
     nib.aff2axcodes(new_affine) round-trips to exactly the axcodes you
     supplied. If it doesn't match, the file is wrong before you even
     open it in ITK-SNAP.
  3. Adds an optional --spacing override so you can supply the TRUE
     physical spacing for one or more axes instead of trusting whatever
     is in the header. This matters a lot for sparse stacks (e.g. a
     13-section microscopy stack) where the header's zoom for the
     slice axis often reflects unrelated metadata, not the real
     distance between sampled sections.

IMPORTANT LIMITATION (unchanged from original, but worth restating):
  This script only encodes 90-degree axis permutations and mirror flips
  (R<->L, A<->P, S<->I) via the axcodes letters. It CANNOT correct a true
  in-plane rotation (e.g. sections cut at a real angle to the anatomical
  planes). If your microscopy still looks rotated -- not just mirrored
  or on the wrong pane -- after you've got the axcodes right, that is a
  genuine rigid-body rotation problem and needs actual registration
  (e.g. `antsRegistration --transform Rigid` or `flirt -dof 6`) against
  the MRI, not more axcodes fiddling.

AXIS CODE STRING FORMAT: see original docstring -- unchanged.

Usage:
    python3 build_true_affine.py <input.nii.gz> <output.nii.gz> <AXCODES> \\
        [--spacing AX0,AX1,AX2]

    --spacing lets you override some or all of the header's voxel
    spacings with known-correct values. Use "None" for any axis you
    still want to trust from the header, e.g.:

        --spacing None,None,1.2

    to override only axis2 (the slice axis) with a true 1.2mm spacing,
    keeping the header's axis0/axis1 in-plane spacing.

Example:
    python3 build_true_affine.py registered_stack_fixed.nii.gz \\
                                  registered_stack_fixed_trueorient.nii.gz \\
                                  SLP --spacing None,None,1.2
"""
import argparse
import sys
import numpy as np
import nibabel as nib

LETTER_TO_CANONICAL_AXIS = {
    'R': 0, 'L': 0,
    'A': 1, 'P': 1,
    'S': 2, 'I': 2,
}
LETTER_TO_SIGN = {
    'R': +1, 'L': -1,
    'A': +1, 'P': -1,
    'S': +1, 'I': -1,
}


def build_affine(shape, spacing, axcodes):
    assert len(axcodes) == 3
    # NOTE: must start from zeros, NOT np.eye(4). If any letter's
    # canonical_axis differs from its array_axis (i.e. axcodes encodes a
    # permutation, not just in-place flips), np.eye(4)'s leftover
    # diagonal 1.0s never get overwritten and contaminate the matrix --
    # this was the root cause of the "stretched image, wrong viewer
    # pane" bug. Confirmed via header diagnostic: for axcodes="LSP",
    # np.eye(4) left affine[1,1]=1.0 and affine[2,2]=1.0 sitting
    # alongside the real 0.4mm spacing values placed at [2,1] and [1,2].
    affine = np.zeros((4, 4))
    affine[3, 3] = 1.0
    for array_axis, letter in enumerate(axcodes.upper()):
        canonical_axis = LETTER_TO_CANONICAL_AXIS[letter]
        sign = LETTER_TO_SIGN[letter]
        affine[canonical_axis, array_axis] = sign * spacing[array_axis]
    center_vox = (np.array(shape) - 1) / 2.0
    center_mm = affine[:3, :3] @ center_vox
    affine[:3, 3] = -center_mm
    return affine


def parse_spacing_override(s, header_spacing):
    if s is None:
        return header_spacing.copy()
    parts = s.split(',')
    if len(parts) != 3:
        raise ValueError("--spacing must have exactly 3 comma-separated values")
    out = header_spacing.copy()
    for i, p in enumerate(parts):
        p = p.strip()
        if p.lower() != 'none':
            out[i] = float(p)
    return out


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('in_path')
    ap.add_argument('out_path')
    ap.add_argument('axcodes')
    ap.add_argument('--spacing', default=None,
                     help='Override header spacing, e.g. "None,None,1.2"')
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    args = ap.parse_args()

    img = nib.load(args.in_path)
    header_spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    shape = img.shape[:3]

    spacing = parse_spacing_override(args.spacing, header_spacing)
    if not np.allclose(spacing, header_spacing):
        print(f"NOTE: overriding header spacing {header_spacing} -> {spacing}")

    axcodes = args.axcodes
    new_affine = build_affine(shape, spacing, axcodes)

    new_img = nib.Nifti1Image(img.get_fdata(caching='unchanged'), new_affine)
    new_img.header.set_zooms(spacing)
    # Explicitly stamp both forms as "aligned" -- this is the fix for the
    # "everything shows up as Axial" symptom.
    new_img.set_sform(new_affine, code=1)
    new_img.set_qform(new_affine, code=1)

    nib.save(new_img, args.out_path)

    # --- Verification block ---
    check = nib.load(args.out_path)
    round_trip = ''.join(nib.aff2axcodes(check.affine))
    print(f"Input shape: {shape}, spacing used: {spacing}")
    print(f"Axis codes applied: {axcodes.upper()}")
    print(f"New affine:\n{new_affine}")
    print(f"Resulting physical extent (mm): "
          f"{np.abs(new_affine[:3, :3]) @ (np.array(shape) - 1)}")
    print(f"sform_code={check.header['sform_code']} "
          f"qform_code={check.header['qform_code']}")
    print(f"Round-trip aff2axcodes on saved file: {round_trip}  "
          f"{'OK' if round_trip == axcodes.upper() else '*** MISMATCH -- something is wrong ***'}")
    print(f"Saved to: {args.out_path}")
    print("\n>>> Open in ITK-SNAP: does it land in the Coronal pane now?")
    print(">>> If still rotated (not just mirrored), that is a true rigid")
    print(">>> rotation that axcodes cannot fix -- you need registration,")
    print(">>> not another letter permutation.")


if __name__ == "__main__":
    main()