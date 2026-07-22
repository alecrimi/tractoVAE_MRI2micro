# Patch snippet for fix_microscopy_spacing.py:
# Replace the hardcoded size[0]->X, size[1]->Y assumption with a lookup
# based on which array axis actually points along which world axis.

import numpy as np
import SimpleITK as sitk

def fix_spacing(in_path, out_path, fixed_x_extent_mm, fixed_y_extent_mm, fixed_z_extent_mm):
    img = sitk.ReadImage(in_path)
    size = np.array(img.GetSize())
    old_spacing = np.array(img.GetSpacing())
    old_origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)

    # direction[:, k] is the world-space direction of array axis k.
    # Find which world axis (0=X,1=Y,2=Z) each array axis is most aligned with.
    world_axis_for_array_axis = np.argmax(np.abs(direction), axis=0)  # len-3 array

    fixed_extents = {0: fixed_x_extent_mm, 1: fixed_y_extent_mm, 2: fixed_z_extent_mm}

    new_spacing = old_spacing.copy()
    for array_axis in range(3):
        world_axis = world_axis_for_array_axis[array_axis]
        target_extent = fixed_extents[world_axis]
        new_spacing[array_axis] = target_extent / size[array_axis]
        print(f"array axis {array_axis} -> world axis {world_axis}: "
              f"{size[array_axis]} vox, old spacing {old_spacing[array_axis]:.4f}, "
              f"new spacing {new_spacing[array_axis]:.4f} "
              f"(target extent {target_extent}mm)")

    center_vox = (size - 1) / 2.0
    old_center_world = old_origin + direction @ (old_spacing * center_vox)
    new_origin = old_center_world - direction @ (new_spacing * center_vox)

    img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(tuple(new_origin))
    sitk.WriteImage(img, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    import sys
    fix_spacing(sys.argv[1], sys.argv[2], 51.2, 4.0, 38.4)