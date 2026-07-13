import numpy as np
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import set_number_of_points

# Path to your .trk file
trk_path = "path/to/your/tractography.trk"

# Load streamlines
streamlines_obj = load_trk(trk_path, "same")
streamlines = list(streamlines_obj.streamlines)

# Optional: interpolate streamlines to have equal number of points
n_points = 100  # for example, 100 points per streamline
interpolated_streamlines = set_number_of_points(streamlines, n_points)

# Convert to numpy array (N_streamlines x N_points x 3)
interpolated_streamlines = np.array(interpolated_streamlines)

# Save as .npz file
np.savez("interpolated_streamlines.npz", streamlines=interpolated_streamlines)
print(f"Saved {len(interpolated_streamlines)} streamlines with {n_points} points each.")
