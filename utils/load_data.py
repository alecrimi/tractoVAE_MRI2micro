import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_diffusion_data():
    # Load the diffusion data
    data_path = './Sample1_MRI/Data/data.nii.gz'
    img = nib.load(data_path)
    data = img.get_fdata()
    print(f"Data shape: {data.shape}")

    # Load bvals and bvecs
    bvals = np.loadtxt('./Sample1_MRI/Data/bvals')
    bvecs = np.loadtxt('./Sample1_MRI/Data/bvecs')

    print(f"Number of diffusion directions: {len(bvals)}")
    print(f"B-values range: {bvals.min():.1f} - {bvals.max():.1f}")
    
    return data, bvals, bvecs

def display_middle_slice(data, bvals):
    # Display a middle slice from the b0 image (where bval = 0)
    b0_idx = np.where(bvals < 10)[0][0]  # Find first b0 image
    middle_slice = data.shape[2] // 2

    plt.figure(figsize=(10, 10))
    plt.imshow(data[:, :, middle_slice, b0_idx], cmap='gray')
    plt.axis('off')
    plt.title('Middle slice of b0 image')
    plt.show()

if __name__ == "__main__":
    # Load the data
    data, bvals, bvecs = load_diffusion_data()
    
    # Display the middle slice
    display_middle_slice(data, bvals) 