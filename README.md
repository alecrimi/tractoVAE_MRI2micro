# Generative AI framework from MRI to Microscopy tractography

This framework is able to to generate microscopy tractography from MRI tractography since it is expected that the microscopy
tractography can contain more relevant details as it is based on real cell imaging and not MRI water disctribution approximation.

These are the files that need to be executed:

## MRI
0. Files shuold have eddy-current interference removed e.g. with FSL

1. The file to generate the tractography MRI is **generate_streamlines_MRI.py** (output a trk file)

## Microscopy
2. The microscopy data contain a lot of noise and stitching artefact which can be removed with the script
**preprocessing_pli.py**  (output a series of TIF files)

3. The microscopy data is given as a individual slices which has to be stacked and registered each other. This is done using the script 
**register_pli_stack.py** (output a nii.gz file)

4. Compute the microscopy tractography with the cleaned self-registered and registered microscopy data with the script
   **run_microscopy_tractography.py** (output a trk file), if you use a PLI use  **run_PLI_tractography.py** because PLI tractography is based on coherence of  
   Polarized light imaging rather than structure tenso

5. In case the tractography is squeezing the z-axis, rescale it with **rescale_microscopy_trk.py**

# VAE translation

6. Train and use an  VAE going back and forth one latent space to the other.

![pipeline](pipeline.png)


![im1](im1.png)


4b. Further register the self registered stack microscopy to the original MRI volume with **PLI_MRI_Registration.py**  This has some bugs in case of very large microscopy images.
