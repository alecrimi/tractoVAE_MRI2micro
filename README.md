# Generative AI framework from MRI to Microscopy tractography

This framework is able to to generate microscopy tractography from MRI tractography since it is expected that the microscopy
tractography can contain more relevant details as it is based on real cell imaging and not MRI water disctribution approximation.

These are the files that need to be executed:

## MRI
0. Files shuold have eddy-current interference removed e.g. with FSL

1. The file to generate the tractography MRI is  generate_streamlines_MRI.py (output a trk file)

## Microscopy
2. The microscopy data contain a lot of noise and stitching artefact which can be removed with the script
preprocessing_pli.py  (output a series of TIF files)

3. The microscopy data is given as a individual slices which has to be stacked and registered each other. This is done using the script 
register_pli_stack.py (output a nii.gz file)

4. Further register the self registered stack microscopy to the original MRI volume with ... (output a nii.gz file)

5. Compute the microscopy tractography with the cleaned self-registered and registered microscopy data with the script
   (output a trk file)
 
# LSTM/VAE translation




1. generate_streamlines.py  script is to generate MRI tractography using DiPy
be aware that the bvecs need to be negated: bvecs[:, 1] = -bvecs[:, 1]

![im1](im1.png)
