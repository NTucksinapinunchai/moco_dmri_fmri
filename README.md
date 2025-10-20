# Deep Learning Approaches for Motion Correction in Spinal Cord Diffusion MRI and Functional MRI
This repository introduces a DenseNet-based slice-wise regressor that estimates rigid in-plane translations (Tx, Ty) for motion correction.

## Overview
1. **Generate motion-affected data** using `augmentation.py`  
   → Simulate slice-wise rigid motion artifacts (only in-plane motion) in dMRI and fMRI data.

2. **Preprocess the dataset** using `preprocessing.py`  
   → Prepare mean dMRI/fMRI volumes, perform spinal cord segmentation and create mask along spinal cord via **Spinal Cord Toolbox (SCT)**.

3. **Prepare datasets for training and validation** using `dataset_preparation.py`  
   → Organize the dataset structure and split into training, validation, (in .pt format) and testing sets.

4. **Train the DenseNet-based deep learning model** using `moco_main.py`  
   → Learn rigid slice-wise translations (Tx, Ty) for motion correction across time.

5. **Evaluate and test model performance** using `test_model.py`  
   → Apply the trained model to new data, correct motion, and export motion-corrected 4D volumes as well as Tx and Ty translation parameter.

## Dependencies
The primary dependencies for this project are:
*   [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/): Required for the `preprocessing.py` script. Please ensure SCT is installed and its command-line tools are available in your PATH.
*   Python 3.9
*   PyTorch Lightning
*   [MONAI](https://github.com/Project-MONAI/MONAI)
*   nibabel
*   scikit-image
*   PyYAML
