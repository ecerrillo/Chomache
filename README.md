# Chomache
VB-GMM Python scripts for image processing in rock art.

These functions were created to process rock art images, statistically separating different color tones. The functions on the scripts can be considered part of an evolving project. The functions are fully operational, but they may need to be adjusted depending on the input data.

The basic functions to develop the workflow are found in the scripts. conversion_functions.py transforms to the LCHuv color space, in addition to reducing dimensionality through Principal Components (PCA) and Independent Components (ICA).

VB-GMM.py applies a Gaussian mixture model (variational Bayesian) to classify pigments, based on samples to which masks can be added. 
statistics.py performs a Spearman correlation analysis.

The code is related to the following paper: Cerrillo-Cuenca, E., Sepúlveda, M., Cabello, G., Bastías, F. (2024). Color-based discrimination of color hues in rock paintings through Gaussian mixture models: a case study from Chomache site (Chile). Heritage Science 12, 169. https://doi.org/10.1186/s40494-024-01239-9
