import numpy as np
from scipy.stats import spearmanr

# Assuming imgstck is your 3D numpy array with dimensions n x m x number of bands
imgstck = # your image stack here

spearman_correlation_matrix = np.zeros((imgstck.shape[2], imgstck.shape[2]))
flattened_bands = [imgstck[:, :, i].flatten() for i in range(imgstck.shape[2])]

# Calculate Spearman correlation for each pair of bands
for i in range(imgstck.shape[2]):
    for j in range(imgstck.shape[2]):
        corr, _ = spearmanr(flattened_bands[i], flattened_bands[j])
        spearman_correlation_matrix[i, j] = corr

# This will display the Spearman correlation matrix
print(spearman_correlation_matrix)

# Uncomment the following line to calculate the Pearson correlation matrix
# correlation_matrix = np.corrcoef([imgstck[:, :, i].flatten() for i in range(imgstck.shape[2])])
