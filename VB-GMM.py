import cv2  # For image loading and processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization of images and clusters
from sklearn.mixture import BayesianGaussianMixture  # For Bayesian Gaussian Mixture Model clustering


def apply_vbgmm_to_sample(in_image, num_clusters=15, sample_size=100000, verbose=1, max_iter=2000, tol=1e-3):
    # Load and process the mask
    mask = cv2.imread('mask.tif', 0)
    _, binary_mask = cv2.threshold(mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    # Flatten the image and the mask
    reshaped_image = in_image.reshape(-1, in_image.shape[2])
    reshaped_mask = binary_mask.flatten()
    # Filter indices to sample only where the mask is not 0
    available_indices = np.where(reshaped_mask != 0)[0]
    sampled_indices = np.random.choice(available_indices, size=sample_size, replace=False)
    sampled_pixels = reshaped_image[sampled_indices]
    # Apply Variational Bayesian Gaussian Mixture Model (VBGMM) to the sampled pixels
    vbgmm = BayesianGaussianMixture(
        n_components=num_clusters, 
        weight_concentration_prior_type='dirichlet_process', 
        verbose=verbose, max_iter=max_iter, tol=tol
    )
    vbgmm.fit(sampled_pixels)
    # Print model parameters and convergence information
    print("Mixture weights (weights_):", vbgmm.weights_)
    print("Mixture means (means_):", vbgmm.means_)
    print("Covariances (covariances_):", vbgmm.covariances_)
    print("Mixture precisions (precisions_):", vbgmm.precisions_)
    print("Algorithm convergence (converged_):", vbgmm.converged_)
    print("Number of iterations performed (n_iter_):", vbgmm.n_iter_)
    # Count the number of components with significant weights
    threshold = 0.05
    significant_components = [i for i, weight in enumerate(vbgmm.weights_) if weight > threshold]
    print("Significant components:", significant_components)
    print(' ')
    print('Prediction begins')
    # Predict clusters for the entire image
    labels = vbgmm.predict(reshaped_image)
    labels_image = labels.reshape(in_image.shape[:2])
    # Optional: Get probabilities for the entire image
    probabilities = vbgmm.predict_proba(reshaped_image)
    probabilities_image = probabilities.reshape(*in_image.shape[:2], -1)
    log_prob = vbgmm.score_samples(reshaped_image)
    return labels_image, probabilities_image, log_prob

def visualize_probabilities(probabilities):
    # Visualize the probability for each class (for simplicity, we choose one class)
    plt.imshow(probabilities[:, :, 0], cmap='viridis')  # Visualizes the probability of the first class
    plt.title('Class 1 Probabilities')
    plt.axis('off')
    plt.show()
