import cv2  # Import OpenCV for image processing
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting (though it's not used in the code)
from sklearn.decomposition import PCA, FastICA  # Import PCA and FastICA from scikit-learn for dimensionality reduction

def rgb_to_luv(image_path):
    # Load the image from the given path
    image = cv2.imread(image_path)
    # Convert from RGB to BGR since OpenCV uses BGR by default
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Convert from BGR to Luv color space
    image_luv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Luv)
    return image_luv

def rgb_to_lchluv(image_path):
    # Convert the RGB image to Luv color space
    luv_image = rgb_to_luv(image_path)
    # Separate L, u, and v channels
    L, u, v = cv2.split(luv_image)
    # Ensure L, u, and v are of the same size and type, converting them to float32
    L, u, v = [channel.astype(np.float32) for channel in [L, u, v]]
    # Calculate C (chroma) from u and v
    C = np.sqrt(u**2 + v**2)
    # Calculate H (hue) from u and v
    H = np.arctan2(v, u) * 180 / np.pi
    H = np.mod(H, 360)  # Ensure H is between 0 and 360 degrees
    return L, C, H

def create_6_layer_array(image_path):
    # Load the original RGB image
    rgb_image = cv2.imread(image_path)
    R, G, B = [rgb_image[:, :, i] for i in range(3)]
    # Obtain L, C, H channels from the rgb_to_lchluv function
    L, C, H = rgb_to_lchluv(image_path)
    # Stack all the channels together to form a 6-layer image
    six_layer_image = np.dstack((R, G, B, L, C, H))
    return six_layer_image

def calculate_correlation_matrix(image_path):
    # Create the 6-layer matrix from the image
    six_layer_image = create_6_layer_array(image_path)
    # Reshape the image so that each band becomes a column
    reshaped_image = six_layer_image.reshape(-1, six_layer_image.shape[2])
    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(reshaped_image, rowvar=False)
    return correlation_matrix

def apply_pca_to_image(image_path, num_components=6):
    # Load the image and create the 6-layer array
    six_layer_image = create_6_layer_array(image_path)
    # Reshape the image into a 2D matrix (num_pixels x 6)
    num_rows, num_cols, num_channels = six_layer_image.shape
    image_2d = six_layer_image.reshape((num_rows * num_cols, num_channels))
    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(image_2d, rowvar=False)
    # Apply PCA
    pca = PCA(n_components=num_components)
    transformed_image = pca.fit_transform(image_2d)
    # Reshape back into a 3D image for visualization or further processing
    transformed_image_3d = transformed_image.reshape((num_rows, num_cols, num_components))
    return correlation_matrix, transformed_image_3d

def apply_ica_to_image(image_path, num_components=6):
    # Load the image and create the 6-layer array
    six_layer_image = create_6_layer_array(image_path)
    # Reshape the image into a 2D matrix (num_pixels x 6)
    num_rows, num_cols, num_channels = six_layer_image.shape
    image_2d = six_layer_image.reshape((num_rows * num_cols, num_channels))
    # Apply ICA
    ica = FastICA(n_components=num_components, random_state=0)
    transformed_image = ica.fit_transform(image_2d)
    # Reshape back into a 3D image for visualization or further processing
    transformed_image_3d =
