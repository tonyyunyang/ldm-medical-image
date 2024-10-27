import cv2
import numpy as np
from scipy import ndimage

def scale_data(data):
    min_val = data.min()
    max_val = data.max()
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8)
    
    scaled = (data - min_val) * (255.0 / (max_val - min_val))

    normalized = (scaled / 127.5) - 1

    return normalized

def detect_edges(data):
    # Normalize the data
    data_normalized = ((data - data.min()) * (255.0 / (data.max() - data.min()))).astype(np.uint8)

    # 1. Create a mask for dark regions (adjust threshold as needed)
    dark_threshold = 10  # Adjust this value to determine what's considered "dark"
    
    # 2. Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(data_normalized, d=11, sigmaColor=100, sigmaSpace=100)
    
    # 3. Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 4. Create mask after enhancement
    mask = (enhanced > dark_threshold).astype(np.uint8)
    
    # 5. Apply Canny with automatic threshold calculation
    enhanced_flat = enhanced.flatten()
    mask_flat = mask.flatten()
    median = np.median(enhanced_flat[mask_flat > 0])
    
    sigma = 0.38
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges_advanced = cv2.Canny(enhanced, lower, upper)
    
    # 6. Apply mask to remove edges in dark regions
    edges_advanced = edges_advanced * mask

    edges_advanced = edges_advanced / 255
    
    return edges_advanced


def scale_semantic_map(data: np.ndarray, target_resolution: tuple) -> np.ndarray:
    # Quantize to integers
    quantized_data = np.round(data).astype(np.int32)
    
    # Check for negative values
    if np.any(quantized_data < 0):
        raise ValueError("Quantized data contains negative values")
    
    # Calculate zoom factors
    if len(data.shape) == 3:
            zoom_factors = (target_resolution[0] / data.shape[0],
                        target_resolution[1] / data.shape[1],
                        1)  # Don't resize the channel dimension
    elif len(data.shape) == 2:
        zoom_factors = (target_resolution[0] / data.shape[0],
                    target_resolution[1] / data.shape[1])
    else:
        raise ValueError(f"Unexpected input dimensions: {len(data.shape)}. Expected 2 or 3.")

    
    # Resize using zoom and round to integers
    resized_data = ndimage.zoom(quantized_data, zoom_factors, order=0)  # order=0 for nearest neighbor
    resized_data = np.round(resized_data).astype(np.uint8)

    # Check for minimum number of unique values
    unique_values = np.unique(resized_data)
    if len(unique_values) > 4:
        raise ValueError(f"After quantization, data contains {len(unique_values)} unique values. Expected no more than 4.")
    
    return resized_data