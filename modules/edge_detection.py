import cv2
import numpy as np

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
    
    return edges_advanced