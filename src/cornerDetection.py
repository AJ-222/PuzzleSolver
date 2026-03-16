import os
import cv2
import math
import numpy as np
import torch
import segmentation_models_pytorch as smp
from skimage.feature import peak_local_max

def get_corner_model(device, encoder_name="efficientnet-b0"):
    """Loads the U-Net regression model for 1-channel heatmap prediction."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1, 
    ).to(device)
    return model

def create_heatmaps(keypoints, img_size, hm_size):
    """Creates a master heatmap with 4 Gaussian peaks for training."""
    heatmaps = np.zeros((1, hm_size, hm_size), dtype=np.float32)
    scale_x, scale_y = hm_size / img_size[1], hm_size / img_size[0] 
    heatmap = np.zeros((hm_size, hm_size), dtype=np.float32)

    for pt in keypoints:
        x = max(0, min(int(pt[0] * scale_x), hm_size - 1))
        y = max(0, min(int(pt[1] * scale_y), hm_size - 1))
        heatmap[y, x] = 1.0
        
    g_heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3, sigmaY=3)
    if np.max(g_heatmap) > 0:
        g_heatmap /= np.max(g_heatmap)
        
    heatmaps[0] = g_heatmap 
    return heatmaps

def extract_corners_from_heatmap(heatmap, mask, orig_size, hm_size):
    """
    Extracts the top 4 corners from a predicted heatmap using mask filtering,
    centroid distance, and angle sorting (TL, TR, BR, BL).
    """
    orig_h, orig_w = orig_size
    scale_x, scale_y = orig_w / hm_size, orig_h / hm_size
    
    # Mask Filtering
    resized_mask = cv2.resize(mask, (hm_size, hm_size), interpolation=cv2.INTER_NEAREST)
    resized_mask = (resized_mask > 128).astype(np.float32) 
    filtered_hm = heatmap * resized_mask 
    
    # Centroid Calculation
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return [[0,0], [0,0], [0,0], [0,0]]
    cx_hm = (M["m10"] / M["m00"]) / scale_x
    cy_hm = (M["m01"] / M["m00"]) / scale_y
    
    # Peak Extraction
    candidate_coords = peak_local_max(filtered_hm, min_distance=10, threshold_rel=0.2)
    if len(candidate_coords) < 4:
        flat = filtered_hm.flatten()
        top4_indices = np.argpartition(flat, -4)[-4:]
        candidate_coords = [np.unravel_index(idx, filtered_hm.shape) for idx in top4_indices]
    
    # Filter by distance to centroid (take the 4 farthest)
    distances = []
    for (y_hm, x_hm) in candidate_coords:
        dist = math.sqrt((x_hm - cx_hm)**2 + (y_hm - cy_hm)**2)
        distances.append((dist, (x_hm, y_hm)))
    
    distances.sort(key=lambda x: x[0], reverse=True)
    final_4_peaks_hm = [pt for dist, pt in distances[:4]]
    
    # Robust Sorting (TL, TR, BR, BL)
    final_4_peaks_hm.sort(key=lambda p: math.atan2(p[1] - cy_hm, p[0] - cx_hm))
    final_4_peaks_hm = final_4_peaks_hm[-1:] + final_4_peaks_hm[:-1]
    
    # Scale back to original resolution
    return [[int(x * scale_x), int(y * scale_y)] for x, y in final_4_peaks_hm]