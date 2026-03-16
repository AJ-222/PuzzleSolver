import cv2
import numpy as np

def resample_contour(contour, n_points=50):
    """Resamples a contour to have exactly n_points using linear interpolation."""
    if len(contour) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
        
    dists = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    
    if total_len == 0:
        return np.zeros((n_points, 2), dtype=np.float32)
        
    target_dists = np.linspace(0, total_len, n_points)
    new_x = np.interp(target_dists, cum_dists, contour[:, 0])
    new_y = np.interp(target_dists, cum_dists, contour[:, 1])
    
    return np.column_stack((new_x, new_y)).astype(np.float32)

def get_side_color(image, contour):
    """Extracts the average CIELAB color of the piece edge."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, thickness=4)
    
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return [0, 0, 0]
        
    pixels_bgr = pixels.reshape(1, -1, 3).astype(np.uint8)
    pixels_lab = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2Lab)
    
    mean_lab = np.mean(pixels_lab[0], axis=0)
    return mean_lab.tolist()

def classify_side(contour, flat_threshold=0.03):
    """Determines if a side is FLAT, OUT (Tab), or IN (Hole)."""
    if len(contour) < 5:
        return "flat", 0.0
        
    start, end = contour[0], contour[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0: 
        return "flat", 0.0

    cross_prod = np.cross(line_vec, contour - start)
    max_dev = np.max(np.abs(cross_prod)) / line_len
    mean_dev = np.mean(cross_prod) / line_len 
    ratio = max_dev / line_len
    
    if ratio < flat_threshold:
        return "flat", ratio
    
    return "curved", mean_dev

def calculate_match_score(side_a, side_b, color_weight=0.1):
    """
    Calculates adjacency score combining Euclidean shape distance and Delta-E color difference.
    Lower score indicates a better match.
    """
    if side_a['type'] == 'flat' and side_b['type'] == 'flat':
        return 0.0 
    if side_a['type'] == side_b['type'] or side_a['type'] == 'flat' or side_b['type'] == 'flat': 
        return 100.0 
    
    shape_a = side_a['shape']
    shape_b_flipped = side_b['shape'].copy()
    shape_b_flipped[:, 1] *= -1 
    
    shape_diff = np.mean(np.linalg.norm(shape_a - shape_b_flipped, axis=1))
    col_diff = np.linalg.norm(np.array(side_a['color']) - np.array(side_b['color']))
    
    return (shape_diff * 1.0) + (col_diff * 0.1 * color_weight)