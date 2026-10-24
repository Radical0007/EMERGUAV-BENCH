import cv2
import numpy as np

def apply_blur(image: np.ndarray, severity: str) -> np.ndarray:
    """Applies Gaussian blur to simulate motion blur or out-of-focus."""
    if severity == "high" or severity == "critical":
        ksize = (31, 31)
    else:
        ksize = (15, 15)
    return cv2.GaussianBlur(image, ksize, 0)

def apply_dust_veil(image: np.ndarray, severity: str) -> np.ndarray:
    """Applies an orange/brown tint, reduces contrast, and adds noise to simulate a dust veil."""
    if severity == "high" or severity == "critical":
        alpha = 0.7
    else:
        alpha = 0.4
        
    # Orange/dust color in BGR
    dust_color = (50, 130, 200)
    overlay = np.full_like(image, dust_color, dtype=np.uint8)
    
    # Blend image with dust color
    blended = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
    
    # Generate some random noise
    noise = np.random.normal(0, 15, blended.shape).astype(np.uint8)
    blended = cv2.add(blended, noise)
    
    return blended

def apply_degradation(image_path: str, degradation_label: str, severity: str) -> np.ndarray:
    """Reads an image, applies the specified degradation, and returns the modified image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
        
    if degradation_label == "blur":
        return apply_blur(image, severity)
    elif degradation_label == "dust_veil":
        return apply_dust_veil(image, severity)
    else:
        # Fallback to returning original for unsupported types or "clean_visual"
        return image
