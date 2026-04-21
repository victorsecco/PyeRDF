import numpy as np


def bin_image(img, factor=2):
    """Bin the image by the given factor."""
    if factor <= 1:
        return img
    h, w = img.shape
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    img = img[:h_trim, :w_trim]
    img = img.reshape(h_trim // factor, factor, w_trim // factor, factor)
    return img.mean(axis=(1, 3))

def normalize_image(img):
    """Normalize a 16-bit image to 0-255 for display."""
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return (img * 255).astype(np.uint8)