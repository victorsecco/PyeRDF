import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from medpy.filter.smoothing import anisotropic_diffusion


class DataLoader:
    """Handles loading of .ser, .png, and .tif files."""

    def __init__(self):
        pass

    def load_ser(self, file_path):
        import hyperspy.api as hs
        """Loads a .ser diffraction series using HyperSpy."""
        data = hs.load(file_path, signal_type='diffraction', lazy=True)
        num_frames = data.data.shape[0]
        return data, num_frames

    def load_png(self, file_path):
        """Loads a .png image as numpy array (grayscale)."""
        img = Image.open(file_path).convert("L")  # grayscale 8-bit
        return np.array(img)

    def load_tif(self, file_path):
        """Loads a .tif or .tiff file, preserving bit depth."""
        img = Image.open(file_path)
        return np.array(img)

class ImageProcessing:
    def __init__(self):
        pass
    def load_images(self, num_images, Binary = 1):
        if not os.path.isdir(self.path):
          img = Image.open(self.path)
          img = np.array(img)
          return img
        else:
          images_list = os.listdir(self.path)
          images_names = [image for image in images_list if (image.lower().endswith(".tif") or image.lower().endswith(".tiff"))]
          images_names.sort()
          images = []
          images_array = []
          for filename in images_names[:num_images]:
              img = Image.open(os.path.join(self.path, filename))
              images.append(img)
              if Binary == 0:
                img[img == 255] = 1
                img= img[:,:,0]
              img = np.array(img)
              images_array.append(img)
          images_array = np.array(images_array)
          return images_array, images_names

    def load_mask(self, mask_path):
      return np.array(Image.open(mask_path))

    def subtract_mask(self, image, mask):
      if image.shape == mask.shape:
        image[mask==255] = 0
        image = ma.masked_equal(image, 0)
        return image
      else:
        raise Exception('A imagem tem'f'{image.shape} e a mascara tem 'f'{mask.shape}')

#Retirar os defeitos fixos das imagens
    def fixed_defects_mask(self, image, microscope):
      if microscope.lower() in ("titan"):
        image[2140:2160, 2030:2070]=0
        image[:,4087:]=0
        image[:,0:5]=0
        image[4051:4053]=0
        #image[3072:]=False
        image = ma.masked_equal(image, 0)
        return image
      else:
        image[:,:7]=0
        image[3072:]=0
        image[2136,1976:2225]=0
        image = ma.masked_equal(image, 0)
        return image

    def remove_border(self, image, border_size):
        return image[border_size:-border_size, border_size:-border_size]
    
    def pad_for_center(self, image, pad_width=512, mode='constant'):
        p = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode=mode)
        return p, pad_width

#Encontrar o centro com a transformada de Hough para usar como chute inicial
class ImageAnalysis:
    def __init__(self):
        pass

    def find_center(self, image,  r, R, threshold, niter=25, kappa=40, gamma=0.1, anisotropic=True):
        if image is None:
            return "Image not loaded properly. Check the image path."


        blur = cv2.normalize(src=image, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Apply anisotropic diffusion
        if anisotropic:
            blur = anisotropic_diffusion(blur, niter=niter, kappa=kappa, gamma=gamma, option=1)
        else:
            blur = cv2.GaussianBlur(blur, (3, 3), 30)

        # Convert back to [0, 255] uint8
        blur = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        final_im = np.where(blur > threshold, 255, 0)
        final_im = final_im.astype(np.uint8)
        
        edges = cv2.Canny(final_im, 0, 255)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=255, param2=10, minRadius= r, maxRadius=R)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            center_x, center_y, r = circles[0]
            return center_x, center_y, r, final_im, blur, edges
        else:
            return final_im, blur, edges
        
    def azimuth_integration_cv2(self, img, center):
      binning = img.shape[0]
      polar_image = cv2.linearPolar(img,(int(center[0]), int(center[1])), binning, cv2.WARP_FILL_OUTLIERS)

      #Assigning a binary mask, the zero values will be left out
      mask = np.where(polar_image > 0, 1, 0).astype(np.uint8)

      #Creating a copy and using the mask to set the values that will be cropped to the max value of the image 
      integrated_img_processed = polar_image.copy()
      integrated_img_processed[mask == 0] = polar_image.max()

      #Here is where the masking really happens, the max values are excluded
      masked_image = ma.masked_equal(integrated_img_processed, polar_image.max())

      #The polar transform gives 4096 image, here I make an interpolation to the real max distance to the border
      original_data = masked_image.mean(axis = 0)
      new_length = int(binning)

      old_indices = np.linspace(0, len(original_data) - 1, num=len(original_data))
      new_indices = np.linspace(0, len(original_data) - 1, num=new_length)

      interp_func = interp1d(old_indices, original_data, kind='linear')

      #Final data containing the azimuthal average interpolated to the desired inverval
      new_data = interp_func(new_indices)

      return new_data, polar_image, masked_image

    def optimize_center(self, image, initial_center_x, initial_center_y, azimuth_ranges, max_iterations):
      center_x, center_y = initial_center_x, initial_center_y

      for _ in range(max_iterations):
          # Find peaks and integrate for the first set
          peaks3 = self.find_and_integrate(image, center_x, center_y, azimuth_ranges[0])
          peaks4 = self.find_and_integrate(image, center_x, center_y, azimuth_ranges[1])

          # Adjust center based on peaks
          if peaks3[1] < peaks4[1]:
              center_x, center_y = center_x, center_y - 1
          else:
              center_x, center_y = center_x, center_y + 1

          # Check convergence condition
          if abs(peaks3[1] - peaks4[1]) <= 3:
              break

      for _ in range(max_iterations):
          # Find peaks and integrate for the second set
          peaks1 = self.find_and_integrate(image, center_x, center_y, azimuth_ranges[2])
          peaks2 = self.find_and_integrate(image, center_x, center_y, azimuth_ranges[3])

          # Adjust center based on peaks
          if peaks1[0] < peaks2[1]:
              center_x, center_y = center_x - 1, center_y
          else:
              center_x, center_y = center_x + 1, center_y

          # Check convergence condition
          if abs(peaks1[1] - peaks2[1]) <= 3:
              break

      return center_x, center_y
    
    def MSE(self, x, y):
        return math.sqrt(1/len(x) * sum((x - y)**2))

 
def pad_image_for_hough(image, pad_width=512, mode='constant'):
    p = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode=mode)
    return p, pad_width

def bin_2d_by_2(arr):
    h, w = arr.shape
    return arr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

def apply_timepix_cross(img):
    img = img.copy()
    img[255, :] = 0
    img[:, 255] = 0
    return img

def apply_us4000_mask(img):
    img = img.copy()
    img[1915:,1002:1022] = 0
    img[:120,1013:1030] = 0
    return img

def apply_beamstop_mask(img, mask_path):
    mask = tifffile.imread(mask_path)
    if mask.shape == img.shape:
        m = mask
    elif mask.shape[0] == 4096 and img.shape[0] == 2048:
        m = bin_2d_by_2(mask)
    else:
        raise ValueError("Mask and image shapes incompatible.")
    out = img.copy()
    out[m == 255] = -10
    return out

def find_center_dispatch(img, padded, offset, analysis, manual, thresh = 100, c=None):
    if manual:
        cx, cy = c
        cx, cy = 16.838044892381255, 7.780054061182113
        r = thre = blur = edges = None
        return cx, cy, r, thre, blur, edges
    if padded:
        cx, cy, r, thre, blur, edges = analysis.find_center(img, r=1, R=5000, threshold=thresh, niter=20, kappa=100, anisotropic_diffusion=False)
    else:
        cx, cy, r, thre, blur, edges = analysis.find_center(img, r=1, R=5000, threshold=thresh, niter=20, kappa=100, anisotropic_diffusion=False)
    return cx, cy, r, thre, blur, edges


def hot_pixel_filter(img, thr=100, ksize=3):
    src = img.astype(np.float32, copy=False)
    med = cv2.medianBlur(src, ksize)
    mask = (src - med) > thr
    out = src.copy()
    out[mask] = med[mask]
    return out.astype(img.dtype)


def load_empad_data(file_path, num_images):
    """
    Load EMPAD data from a given file path, processing a specified number of images.
    
    Parameters:
    - file_path: str, the path to the EMPAD raw data file.
    - num_images: int, the number of images in the data file.
    
    Returns:
    - data: numpy.ndarray, the processed data array with shape (num_images, 128, 128),
            or None if there's a size mismatch or other loading issue.
    """
    pattern_size = 130 * 128 * 4  # 4 bytes per pixel, accounting for 2 metadata rows per image
    filesize = os.path.getsize(file_path)
    expected_size = num_images * pattern_size

    # Check if the file size matches the expected size based on the number of images
    if filesize != expected_size:
        print("Warning: File size does not match the expected size based on the number of images.")
        print(f"Expected {expected_size}, but got {filesize}. Please check the file path and number of images. Probably is {filesize/(130*256)}")
        return None
    
    with open(file_path, "rb") as fid:
        data = np.fromfile(fid, dtype=np.float32)
        if len(data) == num_images * 130 * 128:
            # Reshape and crop the data to remove metadata rows
            data = data.reshape(num_images, 130, 128)[:, :128, :]
            return data
        else:
            print("Data size does not match the expected pattern size. Please check the file or pattern_size calculation.")
            return None
        
def highest_distance_to_border(point, image_width, image_height):
    # Unpack the point coordinates
    x, y = point

    # Define the corners of the image
    corners = [(0, 0), (0, image_height), (image_width, 0), (image_width, image_height)]

    # Calculate the distance from the point to each corner
    distances = [math.sqrt((x - corner_x)**2 + (y - corner_y)**2) for corner_x, corner_y in corners]

    # Find the highest distance
    max_distance = max(distances)

    return max_distance

def peak_calibration(pixel_positions, standard = 'gold', peaks = None):
    ds = []
    if standard =='gold':  
      peaks = [2.354977, 2.039470, 1.442123, 1.229847, 1.177489, 1.019735, 0.935773, 0.912079]

    for i in range(len(pixel_positions)):
        ds.append(1/(pixel_positions[i]*peaks[i])*2*math.pi)
        
    ds = np.array(ds).mean()
    print('The calculated calibration factor ds: 'f'{ds}')
    return ds