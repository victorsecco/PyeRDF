import numpy as np
import cv2
import math
import numpy.ma as ma
from scipy.interpolate import interp1d
from medpy.filter.smoothing import anisotropic_diffusion

class ImageProcessing:
    def __init__(self, img = None):
        if len(img.shape) > 2:
            img = self.sum_stack(img)
            print(img.shape)
        self.img = img
        pass

    def sum_stack(self, img):
        return img.sum(axis = 0)

    def subtract_mask(self, mask):
      if self.img.shape == mask.shape:
        self.img[mask==255] = 0
        self.img = ma.masked_equal(self.img, 0)
        return self.img
      else:
        raise Exception('A imagem tem'f'{self.img.shape} e a mascara tem 'f'{mask.shape}')

    def fixed_defects_mask(self, microscope):
      if microscope.lower() in ("titan"):
        self.img[2140:2160, 2030:2070]=0
        self.img[:,4087:]=0
        self.img[:,0:5]=0
        self.img[4051:4053]=0
        self.img = ma.masked_equal(self.img, 0)
        return self.img
      else:
        self.img[:,:7]=0
        self.img[3072:]=0
        self.img[2136,1976:2225]=0
        self.img = ma.masked_equal(self.img, 0)
        return self.img

    def remove_border(self, border_size):
        return self.img[border_size:-border_size, border_size:-border_size]
    
    def pad_for_center(self, pad_width=512, mode='constant', axis=None, side=None):
        h, w = pad_width, pad_width

        if axis is None and side is None:
            pad = ((h, h), (w, w))

        elif axis == 0:  # y axis (rows)
            if side == 'left':
                pad = ((h, 0), (0, 0))
            elif side == 'right':
                pad = ((0, h), (0, 0))
            else:  # both
                pad = ((h, h), (0, 0))

        elif axis == 1:  # x axis (columns)
            if side == 'left':
                pad = ((0, 0), (h, 0))
            elif side == 'right':
                pad = ((0, 0), (0, h))
            else:  # both
                pad = ((0, 0), (h, h))

        else:
            raise ValueError("axis must be 0, 1, or None")

        p = np.pad(self.img, pad, mode=mode)
        return p, pad

    def bin_to_512(self):
        h, w = self.img.shape
        if (h, w) != (512, 512):
            factor_h = h // 512
            factor_w = w // 512
            factor = min(factor_h, factor_w)
            return self.img[:factor*512, :factor*512].reshape(512, factor, 512, factor).mean((1, 3)), factor
        else:
            return self.img
        
    def pad_image_for_hough(self, image, pad_width=512, mode='constant'):
        p = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode=mode)
        return p, pad_width

    def bin_image(self, bin = 2):
        h, w = self.img.shape
        return self.img.reshape(h // bin, bin, w // bin, bin).mean(axis=(1, 3))

    def apply_timepix_cross(self):
        self.img[255, :] = 0
        self.img[:, 255] = 0
        return self.img

    def apply_us4000_mask(self):
        self.img = self.img.copy()
        self.img[1915:,1002:1022] = 0
        self.img[:120,1013:1030] = 0
        return self.img

    def apply_beamstop_mask(self, mask):
        if isinstance(mask, np.ndarray):
            H, W = self.img.shape      # 2048, 1024
            h, w = mask.shape  # 2048, 2048

            if (h, w) != (H, W):
                if h != H or w < W:
                    raise ValueError("Mask must match image height and have width ≥ image width.")

                # keep columns 1024:2048
                mask = mask[:, w-W:w]   # becomes [:, 1024:2048]

            self.img[mask == 255] = 0
        return self.img

    def hot_pixel_filter(self, thr=100, ksize=3):
        src = self.img.astype(np.float32, copy=False)
        med = cv2.medianBlur(src, ksize)
        mask = (src - med) > thr
        out = src.copy()
        out[mask] = med[mask]
        return out.astype(self.img.dtype)
    
    def hot_pixel_filter_sigma(self, ksize=5, sigma=3):
        src = self.img.astype(np.float32, copy=False)

        from scipy.ndimage import uniform_filter

        mean = uniform_filter(src, size=ksize, mode='reflect')
        mean_sq = uniform_filter(src * src, size=ksize, mode='reflect')
        var = mean_sq - mean * mean
        std = np.sqrt(np.maximum(var, 0))

        mask = src > (mean + sigma * std)

        out = src.copy()
        out[mask] = mean[mask]
        return out.astype(self.img.dtype)

    
    def log_intensity(self, img=None):
        if img is None:
            img = self.img

        return np.log(img)

    def sqrt_intensity(self, img=None):
        if img is None:
            img = self.img

        return np.sqrt(img)

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
      mask_or = np.where(img > 0, 1, 0).astype(np.uint8)

      #Creating a copy and using the mask to set the values that will be cropped to the max value of the image 
      integrated_img_processed = polar_image.copy()
      integrated_img_processed[mask == 0] = polar_image.max()

      #Here is where the masking really happens, the max values are excluded
      masked_image = ma.masked_equal(integrated_img_processed, polar_image.max())

      masked_image_or = img.copy()
      masked_image_or[mask_or == 0] = img.max()
      masked_image_or = ma.masked_equal(masked_image_or, img.max())

      #The polar transform gives 4096 image, here I make an interpolation to the real max distance to the border
      original_data = masked_image.mean(axis = 0)
      new_length = int(binning)

      old_indices = np.linspace(0, len(original_data) - 1, num=len(original_data))
      new_indices = np.linspace(0, len(original_data) - 1, num=new_length)

      interp_func = interp1d(old_indices, original_data, kind='linear')

      #Final data containing the azimuthal average interpolated to the desired inverval
      new_data = interp_func(new_indices)

      return new_data, masked_image_or, masked_image

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
    dq = []
    if standard =='gold':  
      peaks = [2.354977, 2.039470, 1.442123, 1.229847, 1.177489, 1.019735, 0.935773, 0.912079]

    for i in range(len(pixel_positions)):
        dq.append(1/(pixel_positions[i]*peaks[i])*2*math.pi)
        
    dq = np.array(dq).mean()
    print('The calculated calibration factor dq: 'f'{dq}')
    return dq