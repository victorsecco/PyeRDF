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
    def __init__(self, path):
        self.path = path

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

#Função teste para inverter as imagens a partir do centro
    def invert_images(self, images, center, beamstopper = "right"):
      for i in range(len(images)):
        if beamstopper == "left":
          images[i] = images[i][:,:-1]
        left_side = images[i][:,:center]
        mirrored_half = np.flip(left_side, axis=1)
        mirrored_image = np.concatenate((left_side, mirrored_half), axis=1)
        mirrored_image = mirrored_image[:,106:4202]
        cv2.imwrite(path + '/Sem beamstopper/' + names[i][:-4] + '.jpg', mirrored_image)
      return None

    #Função para alinhar as imagens e somá-las
    def stack_translate(self, stack):
        out_previous = StackReg(StackReg.TRANSLATION).register_transform_stack(stack, reference='previous')
        return sum(out_previous)

    def stack_rotate(self, stack):
        out_previous = StackReg(StackReg.RIGID_BODY).register_transform_stack(stack, reference='previous')
        return sum(out_previous)

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

    def save_iq(self, iq, name):
        if os.path.isfile(self.path):
    # Get the directory part of the file path
          directory = os.path.dirname(self.path)
          iq = pd.DataFrame(np.transpose(np.array(iq)))
          full_path = os.path.join(directory, "Results")
          if not os.path.exists(full_path):
            os.makedirs(full_path)
          final_path = os.path.join(full_path, name)
          iq.to_csv(f'{final_path}.csv', sep='\t', index=False, header=False)
          return None
        else:
          iq = pd.DataFrame(np.transpose(np.array(iq)))
          full_path = os.path.join(directory, "Results")
          if not os.path.exists(full_path):
            os.makedirs(full_path)
          final_path = os.path.join(full_path, name)
          iq.to_csv(f'{final_path}.csv', sep='\t', index=False, header=False)
          return None

    def save_iq_only_y(self, iq, name):
        if os.path.isfile(self.path):
    # Get the directory part of the file path
          directory = os.path.dirname(self.path)
          iq = pd.DataFrame(np.transpose(np.array(iq)))
          if iq.shape[1] >= 2:
            iq.drop(columns=[0], inplace=True)
          full_path = os.path.join(os.path.dirname(directory), "Results")
          if not os.path.exists(full_path):
            os.makedirs(full_path)
          final_path = os.path.join(full_path, name)
          iq.to_csv(f'{final_path}.csv', sep='\t', index=False, header=False)
          return None
        else:
          directory = self.path
          iq = pd.DataFrame(np.transpose(np.array(iq)))
          if iq.shape[1] >= 2:
            iq.drop(columns=[0], inplace=True)
          full_path = os.path.join(os.path.dirname(directory), "Results")
          if not os.path.exists(full_path):
            os.makedirs(full_path)
          final_path = os.path.join(full_path, name)
          iq.to_csv(f'{final_path}.csv', sep='\t', index=False, header=False)
          return None

#Encontrar o centro com a transformada de Hough para usar como chute inicial
class ImageAnalysis:
    def __init__(self):
        pass

    def find_center(self, image,  r, R, threshold, edges_thresh1 = 255, edges_thresh2 = 10):
        if image is None:
            return "Image not loaded properly. Check the image path."


        blur = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blur = cv2.GaussianBlur(blur, (3, 3), 30)
        
        final_im = np.where(blur > threshold, 255, 0)
        final_im = final_im.astype(np.uint8)
        
        edges = cv2.Canny(final_im, 0, 255)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=edges_thresh1, param2=edges_thresh2, minRadius= r, maxRadius=R)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            center_x, center_y, r = circles[0]
            return center_x, center_y, r
        else:
            return final_im, blur, edges
        
    def azimuth_integration_cv2(self, img, center, binning):
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

    def find_and_integrate(self, image, center_x, center_y, azimuth_range):
      ai = pyFAI.AzimuthalIntegrator(dist=0.1,  pixel1=1e-4, pixel2=1e-4)
      ai.setFit2D(image.shape[0]/2, center_x, center_y)
      result = ai.integrate1d(image, 3000, unit='2th_deg', azimuth_range=azimuth_range, method="ocl_lut_integr")
      peaks, _ = find_peaks(result[1], distance=150)
      return peaks

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

    def plot_iq(self, x):
        x = np.array(x)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        fig.suptitle("Integrated Iq", fontsize=16)
        ax.set_xlim(0, x[0].max())
        ax.set_ylim(0, x[1].max())
        ax.set_yticks(np.arange(0, 100, 160))
        ax.grid()
        ax.grid(which='minor', linestyle='--')
        ax.minorticks_on()
        ax.plot(x[0], x[1], label='Python')
        plt.show()
        return fig, ax

    def MSE(self, x, y):
        return math.sqrt(1/len(x) * sum((x - y)**2))

 

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