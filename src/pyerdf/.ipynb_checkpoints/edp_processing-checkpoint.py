import os
import numpy as np
from PIL import Image
import cv2
from pystackreg import StackReg
import pandas as pd
import math
import matplotlib.pyplot as plt
import pyFAI
import copy


class ImageProcessing:
    def __init__(self, path):
        self.path = path

    def load_images(self, Binary = 1):
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
          for filename in images_names:
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
        return image
      else:
        raise Exception('A imagem tem'f'{image.shape} e a mascara tem 'f'{mask.shape}')


#Retirar os defeitos fixos das imagens
    def fixed_defects_mask(self, image, microscope):
      if microscope.lower() in ("titan"):
        image[2140:2160, 2030:2070]=False
        image[:,4087:]=False
        image[:,0:5]=False
        image[4051:4053]=False
        #image[3072:]=False
        return image
      else:
        image[:,:7]=False
        image[3072:]=False
        image[2136,1976:2225]=False
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

    def find_center(self, image,  r, R, threshold):
        if image is None:
            return "Image not loaded properly. Check the image path."


        blur = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blur = cv2.GaussianBlur(blur, (5, 5), 30)
        
        final_im = np.where(blur > threshold, 255, 0)
        final_im = final_im.astype(np.uint8)
        
        edges = cv2.Canny(final_im, 0, 255)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius= r, maxRadius=R)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            center_x, center_y, r = circles[0]
            return center_x, center_y, r
        else:
            return "No circles detected in the image."

    def find_and_integrate(self, image, center_x, center_y, azimuth_range):
      ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=0.1,  pixel1=1e-4, pixel2=1e-4)
      ai.setFit2D(image.shape[0]/2, center_x, center_y)
      result = ai.integrate1d(image, binning, unit='2th_deg', azimuth_range=azimuth_range, method="ocl_lut_integr")
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

    def azimuthal_projection(shape, center, original_image):
        output_image = np.zeros_like(shape)
        x_values, y_values = shape
        for x in range(x_values):
          for y in range(y_values):
            r = np.sqrt((x-center[0])**2 + (y-center[1])**2)

            theta = np.arctan2(y - center[0], x - center[1])

            src_x = int(center_x + r * np.cos(theta))
            src_y = int(center_y + r * np.sin(theta))


        if 0 <= src_x < original_image.shape[0] and 0 <= src_y < original_image.shape[1]:
            output_image[y, x] = original_image[src_y, src_x]

        return output_image
