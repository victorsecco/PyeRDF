import tifffile
# import dm4
from PIL import Image


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
        return tifffile.imread(file_path)
    
    def load_dm4(self, file_path):
        import dm4
        with dm4.DM4File.open(file_path) as dm4file:

            tags = dm4file.read_directory()

            image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
            image_tag = image_data_tag.named_tags['Data']

            XDim = dm4file.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
            YDim = dm4file.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])

            img = np.array(dm4file.read_tag_data(image_tag), dtype=np.float64)
            return np.reshape(img, (YDim, XDim))
        
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
    
    def load_empad_data(self, file_path, num_images):
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