import tkinter as tk
from tkinter import ttk
import periodictable
from data_loader import DataLoader
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
from pathlib import Path
import math

def get_elements():
    return [periodictable.elements[i].name for i in range(1, 82)]
    

def show_values():
    values = [v.get() for v in num_vars]
    print("Inputs:", values)

class Controller:
    def __init__(self):
        self.dl = DataLoader() 

        self.img = None
        self.data = None
        self.img_path = None 
        self.csv_path = None 
        self.save_path = None
        self.num_frames = None

        self.viewer = None
        self.menu_frame = None

    def load_png_file(self):
        self.img_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if not self.img_path:
            raise RuntimeError("No file selected.")

        self.img = self.dl.load_png(self.img_path)
        
        if self.viewer:
            self.viewer.update_image()
        if self.menu_frame:
            self.menu_frame.show_image_inputs()
        # self.full_image = img
        # self.current_image = bin_image(img, factor=2)
        # self.nav_frame.pack_forget()
        # self.display_image(self.current_image, title="PNG Image")

    def load_tif_file(self, passing=None):
        self.img_path = filedialog.askopenfilename(
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )

        if not self.img_path:
            if passing:
                return  # silently skip
            else:
                raise RuntimeError("No TIFF file selected.")
        
        self.img = self.dl.load_tif(self.img_path)

        if self.viewer:
            self.viewer.update_image()
        if self.menu_frame:
            self.menu_frame.show_image_inputs()



        # if img.dtype != np.uint16:
        #     print("Warning: image is not 16-bit.")

        # self.full_image = img
        # self.current_image = bin_image(img, factor=2)
        # self.nav_frame.pack_forget()
        # self.display_image(self.current_image, title="TIFF Image")

    def load_ser_file(self):
        self.img_path = filedialog.askopenfilename(filetypes=[("SER files", "*.ser")])
        if not self.img_path:
            return

        self.img, self.num_frames = self.loader.load_ser(self.img_path)
        
        if self.viewer:
            self.viewer.update_image()
        if self.menu_frame:
            self.menu_frame.show_image_inputs()
        
        # if self.num_frames > 1:
        #     self.nav_frame.pack(pady=5)
        # else:
        #     self.nav_frame.pack_forget()
        #     self.current_index = 0

        # self.frame_entry.delete(0, tk.END)
        # self.frame_entry.insert(0, "0")
        # self.show_frame(self.current_index)

    def load_csv_file(self):        

        self.csv_path = filedialog.askopenfilename(
            title="Select diffraction CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not self.csv_path:
            raise RuntimeError("No file selected.")
        
        self.csv_path = Path(self.csv_path)

        df0 = pd.read_csv(self.csv_path, header=None)
        ds = (df0.iloc[0, 0]) / (2 * math.pi)
        df = pd.read_csv(self.csv_path, header=None, skiprows=2)
        self.data = df.sum(axis=1).values

        if self.viewer:
            self.viewer.update_plot()
        if self.menu_frame:
            self.menu_frame.show_csv_inputs()