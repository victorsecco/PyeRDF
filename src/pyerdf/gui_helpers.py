import tkinter as tk
from tkinter import ttk
import periodictable
from data_loader import DataLoader
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
from pathlib import Path
import math
import numpy as np
from eRDF import *

def get_elements():
    return [periodictable.elements[i].name for i in range(1, 82)]
    

def show_values():
    values = [v.get() for v in num_vars]
    print("Inputs:", values)

class Controller:
    def __init__(self):
        self.dl = DataLoader() 

        self.img = None
        self.img_path = None
        self.num_frames = None

        self.data = None
        self.csv_path = None 
        self.ds = None

        self.center = None
        self.radius = None

        self.save_path = None

        self.viewer = None
        self.menu_frame = None

    def load_image_file(self, passing=None, initial_dir=None):
        self.img_path = None
        self.img_paths = None
        self.num_frames = None

        filetypes = [
            ("Image files", "*.png *.tif *.tiff *.ser *.dm3 *.dm4"),
            ("PNG files", "*.png"),
            ("TIFF files", "*.tif *.tiff"),
            ("SER files", "*.ser"),
            ("DM files", "*.dm3 *.dm4"),
        ]

        paths = filedialog.askopenfilenames(
            filetypes=filetypes
        )

        if not paths:
            if passing:
                return
            raise RuntimeError("No image file selected.")

        ext = Path(paths[0]).suffix.lower()

        if ext == ".png":
            if len(paths) > 1:
                raise RuntimeError("Please select only one PNG file.")

            self.img_path = paths[0]
            self.img = self.dl.load_png(self.img_path)

        elif ext in [".tif", ".tiff"]:
            self.img_paths = paths
            self.img_path = paths[0]
            self.img = self.dl.load_tif(self.img_paths)

        elif ext == ".ser":
            if len(paths) > 1:
                raise RuntimeError("Please select only one SER file.")

            self.img_path = paths[0]
            self.img, self.num_frames = self.loader.load_ser(self.img_path)

        elif ext in [".dm3", ".dm4"]:
            if len(paths) > 1:
                raise RuntimeError("Please select only one DM file.")

            self.img_path = paths[0]
            self.img = self.dl.load_dm3(self.img_path)

        else:
            raise RuntimeError(f"Unsupported file extension: {ext}")

        if self.viewer:
            self.viewer.update_img()

        if self.menu_frame:
            self.menu_frame.show_img_inputs()

    def load_png_file(self):
        self.load_image_file()

    def load_tif_file(self, passing=None, initial_dir=None):
        self.load_image_file(passing=passing)

    def load_ser_file(self):
        self.load_image_file()

    def load_csv_file(self, ds_from_file = False):        

        self.csv_path = filedialog.askopenfilename(
            title="Select diffraction CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not self.csv_path:
            raise RuntimeError("No file selected.")
        
        self.csv_path = Path(self.csv_path)

        df0 = pd.read_csv(self.csv_path, header=None)
        if ds_from_file:
            self.ds = (df0.iloc[0, 0]) / (2 * math.pi)
            df = pd.read_csv(self.csv_path, header=None, skiprows=2)
        else:
            df = pd.read_csv(self.csv_path, header=None, skiprows=0)
        
        self.data = df.sum(axis=1).values

        if self.viewer:
            self.viewer.update_plot()
        if self.menu_frame:
            self.menu_frame.show_csv_inputs()

    def build_element_dict(self, elements, fractions):
        element_dict = {}
        for i, (sym, num) in elements.items():
            if i <= len(fractions):
                element_dict[sym] = [num, fractions[i-3].get()]
        self.element_dict = element_dict
    
    def calibrate_pattern(self, ds_var):
        processor = DataProcessor()
        self.ds = float(ds_var)
        s,_ = processor.build_s_range(self.ds, len(self.data))
        self.data = np.column_stack((s, self.data))
        if self.viewer:
            self.viewer.update_plot()

        