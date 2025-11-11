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

    def load_png_file(self):
        self.img_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if not self.img_path:
            raise RuntimeError("No file selected.")

        self.img = self.dl.load_png(self.img_path)
        
        if self.viewer:
            self.viewer.update_img()
        if self.menu_frame:
            self.menu_frame.show_img_inputs()


    def load_tif_file(self, passing=None):
        self.img_path = None
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
            self.viewer.update_img()
        if self.menu_frame:
            self.menu_frame.show_img_inputs()


    def load_ser_file(self):
        self.img_path = filedialog.askopenfilename(filetypes=[("SER files", "*.ser")])
        if not self.img_path:
            return

        self.img, self.num_frames = self.loader.load_ser(self.img_path)
        
        if self.viewer:
            self.viewer.update_img()
        if self.menu_frame:
            self.menu_frame.show_img_inputs()
        
        # if self.num_frames > 1:
        #     self.nav_frame.pack(pady=5)
        # else:
        #     self.nav_frame.pack_forget()
        #     self.current_index = 0

        # self.frame_entry.delete(0, tk.END)
        # self.frame_entry.insert(0, "0")
        # self.show_frame(self.current_index)

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
                element_dict[sym] = [num, fractions[i - 1]]
        self.element_dict = element_dict
    
    def calibrate_pattern(self, ds_var):
        processor = DataProcessor()
        self.ds = float(ds_var)
        s,_ = processor.build_s_range(self.ds, len(self.data))
        self.data = np.column_stack((s, self.data))
        if self.viewer:
            self.viewer.update_plot()
        