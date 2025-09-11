import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import tifffile
from mypackages.edp_processing import ImageAnalysis, ImageProcessing

def pad_image_for_hough(image, pad_width=512, mode='constant'):
    padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode=mode)
    return padded_image, pad_width

def plot_center(img, cx, cy, r, offset, analysis, side=False):
    binning = img.shape[0]
    data, polar_image, masked_image = analysis.azimuth_integration_cv2(
        img, center=[cx - offset, cy - offset], binning=binning
    )

    circle1 = Circle((cx, cy), r, fill=False, color='white', linestyle='--')

    fig, ax = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

    if side:
        ax[0].imshow(img**(1/2), cmap='inferno')
    else:
        ax[0].imshow(img**(1/2), cmap='inferno')
    ax[0].scatter(cx, cy, s=5, color='white')
    ax[0].add_patch(circle1)
    ax[0].axis('off')
    ax[0].text(cx, cy, f'{cx:.0f}, {cy:.0f}', color='white', fontsize=13, ha='center', va='center')

    ang1, ang2 = 40, 100
    start = int(ang1 * binning / 360)
    end = int(ang2 * binning / 360)

    ax[1].imshow(polar_image[start + 180:end + 180])
    ax[1].imshow(polar_image[start:end])

    ax[2].plot(polar_image[250:750,].mean(axis=0)[200:600])
    ax[2].plot(polar_image[1250:1750,].mean(axis=0)[200:600])

    plt.tight_layout()
    plt.show()

    return data

def refine_center(img, analysis, side=False, offset=0, threshold_init=150):
    threshold = threshold_init
    cx, cy, r, *_ = analysis.find_center(img, r=1, R=5000, threshold=threshold, niter=10, kappa=0, anisotropic=False)
    if offset:
        cx -= offset
        cy -= offset

    while True:
        print(f"Center found: ({cx:.2f}, {cy:.2f}) | threshold={threshold}")
        data = plot_center(img, cx, cy, r, offset, analysis, side=side)

        choice = simpledialog.askstring(
            "Adjust center/threshold",
            "Options: ok | center | threshold | cancel"
        )
        if not choice:
            continue
        choice = choice.strip().lower()

        if choice == "ok":
            return cx, cy, r, threshold, data
        elif choice == "center":
            s = simpledialog.askstring("Set center", "Enter center as x,y (pixels):")
            if s:
                try:
                    x_str, y_str = s.split(",")
                    cx, cy = float(x_str), float(y_str)
                except Exception:
                    messagebox.showerror("Error", "Invalid format. Use x,y")
        elif choice == "threshold":
            t = simpledialog.askinteger("Set threshold", "Enter integer threshold:", minvalue=0, maxvalue=1000000)
            if t is not None:
                threshold = int(t)
                cx, cy, r, *_ = analysis.find_center(img, r=1, R=5000, threshold=threshold, niter=10, kappa=0, anisotropic=False)
                if offset:
                    cx -= offset
                    cy -= offset
        elif choice == "cancel":
            raise RuntimeError("Operation cancelled by user.")

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select diffraction image",
    filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    initialdir=r"Z:\ActualWork\Victor\raw_data" # <- put your preferred default folder here
)

if not file_path:
    raise RuntimeError("No file selected.")

img = tifffile.imread(file_path)

masking = False
side = False
b = 2

analysis = ImageAnalysis()
processing = ImageProcessing(file_path)

offset = 0
img_padded = img
if side:
    img_padded, offset = pad_image_for_hough(img, pad_width=256)
else:
    img_padded = img

threshold0 = 80 if side else 150
cx, cy, r, threshold_used, data = refine_center(img_padded if side else img, analysis, side=side, offset=offset, threshold_init=threshold0)

save_dir = filedialog.askdirectory(title="Select folder to save CSV", initialdir=r"Z:\ActualWork\Victor\processed_data")
if not save_dir:
    raise RuntimeError("No folder selected.")

roi = pd.DataFrame(np.array([data]).T)
save_path = Path(save_dir) / (Path(file_path).stem + ".csv")
roi.to_csv(save_path, index=None, sep="\t")
print("Saved:", save_path)
