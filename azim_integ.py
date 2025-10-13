import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import tifffile
from mypackages.edp_processing import ImageAnalysis, ImageProcessing

def plot_center(img, cx, cy, r, offset, analysis, side=False):
    data, polar_image, masked_image = analysis.azimuth_integration_cv2(
        img, center=[cx - offset, cy - offset]
    )

    fig, ax = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [2, 1, 1, 1]})

    ax[0].imshow(img**0.5, cmap='inferno')
    ax[0].scatter(cx, cy, s=5, color='white')
    ax[0].add_patch(Circle((cx, cy), r, fill=False, color='white', linestyle='--'))
    ax[0].axis('off')
    ax[0].text(cx, cy, f'{cx:.0f}, {cy:.0f}', color='white', fontsize=13, ha='center', va='center')

    nang, nrad = masked_image.shape
    ang1, ang2 = 40, 100
    # start = int(ang1 * nang / 360)
    # end   = int(ang2 * nang / 360)
    # half  = nang // 2
    # quarter = nang // 4

    ax[1].imshow(masked_image, aspect='auto', origin='upper',
                 extent=[0, nrad, 360, 0])  # top→0°, bottom→360°
    ax[1].set_ylabel("Azimuth (°)")
    ax[1].set_xlabel("Radius (pixels)")

    def span_deg(a_deg, b_deg, **kwargs):
        a_pix = a_deg * nang / 360
        b_pix = b_deg * nang / 360
        if a_pix <= b_pix:
            ax[1].axhspan(a_deg, b_deg, alpha=0.25, **kwargs)
        else:
            ax[1].axhspan(0, b_deg, alpha=0.25, **kwargs)
            ax[1].axhspan(a_deg, 360, alpha=0.25, **kwargs)

    # base and rotated angular regions
    span_deg(ang1, ang2)
    span_deg((ang1+180)%360, (ang2+180)%360)
    span_deg((ang1+90)%360, (ang2+90)%360)
    span_deg((ang1+270)%360, (ang2+270)%360)

    rmin, rmax = 200, 600
    rmin = max(0, rmin); rmax = min(nrad, rmax)

    def avg_profile(a_deg, b_deg):
        a_pix = int(a_deg * nang / 360)
        b_pix = int(b_deg * nang / 360)
        if a_pix < b_pix:
            rows = np.arange(a_pix, b_pix)
        else:
            rows = np.r_[np.arange(a_pix, nang), np.arange(0, b_pix)]
        return masked_image[rows, rmin:rmax].mean(axis=0)

    prof1 = avg_profile(ang1, ang2)
    prof2 = avg_profile((ang1+180)%360, (ang2+180)%360)
    prof3 = avg_profile((ang1+90)%360, (ang2+90)%360)
    prof4 = avg_profile((ang1+270)%360, (ang2+270)%360)

    x = np.arange(rmin, rmax)

    ax[2].plot(x, prof1, label=f'{ang1}–{ang2}°')
    ax[2].plot(x, prof2, label=f'{(ang1+180)%360}–{(ang2+180)%360}°')
    ax[2].set_xlabel('radius (pixels)')
    ax[2].set_ylabel('intensity')
    ax[2].legend()

    ax[3].plot(x, prof3, label=f'{(ang1+90)%360}–{(ang2+90)%360}°')
    ax[3].plot(x, prof4, label=f'{(ang1+270)%360}–{(ang2+270)%360}°')
    ax[3].set_xlabel('radius (pixels)')
    ax[3].set_ylabel('intensity')
    ax[3].legend()

    plt.tight_layout()
    plt.show()
    return data

def refine_center(img, analysis, side=False, offset=0, threshold_init=150):
    threshold = threshold_init
    cx, cy, r, *_ = analysis.find_center(img, r=1, R=5000, threshold=threshold, niter=10, kappa=0, anisotropic=False)
    if offset != 0:
        cx -= offset
        cy -= offset

    while True:
        try:
            print(f"Center found: ({cx:.2f}, {cy:.2f}) | threshold={threshold}")
        except TypeError:
            threshold = simpledialog.askstring(
                "Adjust Threshold",
                "Center was not found, set another threshold:"
            )
            try:
                threshold = int(threshold)
            except (TypeError, ValueError):
                print("Invalid input. Enter an integer.")
                continue

            cx, cy, r, *_ = analysis.find_center(
                img, r=1, R=5000, threshold=threshold,
                niter=10, kappa=0, anisotropic=False
            )
        data = plot_center(img, cx, cy, r, offset, analysis, side=side)

        choice = simpledialog.askstring(
            "Adjust center/threshold",
            "Options: ok | c | t | cancel"
        )
        if choice is None:
            raise RuntimeError("Cancelled.")  # user hit Cancel on the options dialog

        choice = choice.strip().lower()
        if not choice:
            continue

        if choice == "ok":
            return cx, cy, r, threshold, data

        elif choice == "c":
            s = simpledialog.askstring("Set center", "Enter center as x,y (pixels):")
            if s is None:
                raise RuntimeError("Cancelled.")
            s = s.strip()
            try:
                x_str, y_str = s.split(",")
                cx, cy = float(x_str), float(y_str)
            except Exception:
                messagebox.showerror("Error", "Invalid format. Use x,y")

        elif choice == "t":
            t = simpledialog.askinteger(
                "Set threshold", "Enter integer threshold:",
                minvalue=0, maxvalue=1_000_000
            )
            if t is None:
                raise RuntimeError("Cancelled.")  # cancelled threshold entry
            threshold = int(t)
            cx, cy, r, *_ = analysis.find_center(
                img, r=1, R=5000, threshold=threshold, niter=10, kappa=0, anisotropic=False
            )
            if offset:
                cx -= offset
                cy -= offset

        else:
            messagebox.showinfo("Info", "Valid options: ok | c | t | cancel")    


def azim_integ(save = False):
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select diffraction image",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    )

    if not file_path:
        raise RuntimeError("No file selected.")

    img = tifffile.imread(file_path)

    masking = False
    side = False

    analysis = ImageAnalysis()
    processing = ImageProcessing()

    padded = img
    if side:
        padded, pad_off = processing.pad_for_center(img, pad_width=pad)
    else:
        padded, pad_off = img, 0

    threshold0 = 80 if side else 150
    _,_,_,_, data = refine_center(padded if side else img, analysis, side=side, offset=pad_off, threshold_init=threshold0)

    if save:
        save_dir = filedialog.askdirectory(title="Select folder to save CSV")
        if not save_dir:
            raise RuntimeError("No folder selected.")

        
        roi = pd.DataFrame(np.array([data]).T)
        save_path = Path(save_dir) / (Path(file_path).stem + ".csv")
        roi.to_csv(save_path, index=None, sep="\t")
        print("Saved:", save_path)


if __name__ == "__main__":
    azim_integ(save = True)
