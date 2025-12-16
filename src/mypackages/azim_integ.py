import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import tifffile
from control_state import control
from edp_processing import ImageAnalysis, ImageProcessing
from plot_style import set_plot_style

set_plot_style()


analysis = ImageAnalysis()

def span_deg(ang1, ang2):
    return [
        (ang1, ang2),
        ((ang1 + 180) % 360, (ang2 + 180) % 360),
        ((ang1 + 90)  % 360, (ang2 + 90)  % 360),
        ((ang1 + 270) % 360, (ang2 + 270) % 360),
    ]


def draw_span(ax, nang, a_deg, b_deg, **kwargs):
    if a_deg <= b_deg:
        ax.axhspan(a_deg, b_deg, alpha=0.25, **kwargs)
    else:
        ax.axhspan(0, b_deg, alpha=0.25, **kwargs)
        ax.axhspan(a_deg, 360, alpha=0.25, **kwargs)

def make_sections(img, cx, cy, r, offset, analysis, side=False):
    data, beam_stop_img, masked_image = analysis.azimuth_integration_cv2(
        img, center=[cx - offset, cy - offset]
    )

    nang, nrad = masked_image.shape
    ang1, ang2 = 40, 80

    spans = span_deg(ang1, ang2)

    rmin, rmax = 200, 600
    rmin = max(0, rmin)
    rmax = min(nrad, rmax)

    def avg_profile(a_deg, b_deg):
        a_pix = int(a_deg * nang / 360)
        b_pix = int(b_deg * nang / 360)
        if a_pix < b_pix:
            rows = np.arange(a_pix, b_pix)
        else:
            rows = np.r_[np.arange(a_pix, nang), np.arange(0, b_pix)]
        return masked_image[rows, rmin:rmax].mean(axis=0)

    profs = [avg_profile(a, b) for (a, b) in spans]
    x = np.arange(rmin, rmax)

    return (
        data,
        beam_stop_img,
        masked_image,
        profs,
        spans,
        nang,
        nrad,
        x,
        ang1,
        ang2,
    )

def main_peak(x, y):
    """Return the x-position of the dominant peak in profile y."""
    peaks, props = find_peaks(y, prominence=2)
    if len(peaks) == 0:
        return None
    # select the peak with highest amplitude
    p = peaks[np.argmax(y[peaks])]
    return x[p]

def plot_center(img, cx, cy, r, offset, analysis, side=False):

    (
        data,
        beam_stop_img,
        masked_image,
        profs,
        spans,
        nang,
        nrad,
        x,
        ang1,
        ang2,
    ) = make_sections(img, cx, cy, r, offset, analysis)

    prof1, prof2, prof3, prof4 = profs

    fig, ax = plt.subplots(
        1, 4, figsize=(18, 5),
        gridspec_kw={'width_ratios': [2, 1, 1, 1]}
    )

    ax[0].imshow(beam_stop_img**0.5, cmap='inferno')
    ax[0].scatter(cx, cy, s=5, color='white')
    ax[0].add_patch(Circle((cx, cy), r, fill=False, color='white', linestyle='--'))
    ax[0].axis('off')
    ax[0].text(cx, cy, f'{cx:.0f}, {cy:.0f}', color='white', fontsize=13,
               ha='center', va='center')

    ax[1].imshow(masked_image, aspect='auto', origin='upper',
                 extent=[0, nrad, 360, 0])
    ax[1].set_ylabel("Azimuth (°)")
    ax[1].set_xlabel("Radius (pixels)")
    ax[1].set_ylim(0, 360)

    for (a_deg, b_deg) in spans:
        draw_span(ax[1], nang, a_deg, b_deg)

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


def refine_center(img, analysis = analysis, side=False, offset=0, threshold_init=150):
    processing = ImageProcessing(img)
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
            data, _, _ = analysis.azimuth_integration_cv2(img, center=[cx - offset, cy - offset])
            control.center = (cx, cy)
            control.data = data
            control.radius = r
            if control.viewer is not None:
                control.viewer.event_generate("<<CenterFound>>")
            break

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


def main(save=False):
    root = tk.Tk()
    root.withdraw()

    side = False
    manual = False  

    control.load_tif_file()
    img = control.img
    file_path = control.img_path

    control.img = False

    control.load_tif_file(
        passing=True,
        initial_dir=r"C:\Users\seccolev\PyeRDF\PyeRDF\src\mypackages\data\masks"
    )
    beamstop_mask = control.img


    processing = ImageProcessing(img)
    img = processing.apply_beamstop_mask(beamstop_mask)
   
    #img = processing.hot_pixel_filter_sigma(ksize=5, sigma=3)

    pad = 0

    if side:
        processing = ImageProcessing(img)
        padded, pad_off = processing.pad_for_center(
            pad_width=pad, axis=1, side='right'
        )
    else:
        padded, pad_off = img, 0

    if manual:
        s = simpledialog.askstring("Set center", "Enter center as x,y (pixels):")
        if s is None:
            raise RuntimeError("Cancelled.")
        s = s.strip()
        try:
            x_str, y_str = s.split(",")
            cx, cy = float(x_str), float(y_str)
        except Exception:
            messagebox.showerror("Error", "Invalid format. Use x,y")
        offset = 0
        data, _, _ = analysis.azimuth_integration_cv2(
            img, center=[cx - offset, cy - offset]
        )
        control.center = (cx, cy)
        control.data = data
        r = 0
        _ = plot_center(
            padded, cx, cy, r, offset, analysis, side=side
        )

    else:
        threshold0 = 80 if side else 150
        refine_center(
            padded if side else img,
            analysis,
            side=side,
            offset=pad_off,
            threshold_init=threshold0
        )

    if save:
        save_dir = filedialog.askdirectory(title="Select folder to save CSV")
        if not save_dir:
            raise RuntimeError("No folder selected.")

        data = control.data
        roi = pd.DataFrame(np.array([data]).T)
        save_path = Path(save_dir) / (Path(file_path).stem + ".csv")
        roi.to_csv(save_path, index=None, sep="\t")
        print("Saved:", save_path)


if __name__ == "__main__":
    main(save = True)

