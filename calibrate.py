import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tkinter import Tk, filedialog
from mypackages.edp_processing import ImageAnalysis, ImageProcessing, peak_calibration
from azim_integ import refine_center


try:
    from mypackages.edp_processing import (
        hot_pixel_filter,
        remove_n_smallest_and_add_offset,
        apply_timepix_cross,
        apply_us4000_mask,
        apply_beamstop_mask,
    )
except Exception:
    def hot_pixel_filter(x, thr=100, ksize=3): return x
    def remove_n_smallest_and_add_offset(x, n=50, offset=0): return x
    def apply_timepix_cross(x): return x
    def apply_us4000_mask(x): return x
    def apply_beamstop_mask(x, path): return x



def _find_center(img, analysis, padded_offset=0, threshold=110):
    cx, cy, _, _, _, _ = analysis.find_center(img, r=1, R=5000, threshold=threshold, niter=20, kappa=100)
    if padded_offset:
        cx -= padded_offset
        cy -= padded_offset
    return float(cx), float(cy)


def _auto_select_peaks(profile_slice, n_peaks=4, distance=10, height=None, prominence=50, min_pixel_rel=0):
    pk_all, props = find_peaks(profile_slice, distance=distance, height=height, prominence=prominence)
    mask = pk_all >= min_pixel_rel
    peaks = pk_all[mask]
    if len(peaks) == 0:
        return peaks
    prom = props.get("prominences", None)
    if prom is None:
        prom = np.ones(len(pk_all), dtype=float)
    prom = np.asarray(prom)[mask]
    top = peaks[np.argsort(prom)[::-1][:n_peaks]]
    return np.sort(top)

def _extract_pixel_size(calib_result):
    if hasattr(calib_result, "pixel_size"):
        return float(calib_result.pixel_size)
    if isinstance(calib_result, dict) and "pixel_size" in calib_result:
        return float(calib_result["pixel_size"])
    if isinstance(calib_result, (float, int, np.floating)):
        return float(calib_result)
    raise ValueError("Calibration result missing pixel size.")

def _plot_profile_with_peaks(profile_slice, peaks_rel, x_start=0, title=None):
    x = np.arange(x_start, x_start + len(profile_slice))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x, profile_slice, lw=1)
    if len(peaks_rel):
        peaks_abs = x_start + peaks_rel
        ax.scatter(peaks_abs, profile_slice[peaks_rel], s=32)
        for i, (p_abs, p_rel) in enumerate(zip(peaks_abs, peaks_rel)):
            ax.text(int(p_abs), float(profile_slice[p_rel]), f"{i}", ha="center", va="bottom", fontsize=9)
        ax.vlines(peaks_abs, 0, profile_slice[peaks_rel], linestyles="dashed", linewidth=0.8)
    ax.set_xlabel("pixel radius (absolute)")
    ax.set_ylabel("intensity")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def _prompt_subset(peaks_rel, profile_slice, default_n, start_offset):
    print("Found peaks (ordered by absolute pixel):")
    for i, p_rel in enumerate(peaks_rel):
        p_abs = int(p_rel + start_offset)
        print(f"[{i}] rel={int(p_rel)} abs={p_abs} I={float(profile_slice[p_rel]):.1f}")
    s = input(f"Select indices (e.g. 0,1,3) or press Enter for first {default_n}: ").strip()
    if not s:
        return np.arange(min(default_n, len(peaks_rel)))
    idx = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            a, b = token.split(":")
            a = int(a) if a else 0
            b = int(b) if b else len(peaks_rel)
            idx.extend(list(range(a, b)))
        else:
            idx.append(int(token))
    idx = np.array(sorted(set([i for i in idx if 0 <= i < len(peaks_rel)])), dtype=int)
    if len(idx) == 0:
        idx = np.arange(min(default_n, len(peaks_rel)))
    return idx

def calibrate_gold_tiff(
    path,
    pad=256,
    threshold_center=110,
    min_pixel_rel=0,
    n_peaks=4,
    distance=10,
    height=None,
    prominence=50,
    use_timepix_mask=False,
    use_us4000_mask=False,
    beamstop_mask_path=None,
    preprocess=True,
    interactive=True,
    show_plot=True,
    subset_indices=None,
    start_offset=0,
    manual=False,
    c=None
):
    analysis = ImageAnalysis()
    processing = ImageProcessing()
    img = tifffile.imread(path)
    # if preprocess:
    #     img = hot_pixel_filter(img, thr=100, ksize=3)
    #     img = remove_n_smallest_and_add_offset(img, n=50, offset=44)
    # if use_timepix_mask:
    #     img = apply_timepix_cross(img)
    # if use_us4000_mask:
    #     img = apply_us4000_mask(img)
    # if beamstop_mask_path:
    #     img = apply_beamstop_mask(img, beamstop_mask_path)
    side = False
    if side:
        padded, pad_off = processing.pad_for_center(img, pad_width=pad)
    else:
        padded, pad_off = img, 0
    

    if manual:
        profile = profile, _, _ = analysis.azimuth_integration_cv2(img, center=[cx, cy])
    else:
        cx,cy, _, _, profile = refine_center(padded if side else img, analysis, side=side, offset=pad_off, threshold_init=threshold_center)
        # cx, cy = _find_center(padded, analysis, padded_offset=pad_off, threshold=threshold_center)
    print(cx, cy)
    
    
    slice_profile = profile[start_offset:]
    peaks_rel = _auto_select_peaks(
        slice_profile,
        n_peaks=max(n_peaks, 4),
        distance=distance,
        height=height,
        prominence=prominence,
        min_pixel_rel=min_pixel_rel
    )
    if show_plot:
        _plot_profile_with_peaks(slice_profile, peaks_rel, x_start=start_offset, title="Azimuthal integration with detected peaks")
        plt.show()
    if subset_indices is None and interactive:
        subset_indices = _prompt_subset(peaks_rel, slice_profile, default_n=n_peaks, start_offset=start_offset)
    elif subset_indices is None:
        subset_indices = np.arange(min(n_peaks, len(peaks_rel)))
    subset_indices = np.asarray(subset_indices, dtype=int)
    subset_pixels_abs = np.asarray(peaks_rel[subset_indices], dtype=float) + float(start_offset)
    calib = peak_calibration(pixel_positions=subset_pixels_abs, standard="gold")
    px = _extract_pixel_size(calib)
    return px, {
        "center": (cx, cy),
        "profile": profile,
        "slice_profile": slice_profile,
        "peaks_rel": peaks_rel,
        "peaks_abs": peaks_rel + start_offset,
        "subset_indices": subset_indices,
        "subset_pixels_abs": subset_pixels_abs
    }
def pick_tiff(initialdir=None):
    root = Tk()
    root.withdraw()
    fp = filedialog.askopenfilename(
        title="Select TIFF",
        initialdir=initialdir,
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
    )
    root.destroy()
    if not fp:
        raise SystemExit("No file selected.")
    return fp

def pick_csv(initialdir=None):
    root = Tk()
    root.withdraw()
    fp = filedialog.asksaveasfilename(
        title="Select or Create CSV file",
        initialdir=initialdir,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    if not fp:
        raise SystemExit("No CSV file selected.")
    return fp

def prepend_csv_row(csv_path, row):
    # Read existing rows if file exists
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            old_rows = f.readlines()
    else:
        old_rows = []

    # Write new row + old rows
    with open(csv_path, "w", newline="") as f:
        f.write(",".join(map(str, row)) + "\n")
        f.writelines(old_rows)

if __name__ == "__main__":
    selected_path = pick_tiff()
    px, diag = calibrate_gold_tiff(
        selected_path,
        pad=256,
        threshold_center=80,
        min_pixel_rel=0,
        n_peaks=10,
        distance=5,
        prominence=20,
        use_timepix_mask=False,
        use_us4000_mask=True,
        beamstop_mask_path=None,
        preprocess=True,
        interactive=True,
        show_plot=True,
        subset_indices=None,
        start_offset=50,
        manual=False,
        c=(2022.00, 1860.00)
    )
    print(px)

    csv_path = pick_csv()
    prepend_csv_row(csv_path, [px])
    print(f"Calibration value {px} prepended to {csv_path}")