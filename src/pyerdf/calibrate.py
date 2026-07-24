import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from edp_processing import ImageAnalysis, ImageProcessing, peak_calibration
from azim_integ import refine_center
from control_state import control
from plot_style import set_plot_style
import tkinter as tk

set_plot_style()

def choose_calibration_source(parent=None):
    choice = {"source": None}

    win = tk.Toplevel(parent)
    win.title("Calibration input")
    win.geometry("260x130")
    win.grab_set()

    tk.Label(win, text="Open calibration from:").pack(pady=10)

    def set_choice(source):
        choice["source"] = source
        win.destroy()

    tk.Button(win, text="TIFF image", command=lambda: set_choice("tif")).pack(fill="x", padx=30, pady=5)
    tk.Button(win, text="CSV profile", command=lambda: set_choice("csv")).pack(fill="x", padx=30, pady=5)

    win.wait_window()
    return choice["source"]

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

    lo, hi = 0, 3000
    profile_slice = profile_slice[lo:hi]
    x = np.arange(x_start + lo, x_start + lo + len(profile_slice))

    peaks_rel = [p - lo for p in peaks_rel if lo <= p < hi]

    fig, ax = plt.subplots(figsize=(7,10))
    ax.plot(x, profile_slice, lw=2)

    if len(peaks_rel):
        peaks_abs = x_start + lo + np.array(peaks_rel)
        ax.scatter(peaks_abs, profile_slice[peaks_rel], s=32, color="red")
        for i, (p_abs, p_rel) in enumerate(zip(peaks_abs, peaks_rel)):
            ax.text(int(p_abs), float(profile_slice[p_rel])+100, f"{i}", 
                    ha="center", va="bottom", fontsize=12)
        ax.vlines(peaks_abs, 0, profile_slice[peaks_rel],
                  linestyles="dashed", linewidth=0.8, color="red")

    ax.set_xlabel("pixel radius", fontsize=25)
    ax.set_ylabel("intensity", fontsize=25)
    ax.tick_params(axis='both', labelsize=22)

    if title:
        ax.set_title(title, fontsize=25)

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
    pad=256,
    threshold_center=110,
    min_pixel_rel=0,
    n_peaks=4,
    distance=10,
    height=None,
    prominence=50,
    interactive=True,
    show_plot=True,
    subset_indices=None,
    start_offset=0,
    manual=False,
    source="tif",
    c=None,
):
    analysis = ImageAnalysis()

    if source == "tif":
        control.load_tif_file()
        img = control.img

        control.img = False
        control.load_tif_file(passing=True)
        beamstop_mask = control.img

        processing = ImageProcessing(img)
        img = processing.apply_beamstop_mask(beamstop_mask)

        side = False
        if side:
            padded, pad_off = processing.pad_for_center()
        else:
            padded, pad_off = img, 0

        if manual:
            if c is None:
                raise ValueError("Manual mode needs c=(cx, cy).")
            cx, cy = c
            profile, _, _ = analysis.azimuth_integration_cv2(
                img,
                center=[cx, cy],
                binning=min(img.shape)
            )
            control.center = (cx, cy)
            control.data = profile
        else:
            refine_center(
                padded if side else img,
                analysis,
                side=side,
                offset=pad_off,
                threshold_init=threshold_center
            )

        cx, cy = control.center
        profile = control.data

    elif source == "csv":
        control.load_csv_file()
        profile = control.data
        cx, cy = None, None

    else:
        raise ValueError("source must be 'tif' or 'csv'.")

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
        _plot_profile_with_peaks(
            slice_profile,
            peaks_rel,
            x_start=start_offset,
            title="Azimuthal integration with detected peaks"
        )
        plt.show()

    if subset_indices is None and interactive:
        subset_indices = _prompt_subset(
            peaks_rel,
            slice_profile,
            default_n=n_peaks,
            start_offset=start_offset
        )
    elif subset_indices is None:
        subset_indices = np.arange(min(n_peaks, len(peaks_rel)))

    subset_indices = np.asarray(subset_indices, dtype=int)
    subset_pixels_abs = np.asarray(peaks_rel[subset_indices], dtype=float) + float(start_offset)

    calib = peak_calibration(pixel_positions=subset_pixels_abs, standard="gold")
    px = _extract_pixel_size(calib)

    return px, {
        "source": source,
        "center": (cx, cy),
        "profile": profile,
        "slice_profile": slice_profile,
        "peaks_rel": peaks_rel,
        "peaks_abs": peaks_rel + start_offset,
        "subset_indices": subset_indices,
        "subset_pixels_abs": subset_pixels_abs,
    }

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

def main(parent=None):
    source = choose_calibration_source(parent)

    if source is None:
        return

    px, diag = calibrate_gold_tiff(
        pad=256,
        threshold_center=200,
        min_pixel_rel=0,
        n_peaks=10,
        distance=5,
        prominence=20,
        interactive=True,
        show_plot=True,
        subset_indices=None,
        start_offset=50,
        manual=False,
        source=source,
        c=(2022.00, 1860.00)
    )

    print(px)

    control.load_csv_file()
    iq_path = control.csv_path
    prepend_csv_row(iq_path, [px])

    print(f"Calibration value {px} prepended to {iq_path}")

if __name__ == "__main__":
    main()



