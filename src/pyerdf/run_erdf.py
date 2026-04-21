import math
import ast
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import filedialog
from pathlib import Path
from eRDF import DataProcessor
from control_state import control

plt.ioff()

class ParameterDialog(tk.Toplevel):
    def __init__(self, master, defaults):
        super().__init__(master)
        self.title("Parameters")
        self.resizable(False, False)
        self.result = None

        self.vars = {
            "q0": tk.StringVar(value=str(defaults["q0"])),
            "qmin": tk.StringVar(value=str(defaults["qmin"])),
            "qmax": tk.StringVar(value=str(defaults["qmax"])),
            "degree": tk.StringVar(value=str(defaults["degree"])),
            "rmax": tk.StringVar(value=str(defaults["rmax"])),
            "dr": tk.StringVar(value=str(defaults["dr"])),
            "elements": tk.StringVar(value=str(defaults["elements"])),
            "damping": tk.StringVar(value=str(defaults["damping"]))
        }

        row = 0
        for label, key in [("q0", "q0"), ("qmin", "qmin"), ("qmax", "qmax"), ("degree", "degree"),
                        ("rmax", "rmax"), ("dr", "dr"), ("damping", "damping")]:
            tk.Label(self, text=label).grid(row=row, column=0, padx=8, pady=6, sticky="e")
            tk.Entry(self, textvariable=self.vars[key], width=18).grid(row=row, column=1, padx=8, pady=6, sticky="w")
            row += 1

        tk.Label(self, text="Elements dict").grid(row=row, column=0, padx=8, pady=6, sticky="ne")
        tk.Entry(self, textvariable=self.vars["elements"], width=32).grid(row=row, column=1, padx=8, pady=6, sticky="w")
        row += 1

        btnf = tk.Frame(self)
        btnf.grid(row=row, column=0, columnspan=2, pady=10)
        tk.Button(btnf, text="Cancel", command=self.on_cancel, width=10).pack(side="right", padx=6)
        tk.Button(btnf, text="OK", command=self.on_ok, width=10).pack(side="right", padx=6)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.wait_window(self)

    def on_ok(self):
        try:
            q0 = float(self.vars["q0"].get())
            qmin = float(self.vars["qmin"].get())
            qmax = float(self.vars["qmax"].get())
            degree = int(float(self.vars["degree"].get()))
            rmax = float(self.vars["rmax"].get())
            dr = float(self.vars["dr"].get())
            elements = ast.literal_eval(self.vars["elements"].get())
            damping = float(self.vars["damping"].get())
            if not isinstance(elements, dict):
                raise ValueError
        except Exception:
            tk.messagebox.showerror("Error", "Invalid parameter value.")
            return
        self.result = dict(q0=q0, qmin=qmin, qmax=qmax, degree=degree, rmax=rmax, dr=dr,
                   damping=damping, elements=elements)
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

def main(ds = None):
    root = tk.Tk()
    root.withdraw()

    control.load_csv_file(ds_from_file=True)
    start_path = control.csv_path

    defaults = {
        "q0": 0,
        "qmin": 1,
        "qmax": 18.0,
        "degree": 8,
        "rmax": 100.0,
        "dr": 0.01,
        "damping": 0.0,
        "elements": {'Fe': [26, 3], 'O': [8, 4]} #{'Au': [79, 1],} 
    }

    dlg = ParameterDialog(root, defaults)
    if dlg.result is None:
        raise RuntimeError("Cancelled.")

    q0 = dlg.result["q0"]
    qmin = dlg.result["qmin"]
    qmax = dlg.result["qmax"]
    degree = dlg.result["degree"]
    rmax = dlg.result["rmax"]
    dr = dlg.result["dr"]
    Elements = dlg.result["elements"]
    damping = dlg.result["damping"]


    data = control.data
    ds = control.ds
    Elements

    try:
        start = int((qmin - q0) / (ds * 2 * math.pi))
    except OverflowError:
        raise OverflowError("The calibration factor ds cannot be zero. The ds should be the first line of your csv")

    end = int((qmax - q0) / (ds * 2 * math.pi))
    start = max(0, start)
    end = min(len(data), max(start + 1, end))

    dp = DataProcessor()

    x, iq, q, s, s2 = dp.load_and_process_data(
        data=data,
        start=start,
        end=end,
        ds=ds,
        q0=q0
    )
    
    dp.Lobato_Factors(elements=Elements)
    dp.compute_weighted_factors()
    dp.N_and_parameters(region=0.0)

    sq, fq = dp.sq_fq(iq, damping=damping)

    # CASE SWITCH: direct fq vs polynomial background fit
    if degree > 0:
        norm_data = (dp.iq / (dp.N * dp.f2_mean)) * dp.q
        coefficients = np.polyfit(dp.q, norm_data, degree)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(dp.q)
        fq_poly = norm_data - y_fit
        fq_used = fq_poly
    else:
        fq_used = fq

    weights = 1/dp.autofit
    weights = weights/weights.max()

    end = int((q[-1]-0) / (ds * 2 * math.pi))
    fq_used = fq_used[:end]

    r_raw, Gr_raw = dp.Gr(fq_used, rmax=rmax, dr=dr)

    try:
        root.destroy()
    except Exception:
        pass
    print(dp.C)
    dp.plot_results(q[:end], sq, r_raw, Gr0=Gr_raw)
    plt.show(block=True)

    save_dir = filedialog.askdirectory(
        title="Select folder to save results",
        initialdir=str(Path(r"Z:\ActualWork\Victor\processed_data"))
    )
    if not save_dir:
        raise RuntimeError("No save folder selected.")
    save_path = Path(save_dir)

    stem = start_path.stem
    fq_path = save_path / f"fq_{stem}.csv"
    gr_path = save_path / f"Gr_{stem}.csv"
    iq_path = save_path / f"iq_{stem}.csv"


    df_fq = pd.DataFrame({"q": dp.q, "fq": fq_used})
    df_gr = pd.DataFrame({"r": r_raw, "Gr": Gr_raw})
    df_iq = pd.DataFrame({"q": dp.q, "iq": dp.iq, "weight": weights})

    with fq_path.open("w", newline="") as f:
        f.write(f"# source={start_path.name}\n")
        df_fq.to_csv(f, index=None)

    with gr_path.open("w", newline="") as f:
        f.write(f"# source={start_path.name}\n")
        df_gr.to_csv(f, index=None)

    with iq_path.open("w", newline="") as f:
        f.write(f"# source={start_path.name}\n")
        df_iq.to_csv(f, index=None)

    print(f"Saved {fq_path}, {gr_path}, {iq_path}")

if __name__ == "__main__":
    main()
