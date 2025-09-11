import math
import os
import ast
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")  # must be before importing pyplot
import matplotlib.pyplot as plt
from tkinter import filedialog
from mypackages.eRDF import DataProcessor, Gr

plt.ioff()

class ParameterDialog(tk.Toplevel):
    def __init__(self, master, defaults):
        super().__init__(master)
        self.title("Parameters")
        self.resizable(False, False)
        self.result = None

        self.vars = {
            "qmin": tk.StringVar(value=str(defaults["qmin"])),
            "qmax": tk.StringVar(value=str(defaults["qmax"])),
            "degree": tk.StringVar(value=str(defaults["degree"])),
            "rmax": tk.StringVar(value=str(defaults["rmax"])),
            "dr": tk.StringVar(value=str(defaults["dr"])),
            "elements": tk.StringVar(value=str(defaults["elements"]))
        }

        row = 0
        for label, key in [("qmin", "qmin"), ("qmax", "qmax"), ("degree", "degree"),
                           ("rmax", "rmax"), ("dr", "dr")]:
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
            qmin = float(self.vars["qmin"].get())
            qmax = float(self.vars["qmax"].get())
            degree = int(float(self.vars["degree"].get()))
            rmax = float(self.vars["rmax"].get())
            dr = float(self.vars["dr"].get())
            elements = ast.literal_eval(self.vars["elements"].get())
            if not isinstance(elements, dict):
                raise ValueError
        except Exception:
            tk.messagebox.showerror("Error", "Invalid parameter value.")
            return
        self.result = dict(qmin=qmin, qmax=qmax, degree=degree, rmax=rmax, dr=dr, elements=elements)
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

def main():
    root = tk.Tk()
    root.withdraw()
    start_name = filedialog.askopenfilename(
        title="Select diffraction CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir= r"Z:\ActualWork\Victor\processed_data"
    )
    if not start_name:
        raise RuntimeError("No file selected.")

    defaults = {
        "qmin": 1.0,
        "qmax": 18.0,
        "degree": 8,
        "rmax": 100.0,
        "dr": 0.01,
        "elements": {"Au": [79, 1]}
    }

    dlg = ParameterDialog(root, defaults)
    if dlg.result is None:
        raise RuntimeError("Cancelled.")

    qmin = dlg.result["qmin"]
    qmax = dlg.result["qmax"]
    degree = dlg.result["degree"]
    rmax = dlg.result["rmax"]
    dr = dlg.result["dr"]
    Elements = dlg.result["elements"]

    df0 = pd.read_csv(start_name, header=None)
    ds = (df0.iloc[0, 0])/(2*math.pi)
    df = pd.read_csv(start_name, header=None, skiprows=1)
    data = df.sum(axis=1)

    start = int(qmin / (ds * 2 * math.pi))
    end = int(qmax / (ds * 2 * math.pi))
    end = min(end, data.shape[-1])

    dp1 = DataProcessor(data=data, q0=0, lobato_path=None, start=start, end=end, ds=ds, Elements=Elements, region=0)
    _, _ = dp1.SQ_PhiQ(dp1.iq, damping=0)

    norm_data = dp1.iq / (dp1.N * dp1.fq_sq)
    norm_data = norm_data * dp1.q

    coefficients = np.polyfit(dp1.q, norm_data, degree)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(dp1.q)
    fq_poly = norm_data - y_fit

    r_raw, Gr_raw = Gr(dp1.q, fq_poly, rmax=rmax, dr=dr)
    # close Tk root so Matplotlib owns the event loop
    try:
        root.destroy()
    except Exception:
        pass

    dp1.plot_results(fq_poly, r_raw, Gr0=Gr_raw)
    plt.show(block=True) 

    save_dir = filedialog.askdirectory(
        title="Select folder to save results",
        initialdir= r"Z:\ActualWork\Victor\processed_data"
    )
    if not save_dir:
        raise RuntimeError("No save folder selected.")

    pd.DataFrame(np.array([dp1.q, fq_poly]).T, columns=["q", "fq"]).to_csv(
        os.path.join(save_dir, "fq_output.csv"), sep="\t", index=None
    )
    pd.DataFrame(np.array([r_raw, Gr_raw]).T, columns=["r", "Gr"]).to_csv(
        os.path.join(save_dir, "gr_output.csv"), sep="\t", index=None
    )

    print("Saved fq_output.csv and gr_output.csv in", save_dir)

if __name__ == "__main__":
    main()
