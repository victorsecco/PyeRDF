import tkinter as tk
from azim_integ import main as azim_main
from calibrate import main as calib_main
from run_erdf import main as erdf_main

def main():
    actions = {
        "Azimuthal integration": azim_main,
        "Calibration": calib_main,
        "Run eRDF": erdf_main,
    }

    root = tk.Tk()
    root.title("PyeRDF launcher")
    root.geometry("350x180")

    for label, func in actions.items():
        tk.Button(
            root,
            text=label,
            height=2,
            command=func
        ).pack(fill="x", padx=20, pady=8)

    root.mainloop()

if __name__ == "__main__":
    main()