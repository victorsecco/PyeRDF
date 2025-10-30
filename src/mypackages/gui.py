import tkinter as tk
from tkinter import ttk
import periodictable
from gui_helpers import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from edp_processing import ImageProcessing

control = Controller()

class pyerdf_gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyeRDF")
        self.geometry("")
        self.minsize(600,600)
        
        
        #widgets
        self.toolbar = Toolbar(self, control)
        self.menu = erdf_menu(self)
        control.menu_frame = self.menu
        self.plot_window = Plot_window(self)

        self.mainloop()


class erdf_menu(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # ttk.Label(self, background= "#722F37").pack(expand  = True, fill  = "both")
        self.pack(side="left")

        self.pack(side="left", fill="y")
        self.image_widgets = []
        self.csv_widgets = []
        self.show_default_inputs()

    def clear_widgets(self, widgets):
        for w in widgets:
            w.destroy()
        widgets.clear()

    def show_default_inputs(self):
        ttk.Label(self, text="Load a file to begin").grid(row=0, column=0)

    def show_img_inputs(self):
        self.clear_widgets(self.image_widgets)
        self.clear_widgets(self.csv_widgets)

        label = ttk.Label(self, text="CSV analysis tools")
        label.grid(row=0, column=0)
        self.csv_widgets.append(label)

        button = ttk.Button(self, text="Plot columns", command=self.controller.plot_csv)
        button.grid(row=1, column=0)
        self.csv_widgets.append(button)

    def show_csv_inputs(self):
        self.clear_widgets(self.csv_widgets)
        self.clear_widgets(self.image_widgets)

        dq_label = ttk.Label(self, text="dq calibration")
        dq_entry = ttk.Entry(self)
        dq_label.grid(row=0, column=0)
        dq_entry.grid(row=0, column=1)
        self.image_widgets.extend([dq_label, dq_entry])

        options = get_elements()
        default = tk.StringVar(value="0")
        for i in range(1, 6):
            drop = DropdownMenu(self, options, default="No element selected")
            frac = ttk.Entry(self, textvariable=default, width=6)
            drop.grid(row=i, column=0)
            frac.grid(row=i, column=1)
            self.image_widgets.extend([drop, frac])

class DropdownMenu(ttk.Frame):
    def __init__(self, parent, options, default=None, width=10):
        super().__init__(parent)
        self.selections = {}
        self.var = tk.StringVar(value=default or options[0])
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=options,
            state="readonly", width=width
        )
        self.combo.pack(side="left", pady=3)
        self.combo.bind("<<ComboboxSelected>>", lambda event: handle_selection(self.var.get()))
        
    def handle_selection(self, value):
        options = get_elements()
        number = periodictable.elements[options.index(value) + 1].number
        symbol = periodictable.elements[number].symbol
        self.selections[symbol] = [number] 


class Plot_window(ttk.Frame):
    def __init__(self, parent): 
        super().__init__(parent)
        
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.pack(side="right")
        control.viewer = self

    def update_image(self):
        self.ax.clear()
        if control.img is not None:
            self.processing = ImageProcessing(control.img)
            binned = self.processing.bin_to_512()
            self.ax.imshow(binned)
            self.ax.axis("off")
            self
        self.canvas.draw()

    def update_plot(self):
        self.ax.clear()
        if control.data is not None:
            self.ax.plot(control.data)
            self.fig = Figure(figsize=(10, 5))
            self.ax.set_xlim(control.data.argmax(), control.data.argmin())
            self
        self.canvas.draw()
        
class Toolbar(tk.Menu):
    def __init__(self, parent, loader):
        super().__init__(parent)
        self.loader = loader
        parent.config(menu=self)
        self.create_tools()

    def create_tools(self):

        # Open menu
        open_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label="File", menu=open_menu)

        img_menu = tk.Menu(open_menu, tearoff=0)  # define before use
        open_menu.add_cascade(label="Images", menu=img_menu)

        img_menu.add_command(label=".ser", command=self.loader.load_ser_file)
        img_menu.add_command(label=".png", command=self.loader.load_png_file)
        img_menu.add_command(label=".tif", command=self.loader.load_tif_file)

        open_menu.add_command(label=".csv", command=self.loader.load_csv_file)

        # --- Analysis menu ---
        # analysis_menu = tk.Menu(menu_bar, tearoff=0)
        # menu_bar.add_cascade(label="Analysis", menu=analysis_menu)
        # analysis_menu.add_command(label="Find Center", command=self.run_find_center)
        # analysis_menu.add_command(label="Apply Mask", command=self.apply_mask)
        # analysis_menu.add_command(label="Azimuthal Integration", command=self.run_azimuthal_integration)
        # analysis_menu.add_command(label="eRDF Analysis (F(Q), G(r))", command=self.open_erdf_window)



if __name__ == "__main__":
    pyerdf_gui()




