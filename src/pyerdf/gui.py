import tkinter as tk
from tkinter import ttk
import periodictable
from control_state import control
from gui_helpers import get_elements
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from edp_processing import ImageProcessing
from azim_integ import refine_center
import run_erdf


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
        self.img_widgets = []
        self.csv_widgets = []
        self.show_default_inputs()

    def clear_widgets(self, widgets):
        for w in widgets:
            w.destroy()
        widgets.clear()

    def show_default_inputs(self):
        self.start_label = ttk.Label(self, text="Load a file to begin")
        self.start_label.grid(row=0, column=0)

    def show_img_inputs(self):
        self.start_label.destroy()
        self.clear_widgets(self.img_widgets)
        self.clear_widgets(self.csv_widgets)

        label = ttk.Label(self, text="Image analysis tools")
        label.grid(row=0, column=0, columnspan=2)
        
        data_menu = ttk.Button(master=self, 
                        text= "Data Analysis", command= lambda: self.show_csv_inputs())
        data_menu.grid(row = 1, column=0)


        find_center = ttk.Button(master=self, 
                                text= "Find Center", command= lambda: refine_center(control.img))
        find_center.grid(row = 2, column=0)

        azim_integ = ttk.Button(master=self, 
                            text= "Azimuthal Average",
                            command = lambda: control.viewer.update_plot())
        azim_integ.grid(row = 3, column=0)

        self.img_widgets.append(label)
        self.img_widgets.append(find_center)
        self.img_widgets.append(data_menu)
        self.img_widgets.append(azim_integ)

    def show_csv_inputs(self):
        self.start_label.destroy()
        self.clear_widgets(self.csv_widgets)
        self.clear_widgets(self.img_widgets)

        label = ttk.Label(self, text="Data analysis tools")
        label.grid(row=0, column=0, columnspan=2)

        data_menu = ttk.Button(master=self, 
                        text= "Image Analysis", command= lambda: self.show_img_inputs())
        data_menu.grid(row = 1, column=0)

        ds_label = ttk.Label(self, text="ds calibration")

        ds_var = tk.StringVar(value="0")
        ds_entry = ttk.Entry(self, textvariable=ds_var)
        ds_confirm = ttk.Button(master=self, 
                        text= "Calibrate", command= lambda: control.calibrate_pattern(ds_var.get()))

        ds_label.grid(row=2, column=0)
        ds_entry.grid(row=2, column=1)
        ds_confirm.grid(row=2, column=2)
        self.img_widgets.extend([ds_label, ds_entry, ds_confirm])

        options = get_elements()
        fraction_vars = []
        self.selections = {}
        
        for i in range(3, 8):
            drop = DropdownMenu(self, options, idx = i, 
                                handle_selection= self.handle_selection, 
                                default = "No element selected")

            default = tk.StringVar(value="0")
            frac = ttk.Entry(self, textvariable=default, width=6)

            drop.grid(row=i, column=0)
            frac.grid(row=i, column=1)
            self.img_widgets.extend([drop, frac])
            fraction_vars.append(default)

        confirm_elements = ttk.Button(master=self, 
                                      text= "Confirm", 
                                      command= lambda: control.build_element_dict(self.selections, fraction_vars))
        confirm_elements.grid(row = i+1, column=1)

        self.csv_widgets.append([label, data_menu, confirm_elements])

        
    def handle_selection(self, value, i):
        options = get_elements()
        number = periodictable.elements[options.index(value) + 1].number
        symbol = periodictable.elements[number].symbol
        self.selections[i] = (symbol, number) 

class DropdownMenu(ttk.Frame):
    def __init__(self, parent, options, idx, handle_selection, default=None, width=10):
        super().__init__(parent)
        self.selections = {}
        self.var = tk.StringVar(value=default or options[0])
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=options,
            state="readonly", width=width
        )
        self.combo.pack(side="left", pady=3)
        self.combo.bind("<<ComboboxSelected>>", lambda event: handle_selection(self.var.get(), idx))

# In Plot_window class
class Plot_window(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(side="right", fill="both", expand=True)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=0)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        control.viewer = self
        self.bind("<<CenterFound>>", self.on_center_found)

    def update_img(self):
        self.ax.clear()
        if control.img is not None:
            self.processing = ImageProcessing(control.img)
            binned, _ = self.processing.bin_to_512()
            log_binned = self.processing.log_intensity(binned)
            self.ax.imshow(log_binned)
            self.ax.axis("off")
            self
        self.canvas.draw()

    def update_plot(self):
        self.ax.clear()
        if control.data is not None:
            if len(control.data.shape) == 1:
                self.ax.plot(control.data)
                self.ax.set_xlim(0, len(control.data) - 1)
                self.ax.set_xlabel("Index")
                self.ax.set_ylabel("Counts")
            else:
                self.ax.plot(control.data[:,0], control.data[:,1])
                self.ax.set_xlim(0, control.data[:,0].max())
                self.ax.set_xlabel("Q(Å)")
                self.ax.set_ylabel("I(Q)")
            self.ax.set_aspect('auto', adjustable='datalim')  # allow free scaling
            self.fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.12)
        self.canvas.draw_idle()

    def on_center_found(self, event):
        self.ax.clear()
        if control.img is not None:
            self.processing = ImageProcessing(control.img)
            binned, factor = self.processing.bin_to_512()
            log_binned = self.processing.log_intensity(binned)
            self.ax.imshow(log_binned)
            from matplotlib.patches import Circle
            # get your stored center and radius
            cx, cy = control.center
            cx_bin, cy_bin = cx/factor, cy/factor
            r = control.radius/factor

            self.ax.scatter(cx_bin, cy_bin, s=5, color='white')
            self.ax.add_patch(Circle((cx_bin, cy_bin), r, fill=False, color='white', linestyle='--'))
            self.ax.text(cx_bin, cy_bin-15, f'{cx:.0f},{cy:.0f}', color='white', fontsize=13,
                        ha='center', va='center')
            self.ax.axis('off')

        self.canvas.draw_idle()
        
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

            # --- Save submenu ---
        save_menu = tk.Menu(open_menu, tearoff=0)
        open_menu.add_cascade(label="Save", menu=save_menu)

        # save_menu.add_command(label="Image as .tif", command=self.loader.save_tif_file)
        # save_menu.add_command(label="Processed data as .csv", command=self.loader.save_csv_file)
        # save_menu.add_separator()
        # save_menu.add_command(label="All outputs", command=self.loader.save_all_outputs)

        # --- Analysis menu ---
        # analysis_menu = tk.Menu(menu_bar, tearoff=0)
        # menu_bar.add_cascade(label="Analysis", menu=analysis_menu)
        # analysis_menu.add_command(label="Find Center", command=self.run_find_center)
        # analysis_menu.add_command(label="Apply Mask", command=self.apply_mask)
        # analysis_menu.add_command(label="Azimuthal Integration", command=self.run_azimuthal_integration)
        # analysis_menu.add_command(label="eRDF Analysis (F(Q), G(r))", command=self.open_erdf_window)



if __name__ == "__main__":
    pyerdf_gui()




