import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
from pathlib import Path

class DataProcessor:
    def __init__(self, lobato_path = False):
  
        #path to electron scattering factors table
        default_lobato_path = Path(__file__).parent / "data" / "Lobato_2014.txt"
        self.lobato = Path(lobato_path) if lobato_path else default_lobato_path

    def load_and_process_data(self, data=None, *, start=None, end=None, ds=None, q0=None):
        Iq = np.array(self.data if data is None else data)
        self.start = self.start if start is None else start
        self.end = self.end if end is None else end
        self.ds = self.ds if ds is None else ds
        self.q0 = self.q0 if q0 is None else q0

        self.x = np.arange(self.start, self.end)
        self.iq = Iq[self.start:self.end]
        self.build_s_range()
        self.build_q_range()
        return self.x, self.iq, self.q, self.s, self.s2
    
    def build_q_range(self):
        self.q = self.q0 + self.x * self.ds * 2 * math.pi

    def build_s_range(self, ds = None, arr_size = None):
        if hasattr(self, "ds"):
            self.s = self.x * self.ds
        else:
            self.x = np.arange(0, arr_size)
            self.s = self.x * ds
        self.s2 = self.s ** 2

    def Lobato_Factors(self, *, lobato_path=None, elements=None, s2=None):
        self.lobato = Path(self.lobato if lobato_path is None else lobato_path)
        self.Elements = self.Elements if elements is None else elements
        self.s2 = self.s2 if s2 is None else np.asarray(s2)

        # normalize in place
        normalize_elements_inplace(self.Elements)

        FACTORS = []
        Lobato_Factors = np.empty(shape=(0))
        df = pd.read_csv(self.lobato, header=None)

        for element in self.Elements:
            if self.Elements[element]:
                FACTORS.append(np.array(df.iloc[self.Elements[element][0]]))

        for LF in FACTORS:
            Lobato_Factors = np.append(
                Lobato_Factors,
                ((LF[0] * (self.s2 * LF[5] + 2) / (self.s2 * LF[5] + 1) ** 2) +
                (LF[1] * (self.s2 * LF[6] + 2) / (self.s2 * LF[6] + 1) ** 2) +
                (LF[2] * (self.s2 * LF[7] + 2) / (self.s2 * LF[7] + 1) ** 2) +
                (LF[3] * (self.s2 * LF[8] + 2) / (self.s2 * LF[8] + 1) ** 2) +
                (LF[4] * (self.s2 * LF[9] + 2) / (self.s2 * LF[9] + 1) ** 2))
            )

        self.lobato_factors = Lobato_Factors.reshape(len(FACTORS), len(self.s2))
        return self.lobato_factors

    def compute_weighted_factors(self):
        """
        Compute weighted scattering-factor moments over Q.

        Attributes set
        --------------
        self.fbar_sq : array
            [Σ_i w_i f_i(Q)]^2 — square of the weighted-average scattering factor.
        self.mean_f2 : array
            Σ_i w_i [f_i(Q)]^2 — weighted average of squared scattering factors.
        self.fbar_sq_ref : float
            fbar_sq at the reference index (last point of the processed range).
        self.iq_ref : float
            I(Q) at the same reference index.

        Returns
        -------
        fbar_sq, mean_f2, fbar_sq_ref, iq_ref
        """
        f_terms = []
        f2_terms = []

        self.w = np.array([self.Elements[elem][2] for elem in self.Elements], float)
        f = np.asarray(self.lobato_factors, float)  # shape (n_elem, n_Q)

        f_terms = self.w[:, None] * f
        f2_terms = self.w[:, None] * (f ** 2)


        fbar = np.sum(f_terms, axis=0)             # Σ_i w_i f_i(Q)
        self.fbar_sq = fbar ** 2                   # [Σ_i w_i f_i(Q)]^2
        self.mean_f2 = np.sum(f2_terms, axis=0)    # Σ_i w_i f_i(Q)^2

        ref_idx = -1
        self.fbar_sq_ref = self.fbar_sq[ref_idx]
        self.mean_f2_ref = self.mean_f2[ref_idx]

        self.iq_ref = self.iq[ref_idx]

        return self.fbar_sq, self.mean_f2

    def N_and_parameters(self, region=0):
        """
        Estimate scaling constant N and background parameter C by
        fitting the experimental I(Q) against calculated scattering
        moments.

        Parameters
        ----------
        region : float
            Fraction (0–1) of the Q range from which to start the fit.

        Returns
        -------
        N : float
            Scaling factor for the scattering.
        C : float
            Constant background offset.
        autofit : array
            Fitted I(Q) curve from the model.
        """
    
        interval = int(region * len(self.x))
        wi = np.ones_like(self.x[interval:])

        a1 = np.sum(self.mean_f2[interval:] * self.iq[interval:])
        a2 = np.sum(self.iq[interval:] * self.fbar_sq_ref)
        a3 = np.sum(self.mean_f2[interval:] * self.iq_ref)
        a4 = np.sum(wi[interval:]) * self.fbar_sq_ref * self.iq_ref
        a5 = np.sum(self.mean_f2[interval:] ** 2)
        a6 = 2 * np.sum(self.mean_f2[interval:]) * self.fbar_sq_ref
        a7 = np.sum(wi[interval:]) * self.fbar_sq_ref ** 2

        self.N = (a1 - a2 - a3 + a4) / (a5 - a6 + a7)


        
        # Fitting parameters
        self.C = self.iq_ref - self.N * self.mean_f2_ref
        self.autofit = self.N * self.mean_f2 + self.C

        return self.N, self.C, self.autofit
    
    def diffuse_sc(self, B_list):

        B_arr = np.asarray(B_list, dtype=float)
        u_arr = B_arr / (8 * np.pi**2)
        wu_arr = u_arr*self.w
        u2 = wu_arr.mean()
        self.diffuse_scat = np.exp(-(u2 * self.q**2))*(self.mean_f2/self.fbar_sq_ref)


    def sq_fq(self, iq, damping):
        """
        Compute reduced structure function S(Q) and total scattering function F(Q) from intensities.

        Attributes set
        --------------
        self.sq : array
            Structure function S(Q).
        self.fq : array
            Total scattering function F(Q).

        Parameters
        ----------
        iq : array
            Experimental intensity I(Q).
        damping : float
            Damping factor applied to F(Q) at high Q.

        Returns
        -------
        sq : array
            Structure function S(Q).
        fq : array
            Total scattering function F(Q).
        """
        numerator = iq - self.autofit
        self.sq = (numerator / (self.N * self.fbar_sq)) + 1
        self.fq = (numerator * self.s / (self.N * self.fbar_sq)) * np.exp(-self.s2 * damping)

        return self.sq, self.fq
    
    def Gr(self, fq, rmax, dr):
        Gr = []
        r = np.arange(0, rmax, dr)

        for r_step in r:
            integrand = 8 * math.pi * fq * np.sin(self.q * r_step)
            Gr.append(np.trapezoid(integrand, self.q/(2* np.pi)))
        
        r = np.array(r, dtype=np.float64)
        Gr = np.array(Gr, dtype=np.float64)

        return r, Gr/(2 * np.pi)

    def Gr_Lorch(self, fq, rmax, dr):
        """
        Implements the Lorch correction as originally defined in Lorch (1969).
        """
        r = np.arange(0, rmax, dr)
        Gr = np.zeros_like(r)

        # Delta = 2π / Q_max
        delta = 2 * np.pi / self.q.max()

        # Lorch window: sinc(Q * delta / 2)
        lorch = np.sinc((self.q * delta) / (2 * np.pi))  # sinc(x) = sin(πx)/(πx)

        fq_mod = fq * lorch  # Apply Lorch window to F(Q)

        for i, r_step in enumerate(r):
            integrand = 8 * np.pi * fq_mod * np.sin(self.q * r_step)
            Gr[i] = np.trapezoid(integrand, self.q / (2 * np.pi))  # Integration over q in Å⁻¹

        return r, Gr


    def Gr_Lorch_arctan(self, fq, rmax, dr, a, b, c):
        Gr = np.zeros_like(self.q)
        r = np.linspace(dr, rmax, self.end-self.start)
        for i, r_step in enumerate(r):
            delta = (math.pi / self.q.max()) * (
                    (1 - np.exp(-abs(r_step - a) / b)) +
                    (1 / 2 + 1 / math.pi * np.arctan(r_step - c / (c / (2 * math.pi))) * r_step ** (1 / 2))
                    )
            lorch = np.sin(self.q * delta)/(self.q * delta)
            integrand = 8 * lorch * math.pi * fq * np.sin(self.q * r_step)
            Gr[i] = np.trapz(integrand, self.s)

        return r, Gr

    def low_r_correction(self, Gr, nd, r, r_cut, scale_factor = 1):
        #empirical low-r G(r) correction

        number_density_line = -4 * math.pi * nd * r * scale_factor
        Gr_low_r = np.where(r < r_cut, Gr, 0)
        Gr = np.where(r < r_cut, number_density_line, Gr)
        return Gr, Gr_low_r


    def cut_Gr_spherical(self, Gr, r_values, diameter):
        Gr = Gr * (1 - ((3 / 2) * (r_values / diameter)) + (0.5 * (r_values / diameter) ** 3)) * np.exp(-((r_values * 0.2 ** 2) / 2))
        return Gr

    def inverse_fourier_transform(self, r_values, Gr):
        Gr = np.asarray(Gr, dtype=np.float64)
        r = np.asarray(r_values, dtype=np.float64)
        if Gr.shape != r.shape:
            raise ValueError("Gr and r_values must have the same shape.")
        Q = np.asarray(self.q, dtype=np.float64)
        sin_qr = np.sin(np.outer(Q, r))
        fq_inverse = np.trapezoid(Gr * sin_qr, x=r, axis=1)
        return fq_inverse


    def IQ(self, fq, damping):
        """
        Reverse calculation of the total scattering intensity based on the calculated fq
        
        Parameters:
        - fq: inverse calculated fq after gr corrections
        - damping: damping parameter used to make high-q signal less expressive
        
        Returns:
        - iq: recalculated total scattering intensity
        """
        iq = fq * self.N * self.fq_sq
        iq = iq / (self.s*np.exp(-self.s2*damping))
        iq = iq + self.autofit
        return iq

    def plot_results(self, fq, r, Gr0):
        plt.ion()  # interactive mode ON
        plt.close('all')  # close any previous figures

        f, ax = plt.subplots(1, 3, figsize=(14, 5))

        # I(Q) vs Fit
        ax[0].plot(self.q, self.autofit, label="Fit")
        ax[0].plot(self.q, self.iq, label="I(Q)")
        ax[0].set_xlabel(r"Q ($\AA^{-1}$)")
        ax[0].set_ylabel("Intensity")
        ax[0].legend()
        ax[0].set_title("Fitting I(Q)")

        # F(Q)
        ax[1].plot(self.q, fq, label=r"F(Q)$")
        ax[1].set_xlabel(r"Q ($\AA^{-1}$)")
        ax[1].set_ylabel(r"F(Q)$")
        ax[1].set_title(r"Calculating F(Q)$")
        ax[1].legend()

        # G(r)
        ax[2].plot(r, Gr0, label="G(r)")
        ax[2].set_xlim([0, 30])
        ax[2].set_xlabel(r"r ($\AA$)")
        ax[2].set_ylabel("G(r)")
        ax[2].set_title("Calculating G(r)")
        ax[2].legend()

        f.tight_layout()
        plt.draw()
        plt.pause(0.001) 


    def save_to_csv(self, data, file_path, separator, x_name, y_name, out='pdfgui'):
        """
        Saves selected data to .csv format

        Parameters:
        - data: tuple of data comprising x,y 
        - file_path: folder in which the file will be saved
        - name: the base name of the output file
        - x, y: names of the columns in the .csv file 
        - separator: delimiter for the CSV output
        - out: formatting the file for input in the discus of pdfgui softwares
        """
        data = pd.DataFrame(np.transpose(np.array(data)))
        if out == 'discus':
            data.rename(columns={0: x_name, 1: y_name}, inplace=True)
            data[f'd{x}'] = data[x] * 0
            data[f'd{y}'] = abs(data[y] / 20)
            data.to_csv(f'{file_path}.csv', sep=separator, float_format="%.10f", index=False)
        else:
            data.to_csv(f'{file_path}.csv', sep=separator, index=False, header=False)

def Gr(q, fq, rmax, dr):
        Gr = []
        r = np.arange(0, rmax, dr)

        for i, r_step in enumerate(r):
            integrand = 8 * math.pi * fq * np.sin(q * r_step)
            Gr.append(np.trapz(integrand, q/(2* np.pi)))
        # Convert lists to numpy arrays for consistency
        r = np.array(r, dtype=np.float64)
        Gr = np.array(Gr, dtype=np.float64)

        return r, Gr/(2 * np.pi)


def calc_Gr_Lorch(q, fq, rmax, dr, rmin=10, transition_width=5):
    """
    Computes G(r) with a Lorch window applied only after a certain rmin.
    Before rmin, the raw F(q) is used. A smooth transition is applied over
    `transition_width` Å.

    Parameters:
    - q: array of q values (Å⁻¹)
    - fq: array of F(q)
    - rmax: maximum r to calculate (Å)
    - dr: r-step (Å)
    - rmin: r below which Lorch is not applied
    - transition_width: width of transition zone for blending (Å)
    """
    r = np.arange(0, rmax, dr)
    Gr = np.zeros_like(r)

    delta = 2 * np.pi / np.max(q)
    lorch_window = np.sinc((q * delta) / (2 * np.pi))
    fq_lorch = fq * lorch_window

    for i, r_step in enumerate(r):
        # Compute transition blending weight
        if r_step < rmin:
            w = 0.0
        elif r_step > rmin + transition_width:
            w = 1.0
        else:
            # Smooth sigmoid transition
            t = (r_step - rmin) / transition_width
            w = 3*t**2 - 2*t**3  # cubic Hermite smoothstep

        fq_blend = (1 - w) * fq + w * fq_lorch

        integrand = 8 * np.pi * fq_blend * np.sin(q * r_step)
        Gr[i] = np.trapz(integrand, q / (2 * np.pi))

    return r, Gr/(2 * np.pi)


def normalize_elements_inplace(Elements):
    mults = [v[1] for v in Elements.values() if v and len(v) >= 2]
    s = float(sum(mults))
    if s == 0:
        raise ValueError("Sum of multiplicities is zero.")
    for k, v in Elements.items():
        if v and len(v) >= 2:
            w = v[1] / s
            if len(v) >= 3:
                v[2] = w         # overwrite existing weight
            else:
                v.append(w)

def q_to_two_theta(q_calibration, pixel_data, wavelength_nm):
    """
    Converts pixel data from Q space to two-theta angles using a given Q space calibration factor.
    
    Parameters:
    - q_calibration: The Q space calibration factor.
    - pixel_data: An array of pixel indices or measurements to be converted.
    - wavelength_nm: The wavelength of the X-rays in nanometers.
    
    Returns:
    - two_theta: An array of two-theta angles in degrees corresponding to the provided pixel data.
    """
    # Convert the wavelength to the same unit as q_calibration, if needed (nm to the unit of q_calibration)
    wavelength = wavelength_nm  # Assuming q_calibration is based on nm units
    
    # Calculate 1/d from pixel data using the Q space calibration factor
    one_over_d = pixel_data * (q_calibration / (2 * np.pi))
    
    # Calculate two-theta angles from 1/d values
    two_theta = 2 * np.degrees(np.arcsin(wavelength * one_over_d / 2))
    
    return two_theta


def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs  # fs = sample rate, Hz
    normal_cutoff = cutoff / nyq # desired cutoff frequency of the filter, Hz, slightly higher than actual 1.2 Hz / Nyquist Frequency

    # Get the filter coefficients
    b, a = butter(order, # sin wave can be approx represented as quadratic
                  normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def read_discus_fit_file(path):
    # Read file ignoring comment lines (those starting with '#')
    with open(path, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#') and line.strip() != '']
    
    # Split lines by whitespace and remove empty tokens
    data = [remove_empty_strings(line.strip().split()) for line in lines]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Assign columns depending on how many there are
    if df.shape[1] == 4:
        df.columns = ['r', 'gr', 'dr', 'dgr']
    elif df.shape[1] == 2:
        df.columns = ['r', 'gr']
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")
    
    return df.astype(float)

def rw(obs, calc, scaling = 1):
    #Calculate residuals metric from experimental and calculated data
    obs = obs * scaling
    return math.sqrt(sum((obs-calc)**2)/sum(obs**2))

def remove_empty_strings(lst):
    return [element for element in lst if element != ""]

# The wrapper function for minimize
def optimize_constant(grob, calc, initial_guess=1):
    # Objective function to minimize, takes only constant as argument
    objective = lambda constant: rw(grob, calc, constant)
    # Run the optimization
    result = minimize(objective, initial_guess)
    return result.x  # This returns the optimized constant