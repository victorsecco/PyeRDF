#@title 1.1. Classes e funções
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

class DataProcessor:
    def __init__(self, data, lobato_path, start, end, ds, Elements, region):
        # Assuming 'data_df' is a DataFrame with multiple columns of data
        self.data = data
        self.start = start
        self.end = end
        self.ds = ds
        self.lobato = lobato_path
        self.region = region

        total = []
        for element in Elements:
            if Elements[element]:
                total.append(Elements[element][1])
        soma = sum(total)

        for element in Elements:
            if Elements[element]:
                Elements[element].append(Elements[element][1] / soma)
        self.Elements = Elements

        # Load and process data in the constructor
        self.x, self.iq, self.q, self.s, self.s2 = self.load_and_process_data(data)
        self.lobato_factors = self.calculate_Lobato_Factors()
        self.fq_sq, self.gq, self.fqfit, self.iqfit = self.calculate_fq_gq()
        self.N, self.C, self.autofit = self.calculate_N_and_parameters(region=self.region)

    def load_and_process_data(self, data_column):
        # Modify this method to process a single column of data
        # 'data_column' is a pandas Series representing one column of your data
        Iq = np.array(data_column)
        x = np.arange(0, len(Iq), 1)
        x, iq = x[self.start:self.end], Iq[self.start:self.end]
        q = x * self.ds * 2 * math.pi
        s = q / (2 * math.pi)
        s2 = s ** 2
        return x, iq, q, s, s2

    def calculate_Lobato_Factors(self):
        FACTORS = []
        Lobato_Factors = np.empty(shape=(0))
        df = pd.read_csv(self.lobato, header=None)
        counter = 0
        for element in self.Elements:
            if self.Elements[element]:
                FACTORS.append(np.array(df.iloc[self.Elements[element][0]]))

        for LF in FACTORS:
            Lobato_Factors = np.append(Lobato_Factors,
                                       (((LF[0] * (self.s2 * LF[5] + 2) / (self.s2 * LF[5] + 1) ** 2)) +
                                        ((LF[1] * (self.s2 * LF[6] + 2) / (self.s2 * LF[6] + 1) ** 2)) +
                                        ((LF[2] * (self.s2 * LF[7] + 2) / (self.s2 * LF[7] + 1) ** 2)) +
                                        ((LF[3] * (self.s2 * LF[8] + 2) / (self.s2 * LF[8] + 1) ** 2)) +
                                        ((LF[4] * (self.s2 * LF[9] + 2) / (self.s2 * LF[9] + 1) ** 2))))
        Lobato_Factors = Lobato_Factors.reshape(len(FACTORS), len(self.x))

        return Lobato_Factors

    def calculate_fq_gq(self):
        fq = np.empty(shape=(0))
        gq = np.empty(shape=(0))


        for i in range(0, len(self.lobato_factors)):
            if self.Elements[i + 1]:
                fq = np.append(fq, self.lobato_factors[i] * self.Elements[i + 1][2])
                gq = np.append(gq, (self.lobato_factors[i] ** 2) * self.Elements[i + 1][2])

        fq_sq = np.sum(fq.reshape(len(self.Elements), len(self.x)), axis=0)
        fq_sq = fq_sq ** 2
        gq = np.sum(gq.reshape(len(self.Elements), len(self.x)), axis=0)
        fqfit = gq[self.end - (self.start+1)]
        iqfit = self.iq[self.end - (self.start+1)]

        return fq_sq, gq, fqfit, iqfit

    def calculate_N_and_parameters(self, region=0):
        interval = int(region*len(self.x))
        wi = np.ones_like(self.x[interval:])

        a1 = np.sum(self.gq[interval:] * self.iq[interval:])
        a2 = np.sum(self.iq[interval:] * self.fqfit)
        a3 = np.sum(self.gq[interval:] * self.iqfit)
        a4 = np.sum(wi[interval:]) * self.fqfit * self.iqfit
        a5 = np.sum(self.gq[interval:] ** 2)
        a6 = 2 * np.sum(self.gq[interval:]) * self.fqfit
        a7 = np.sum(wi[interval:]) * self.fqfit * self.fqfit

        N = (a1 - a2 - a3 + a4) / (a5 - a6 + a7)

        # Fitting Parameters
        C = self.iqfit - N * self.fqfit
        autofit = N * self.gq + C

        return N, C, autofit

    def calculate_SQ_PhiQ(self, iq, damping):
        sq = (((iq - self.autofit)) / (self.N * self.fq_sq)) + 1
        fq = (((iq - self.autofit) * self.s) / (self.N * self.fq_sq)) * np.exp(-self.s2 * damping)

        return sq, fq

    def calculate_Gr(self, fq, rmax, dr):
        dr = round(rmax/(self.end-self.start),4)
        Gr = np.zeros(self.end-self.start)
        r = np.arange(0, rmax+2*dr, dr)
        for i, r_step in enumerate(r):
            integrand = 8 * math.pi * fq * np.sin(self.q * r_step)
            Gr[i] = np.trapz(integrand, self.s)

        return r, Gr

    def calculate_Gr_Lorch(self, fq, rmax, dr, a, b):
        Gr = np.zeros_like(self.q)
        r = np.linspace(dr, rmax, self.end-self.start)
        for i, r_step in enumerate(r):
            delta = (math.pi/self.q.max()) * (1-np.exp(-abs(r_step-a)/b))
            lorch = np.sin(self.q * delta)/(self.q * delta)
            integrand = 8 * lorch * math.pi * fq * np.sin(self.q * r_step)
            Gr[i] = np.trapz(integrand, self.s)

        return r, Gr

    def calculate_Gr_Lorch_arctan(self, fq, rmax, dr, a, b, c):
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
        number_density_line = -4 * math.pi * nd * r * scale_factor
        Gr_low_r = np.where(r < r_cut, Gr, 0)
        Gr = np.where(r < r_cut, number_density_line, Gr)
        return Gr, Gr_low_r


    def cut_Gr_spherical(self, Gr, r_values, diameter):
        Gr = Gr * (1 - ((3 / 2) * (r_values / diameter)) + (0.5 * (r_values / diameter) ** 3)) * np.exp(-((r_values * 0.2 ** 2) / 2))
        return Gr

    def inverse_fourier_transform(self, Gr, r_values, fq_direct, density):
        # Initialize S(Q) to zeros
        fq_inverse = np.zeros_like(self.q)

        # Perform the inverse Fourier transform
        for i, dq in enumerate(self.q):
            integrand = Gr * np.sin(dq * r_values)
            fq_inverse[i] = np.trapz(integrand, r_values)
        #fq_inverse = fq_inverse*normalization_factor
        #if sum(fq_direct) != 0:
        #    fq_inverse = sum(fq_inverse) / sum(fq_direct) * fq_inverse
        #    return fq_inverse
        #else:
        return fq_inverse

    def calculate_IQ(self, fq, damping):
        iq = fq * self.N * self.fq_sq
        iq = iq / (self.s*np.exp(-self.s2*damping))
        iq = iq + self.autofit
        return iq

    def plot_results(self, fq, fq2, Gr0, r, Gr1, rw):
        f, ax = plt.subplots(1, 3, figsize=(14, 5))

        # Plotting I(Q) and Fit
        line1, = ax[0].plot(self.s, self.autofit)
        line2, = ax[0].plot(self.s, self.iq)
        ax[0].legend([line1, line2], ["Fit", "I(Q) $Fe_{3}O_{4}$"])
        #ax[0].text(5, 600000, 'N: ' f'{int(self.N)}')
        ax[0].set_xlabel("Q ($\AA^{-1}$)")
        ax[0].set_ylabel("Intensity")
        ax[0].title.set_text('Fitting I(Q)')

        # Plotting S(Q)
        line3, = ax[1].plot(self.q, fq, label = "S(Q)" )
        line4, = ax[1].plot(self.q, fq2, label = "S(Q) filtered")
        ax[1].set_xlabel("Q ($\AA^{-1}$)")
        ax[1].set_ylabel("$\S(Q)$")
        #ax[1].set_xlim(11.5,13.5)
        #ax[1].set_ylim(0.4,0.6)
        ax[1].legend()
        ax[1].title.set_text('Calculating S(Q)')

        # Plotting G(r)
        line5, = ax[2].plot(r, Gr0, label = "G(r)")
        line6, = ax[2].plot(r, Gr1, label = "G(r) fit")
        ax[2].text(5, 0.8*Gr0.max(), 'Rw = 'f'{rw:.2f}')
        ax[2].set_xlabel("r ($\AA$)")
        ax[2].set_ylabel("G(r)")
        ax[2].set_xlim([0, 10])
        ax[2].title.set_text('Calculating G(r)')
        ax[2].legend()
        plt.subplots_adjust(hspace=1)
        f.tight_layout()
        #plt.savefig("Fe3O4 IQ", dpi=300)


        plt.show()


    def save_to_csv(self, data, file_path, name, separator, x, y, out='pdfgui'):
        if os.path.isfile(file_path):
    # Get the directory part of the file path
          directory = os.path.dirname(file_path)
          full_path = os.path.join(directory, "Gr")
          if not os.path.exists(full_path):
            os.makedirs(full_path)
          data = pd.DataFrame(np.transpose(np.array(data)))
          final_path = os.path.join(full_path, name)
          if out == 'discus':
            data.rename(columns={0:x,1:y},inplace=True)
            data['d'f'{x}']=data[x]*0
            data['d'f'{y}']=abs(data[y]/20)
            data.to_csv(f'{final_path}.csv', sep=separator,float_format="%.10f", index='r')
          else:
            data.to_csv(f'{final_path}.csv', sep=separator, index=False, header=False)
          return None

def calculate_rw(obs, calc):
  return math.sqrt(sum((obs-calc)**2)/sum(obs**2))




def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs  # fs = sample rate, Hz
    normal_cutoff = cutoff / nyq # desired cutoff frequency of the filter, Hz, slightly higher than actual 1.2 Hz / Nyquist Frequency

    # Get the filter coefficients
    b, a = butter(order, # sin wave can be approx represented as quadratic
                  normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

