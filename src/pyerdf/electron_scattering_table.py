import h5py
import numpy as np
from diffpy.srreal.scatteringfactortable import ScatteringFactorTable

class SFTElectron(ScatteringFactorTable):
    def __init__(self, datafile="/home/ABTLUS/victor.secco/data_processing/packages/electron_scattering_data.h5"):
        super().__init__()
        self.datafile = datafile
        self._load_data()

    def _load_data(self):
        """Load scattering data from the HDF5 file."""
        self.scattering_data = {}
        with h5py.File(self.datafile, "r") as hdf:
            for element in hdf.keys():
                Q_values = hdf[element]["Q_values"][:]
                fe_values = hdf[element]["f_e_values"][:]
                self.scattering_data[element] = (Q_values, fe_values)

    def _standardLookup(self, smbl, Q):
        """Lookup scattering factor for a given element and Q."""
        if smbl not in self.scattering_data:
            raise ValueError(f"Scattering data not available for element: {smbl}")
        Q_values, fe_values = self.scattering_data[smbl]
        return np.interp(Q, Q_values, fe_values)

    def radiationType(self):
        """Identify this as electron scattering."""
        return "e"

    def type(self):
        """Return the type string for this scattering table."""
        return "SFTElectron"