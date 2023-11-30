"""
This module includes the LES class and the DHARMA sub-class
"""
import xarray as xr
import numpy as np
import os

class LES():
    """
    Model namelists and unit conversion coefficient required for the 1D model.
    The LES class includes methods to processes model output and prepare the out fields for the 1D model.

    Attributes
    ----------
    Ni_field: dict
        Ice number concentration fieldname and scaling factor required for m^-3 units.
        If fieldname is a list, summing the list fieldname values.
    pflux_field: dict
        Precipitation flux fieldname and scaling factor for mm/h.
        If fieldname is a list, summing the list fieldname values.
    T_field: dict
        Temperature field name and addition factor in case of T reported in C (K is needed).
    q_liq_field: dict
        Liquid mixing ratio fieldname and scaling factor for kg/kg.
    RH_field: dict
        RH fieldname and scaling factor for fraction units (not %).
    model_name: str
        name of model.
    time_dim: str
        name of the time dimension.
    height_dim: str
        name of the height dim.
    height_dim_2nd: str
        name of the height dim for grid cell edge coordinates.
    q_liq_pbl_cut: float
        value of q_liq cut (kg/kg) required to define the PBL top z-index (where LWC becomes negligible) -
        to be used in '_crop_fields' if 'height_ind_2crop' == 'ql_pbl'.
    q_liq_cbh: float
        Threshold value (kg/kg) for cloud base height defined using q_liq.
    """
    def __init__(self, q_liq_pbl_cut=None):
        self.Ni_field = {"name": None, "scaling": None}  # scale to m^-3
        self.pflux_field = {"name": None, "scaling": None}  # scale to mm/h
        self.T_field = {"name": None, "addition": None}  # scale to K
        self.q_liq_field = {"name": None, "scaling": None}  # scale to kg/kg
        self.RH_field = {"name": None, "scaling": None}  # scale to fraction
        self.rho_field = {"name": None, "scaling": None}  # scale to fraction
        self.model_name = ""
        self.time_dim = ""  # assuming in seconds
        self.height_dim = ""  # assuming in m
        self.height_dim_2nd = ""  # assuming in m
        if q_liq_pbl_cut is None:
            self.q_liq_pbl_cut = 1e-6  # kg/kg (default value of 1e-6).
        else:
            self.q_liq_pbl_cut = q_liq_pbl_cut
        self.q_liq_cbh = self.q_liq_pbl_cut  # Equal to the pbl cutoff value by default

        self.les_data_path = os.path.join(os.path.dirname(__file__), 'data_les')

    def _crop_time_range(self, t_harvest=None):
        """
        Crop model output time range.

        Parameters
        ----------
        t_harvest: scalar, 2- or 3-element tuple, list (or ndarray), or None
            If scalar then using the nearest time (assuming units of seconds) to initialize the model
            (single profile).
            If a tuple, cropping the range defined by the first two elements (increasing values) using a
            slice object. If len(t_harvest) == 3 then using the 3rd element as a time offset to subtract from
            the tiem array values.
            If a list, cropping the times specified in the list (can be used take LES output profiles every
            delta_t seconds.
        """
        if isinstance(t_harvest, (float, int)):
            self.ds = self.ds.sel({self.time_dim: [t_harvest]}, method='nearest')
        elif isinstance(t_harvest, tuple):  # assuming a 2-element tuple.
            if np.logical_or(len(t_harvest) < 2, len(t_harvest) > 3):
                raise ValueError("t_harvest (time range) tuple length should be 2 or 3")
            elif len(t_harvest) == 2:
                self.ds = self.ds.sel({self.time_dim: slice(*t_harvest)})
            elif len(t_harvest) == 3:
                self.ds = self.ds.sel({self.time_dim: slice(*t_harvest[:2])})
                self.ds = self.ds.assign_coords({self.time_dim: self.ds[self.time_dim].values - t_harvest[2]})
        elif isinstance(t_harvest, (list, np.ndarray)):
            self.ds = self.ds.sel({self.time_dim: t_harvest}, method='nearest')

    def _crop_fields(self, fields_to_retain=None, height_ind_2crop=None):
        """
        Crop the required fields (and other requested fields), with the option of cropping the
        height dim using specified indices.
        If multiple Ni or pflux fields are specified then those fields are summed.

        Parameters
        ----------
        fields_to_retain: list or None
            Fieldnames to crop from the LES output (required to properly run the model).
            If None, then cropping the minimum number of required fields using the model's namelist convention
            (Temperature, q_liq, RH, precipitation flux, and ice number concentration).
        height_ind_2crop: list, np.ndarray, float, int, str, or None
            Indices of heights (in a list or np.ndarray) to crop from the model output (e.g., up to the PBL top).
            if float then indicating then the value indicates the cropped domain-top height.
            if int then indicating the index of the domain-top height.
            if str then different defitions for PBL:
                - if == "ql_pbl" then cropping all values within the PBL defined here based on the
                'q_liq_pbl_cut' attribute. If more than a single time step exist in the dataset, then cropping
                the highest index corresponding to the cutoff.
                - OTHER OPTIONS TO BE ADDED.
            If None then not cropping.
        """
        if isinstance(self.Ni_field["name"], list):
            self.ds["Ni_tot"] = self.ds[self.Ni_field["name"][0]].copy(deep=True)
            if len(self.Ni_field["name"]) > 1:
                for ff in self.Ni_field["name"]:
                    self.ds["Ni_tot"].values += self.ds[ff].values
            self.Ni_field["name"] = "Ni_tot"

        if isinstance(self.pflux_field["name"], list):
            self.ds["PFtot"] = xr.DataArray(np.zeros_like(self.ds[self.pflux_field["name"][0]].values),
                                            dims=self.ds[self.pflux_field["name"][0]].dims)
            for ff in self.pflux_field["name"]:
                self.ds["PFtot"].values += self.ds[ff].values
            self.pflux_field["name"] = "PFtot"
            if not self.les_bin_phys:  # Requires Arakawa C-grid consideration (taking precip at bottim of cell).
                self.ds[self.pflux_field["name"]] = \
                    xr.DataArray(self.ds[self.pflux_field["name"]].values[:, :-1].T,
                                 dims=self.ds[self.Ni_field["name"]].dims)

        if fields_to_retain is None:
            fields_to_retain = [self.Ni_field["name"], self.pflux_field["name"], self.T_field["name"],
                                self.q_liq_field["name"], self.RH_field["name"], self.rho_field["name"]]

        # crop variables
        self.ds = self.ds[fields_to_retain]  # retain variables needed.

        # crop heights
        if height_ind_2crop is not None:
            if isinstance(height_ind_2crop, str):
                if height_ind_2crop == "ql_pbl":  # 1D model domain extends only to PBL top
                    rel_inds = np.arange(np.max(np.where(
                                         self.ds[self.q_liq_field["name"]].values >= self.q_liq_pbl_cut /
                                         self.q_liq_field["scaling"])[0]) + 1)
                else:
                    print("Unknown croppoing method string - skipping xr dataset (LES domain) cropping.")
            elif isinstance(height_ind_2crop, float):
                rel_inds = np.arange(np.argmin(np.abs(height_ind_2crop -
                                     self.ds[self.height_dim].values)) + 1)
            elif isinstance(height_ind_2crop, int):
                rel_inds = np.arange(height_ind_2crop + 1)
            elif isinstance(height_ind_2crop, (list, np.ndarray)):
                rel_inds = height_ind_2crop
            self.ds = self.ds[{self.height_dim: rel_inds}]

    def _prepare_les_dataset_for_1d_model(self, cbh_det_method="ql_thresh"):
        """
        scale (unit conversion), rename the required fields (prepare the dataset for informing the 1D model
        allowing retaining only the les xr dataset instead of the full LES class), and calculate some additional.

        Parameters
        ----------
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_thresh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        """
        self.ds = self.ds.rename({self.height_dim: "height", self.time_dim: "time",
                                  self.pflux_field["name"]: "prec", self.Ni_field["name"]: "Ni",
                                  self.T_field["name"]: "T", self.q_liq_field["name"]: "ql",
                                  self.RH_field["name"]: "RH", self.rho_field["name"]: "rho"})

        # scale and convert to float64
        for key in self.ds.keys():
            if self.ds[key].dtype == "float32":
                self.ds[key] = self.ds[key].astype(float)
        self.ds["RH"] *= self.RH_field["scaling"]
        self.ds["ql"] *= self.q_liq_field["scaling"]
        self.ds["T"] += self.T_field["addition"]
        self.ds["Ni"] *= self.Ni_field["scaling"]
        self.ds["prec"] *= self.pflux_field["scaling"]
        self.ds["rho"] *= self.q_liq_field["scaling"]

        # set units
        self.ds["height"].attrs["units"] = "$m$"
        self.ds["time"].attrs["units"] = "$s$"
        self.ds["RH"].attrs["units"] = ""
        self.ds["ql"].attrs["units"] = r"$kg\:kg^{-1}$"
        self.ds["T"].attrs["units"] = "$K$"
        self.ds["Ni"].attrs["units"] = "$m^{-3}$"
        self.ds["prec"].attrs["units"] = r"$mm\:h^{-1}$"
        self.ds["rho"].attrs["units"] = r"$kg\:m^{-3}$"

        # calculate ∆aw field for ABIFM
        self._calc_delta_aw()

        # calculated weighted precip rates
        self._find_and_calc_cb_precip(cbh_det_method)

    def _find_and_calc_cb_precip(self, cbh_det_method="ql_thresh"):
        """
        calculate number-weighted precip rate in the domain and allocate a field for values at lowest cloud base.
        The method finds cbh as well as cth

        Parameters
        ----------
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_thresh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        """
        self.ds["P_Ni"] = self.ds['prec'] / self.ds['Ni']
        self.ds["P_Ni"].attrs['units'] = r'$mm\:h^{-1}\:m^{-3}$'
        self.ds["P_Ni"].attrs['long_name'] = "Ni-normalized precipitation rate"

        # find all cloud bases and the precip rate in the lowest cloud base in every time step (each profile).
        if cbh_det_method == "ql_thresh":
            self.ds["cbh_all"] = xr.DataArray(np.diff(self.ds["ql"].values >= self.q_liq_cbh, prepend=0,
                                                      axis=0) == 1, dims=self.ds["P_Ni"].dims)
            self.ds["cth_all"] = xr.DataArray(np.diff(self.ds["ql"].values >= self.q_liq_cbh, append=0,
                                                      axis=0) == -1, dims=self.ds["P_Ni"].dims)
        else:
            print("Unknown cbh method string - skipping cbh detection function")
            return
        self.ds["cbh_all"].attrs['long_name'] = "All detected cloud base heights (receive a 'True' value)"
        self.ds["cbh_all"].attrs['long_name'] = "All detected cloud top heights (receive a 'True' value)"

        cbh_lowest = np.where(np.logical_and(np.cumsum(self.ds["cbh_all"], axis=0) == 1, self.ds["cbh_all"]))
        cth_lowest = np.where(np.logical_and(np.cumsum(self.ds["cth_all"], axis=0) == 1, self.ds["cth_all"]))
        self.ds["lowest_cbh"] = xr.DataArray(np.zeros(self.ds.dims["time"]) * np.nan, dims=self.ds["time"].dims)
        self.ds["lowest_cbh"].attrs['units'] = '$m$'
        self.ds["lowest_cth"] = self.ds["lowest_cbh"].copy()
        self.ds["lowest_cbh"].attrs['long_name'] = "Lowest cloud base height per profile"
        self.ds["lowest_cth"].attrs['long_name'] = "Lowest cloud top height per profile"
        self.ds["lowest_cbh"][cbh_lowest[1]] = self.ds["height"].values[cbh_lowest[0]]
        self.ds["lowest_cth"][cth_lowest[1]] = self.ds["height"].values[cth_lowest[0]]
        self.ds["Pcb_per_Ni"] = xr.DataArray(np.zeros(self.ds.dims["time"]) * np.nan, dims=self.ds["time"].dims)
        self.ds["Pcb_per_Ni"][cbh_lowest[1]] = self.ds["P_Ni"].values[cbh_lowest]
        self.ds["Pcb_per_Ni"].attrs['units'] = r'$mm\:h^{-1}r\:m^{-3}$'
        self.ds["Pcb_per_Ni"].attrs['long_name'] = "Ni-normalized lowest cloud base precipitation rate"

    def _calc_delta_aw(self):
        """
        calculate the ∆aw field for ABIFM
        """
        self.ds["delta_aw"] = self.ds['RH'] - \
            (np.exp(9.550426 - 5723.265 / self.ds['T'] + 3.53068 * np.log(self.ds['T']) - 0.00728332 *
                    self.ds['T']) / (np.exp(54.842763 - 6763.22 / self.ds['T'] -
                                     4.210 * np.log(self.ds['T']) + 0.000367 * self.ds['T'] +
                                     np.tanh(0.0415 * (self.ds['T'] - 218.8)) *
                                     (53.878 - 1331.22 / self.ds['T'] - 9.44523 *
                                      np.log(self.ds['T']) + 0.014025 * self.ds['T']))))
        self.ds['delta_aw'].attrs['units'] = ""


class DHARMA(LES):
    def __init__(self, les_out_path=None, les_out_filename=None, t_harvest=None, fields_to_retain=None,
                 height_ind_2crop=None, cbh_det_method="ql_thresh", q_liq_pbl_cut=None,
                 les_bin_phys=True):
        """
        LES class for DHARMA that loads model output dataset

        Parameters
        ----------
        les_out_path: str or None
            LES output path (can be relative to running directory). Use default if None.
        les_out_filename: str or None
            LES output filename. Use default file if None.
        t_harvest: scalar, 2-element tuple, list (or ndarray), or None
            If scalar then using the nearest time (assuming units of seconds) to initialize the model
            (single profile).
            If a tuple, cropping the range defined by the first two elements (increasing values) using a
            slice object.
            If a list, cropping the times specified in the list (can be used take LES output profiles every
            delta_t seconds.
        fields_to_retain: list or None
            Fieldnames to crop from the LES output (required to properly run the model).
            If None, then cropping the minimum number of required fields using DHARMA's namelist convention
            (Temperature [K], q_liq [kg/kg], RH [fraction], precipitation flux [mm/h], and ice number
            concentration [m^-3]).
        height_ind_2crop: list, str, or None
            Indices of heights to crop from the model output (e.g., up to the PBL top).
            if str then different definitions for PBL:
                - if == "ql_pbl" then cropping all values within the PBL defined here based on the 'q_liq_pbl_cut'
                  attribute. If more than a single time step exist in the dataset, then cropping
                  the highest index corresponding to the cutoff.
                - OTHER OPTIONS TO BE ADDED.
            If None then not cropping.
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_thresh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        q_liq_pbl_cutoff: float
            value of q_liq_cut (kg/kg) required to define the PBL top z-index (where LWC becomes negligible) -
            to be used in '_crop_fields' if 'height_ind_2crop' == 'ql_pbl'.
        les_bin_phys: bool
            IF True, using bin microphysics output namelist for harvesting LES data.
            If False, using bulk microphysics output namelist for harvesting LES data.
        """
        super().__init__(q_liq_pbl_cut=q_liq_pbl_cut)
        if les_bin_phys:
            self.Ni_field = {"name": "ntot_3", "scaling": 1e6}  # scale to m^-3
            self.pflux_field = {"name": "pflux_3", "scaling": 1.}  # scale to mm/h
            self.q_liq_field = {"name": "qc", "scaling": 1.}  # scale to kg/kg
        else:  # bulk
            self.Ni_field = {"name": ["nqic", "nqid", "nqif"], "scaling": 1e6}  # scale to m^-3
            self.pflux_field = {"name": ["PFqic", "PFqid", "PFqif"], "scaling": 1.}  # scale to mm/h
            self.q_liq_field = {"name": "qc", "scaling": 1.}  # scale to kg/kg
        self.T_field = {"name": "T", "addition": 0.}  # scale to K (addition)
        self.RH_field = {"name": "RH", "scaling": 1. / 100.}  # scale to fraction
        self.rho_field = {"name": "rhobar", "scaling": 1.}
        self.model_name = "DHARMA"
        self.time_dim = "time"
        self.height_dim = "zt"
        self.height_dim_2nd = "zw"

        # using the default ISDAC model output if None.
        if les_out_path is None:
            les_out_path = self.les_data_path + '/SHEBA_DHARMA_Baseline/'
        if les_out_filename is None:
            les_out_filename = 'dharma.soundings.cdf'
        self.les_out_path = les_out_path
        self.les_out_filename = les_out_filename
        self.les_bin_phys = les_bin_phys

        # load model output
        self.ds = xr.open_dataset(les_out_path + les_out_filename)
        self.ds = self.ds.transpose(*(self.height_dim, self.time_dim, self.height_dim_2nd))  # validate dim order

        # crop specific model output time range (if requested)
        if t_harvest is not None:
            super()._crop_time_range(t_harvest)

        # crop specific model output fields and height range
        super()._crop_fields(fields_to_retain=fields_to_retain, height_ind_2crop=height_ind_2crop)

        # prepare fieldnames and generate new ones required for the 1D model
        super()._prepare_les_dataset_for_1d_model(cbh_det_method=cbh_det_method)
