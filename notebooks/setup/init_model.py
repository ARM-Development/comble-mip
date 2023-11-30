"""
This module is used to initialize the model, and allocate fields and arrays to a 'ci_model' class.
"""
import xarray as xr
import numpy as np
from time import time
import LES
import AER
import plotting
import copy
from run_model import run_model as Run
import pint


class ci_model():
    """
    Cloud-ice nucleation 1D model class containing:
    1. All initialization model parameters
    2. LES output dataset used to initialize and inform the model (ci_model.les).
    3. Model output  output fields (ci_model.ds).
    """
    def __init__(self, final_t=21600, delta_t=10, use_ABIFM=True, les_name="DHARMA", t_averaged_les=True,
                 custom_vert_grid=None, w_e_ent=1e-3, entrain_to_cth=True,
                 implicit_ent=True, tau_mix=1800., heat_rate=None, tau_act=10., implicit_act=True,
                 implicit_sublim=True, mixing_bounds=None, v_f_ice=0.3, in_cld_q_thresh=1e-6,
                 nuc_RH_thresh=None, time_splitting=True, ent_then_act=True,
                 prognostic_inp=True, prognostic_ice=False, dt_out=None, relative_sublim=True,
                 aer_info=None, les_out_path=None, les_out_filename=None, les_bin_phys=True, t_harvest=10800,
                 fields_to_retain=None, height_ind_2crop="ql_pbl", cbh_det_method="ql_thresh",
                 input_conc_units=None, input_diam_units=None, input_heatrate_units=None,
                 do_act=True, do_entrain=True, do_mix_aer=True, do_mix_ice=True, do_sedim=True,
                 do_sublim=False, output_budgets=False, output_aer_decay=True, run_model=True):
        """
        Model namelists and unit conversion coefficient required for the 1D model.
        The LES class includes methods to processes model output and prepare the out fields for the 1D model.
        This method also initializes the aerosol populations and runs the model.

        Parameters
        ----------
        final_t: float
            Total simulation time [s].
        delta_t: float
            time_step [s].
        use_ABIFM: bool
            True - use ABIFM, False - use singular.
        les_name: str
            Name of LES model to harvest data from.
        t_averaged_les: bool
            If True, use time-averaged LES profile of each variable to inform the 1D model.
            If False then the 1D model is informed by the LES output temporal evolution with extrapolation
            outside the LES output DataSet time range.
            Note: in the case of a single LES output time step requested ('t_harvest' is a scalar), this boolean
            has no effect.
        custom_vert_grid: list, np.ndarray, or None.
            custom vertical grid for the 1D model. If None, then using the processed (and cropped) LES output
            grid.
        w_e_ent: dict or float
            cloud-top entrainment rate [m/s].
            if a float then using its value throughout the simulation time.
            if a dict, must have the keys "time" [s] and "value". Each key contains a list or np.ndarray of
            length s (s > 1) determining time and entrainment rate time series.
            Time values are interpolated between the specified times, and the edge values are used for
            extrapolation.
        entrain_to_cth: bool or int
            If True, entrain to cloud top (mixing layer top) after calculating the corresponding delta.
            If False, entrain to the mixing layer base (surface layer in coupled cases).
            If int, then using this input as index such that 0 or -1 mean consistent entrainment to the surface
            layer or domain top, respectively.
            NOTE: the value of entrain_to_cth will be overwritten if provided as key in aer_info.
        implicit_ent: bool
            If True, using an implicit solver for entrainment. If False, using explicit solver.
        tau_mix: dict or float
            boundary-layer mixing time scale [s].
            if a float then using its value throughout the simulation time.
            if a dict, then treated as in the case of a dict for w_e_ent.
        heat_rate: xr DataArray, dict, or float
            heating rate over the domain added to the LES output sounding (negative values = cooling) [K s-1]
            if a float then using its value throughout the simulation time.
            if a dict, then treated as in the case of a dict for w_e_ent.
            if an xr DataArray, must contain the "height" [m] and "time" [s] coordinates. Values outside the
            coordinate range are extrapolated using the nearest edge values.
        tau_act: float, int, or None [--singular--]
            If float or int, then setting an activation time scale (10 s by default matching the CFDC).
            If None, then singular activation is instantaneous and depends on delta_t.
            Relevant for singular parameterizations.
        implicit_act: bool [--singular--]
            If True and tau_act is a scalar, using implicit solution to activation.
        implicit_sublim: bool
            If True, using implicit solution to sublimation (Ni reduction - relevant for relative_sublim == True).
        mixing_bounds: two-element tuple or list, or None
            Determining the mixing layer (especially relevant when using time-varying LES input).
            The first element provides a fixed lowest range of mixing (float), a time varying range (dict as
            in w_e_ent), or the method with which to determine mixing base (str). The second element is
            similar, but for the determination of the mixing layer top.
            If None, using the full domain.
            NOTE: currently, the only accepted pre-specified mixing determination method is "ql_thresh"
            (q_liq-based cloud base or top height detection method, allowing limit mixing to the cloud).
        v_f_ice: xr DataArray, dict, or float
            number-weighted ice crystal fall velocity [m/s].
            if a float then using its value throughout the simulation time.
            if a dict, then treated as in the case of a dict for w_e_ent.
            if an xr DataArray, must contain the "height" [m] and "time" [s] coordinates. Values outside the
            coordinate range are extrapolated using the nearest edge values.
        in_cld_q_thresh: float
            Mixing ratio threshold [kg/kg] for determination of in-cloud environment; also assigned to the
            'q_liq_pbl_cut' attribute value.
        nuc_RH_thresh: float, str, list, or None [--ABIFM--]
            An RH threshold (fraction) for ABIFM (which can nucleate outside a cloud layer), such that a threshold
            of 1.00 means nucleation only within cloud layers.
            If str equals to "use_ql" then limiting nucleation to levels where ql > in_cld_q_thresh.
            If list and the first element equals to "use_RH_and_ql" then limiting nucleation to levels where
            ql > in_cld_q_thresh and/or RH >= RH threshold set in the second list element.
            Ignored if None.
        time_splitting: bool
            If True, running the model using time splitting (processes are calculated sequentially, each based on
            the state produced by the other).
            If False, using process splitting (process calculations are based on the same state and their
            tendencies are added to produce the updated state).
        ent_then_act: bool
            if True, entrain aerosol and then activate. If False, activate and then entrain (in either case,
            these two processes are followed by mixing).
        prognostic_inp: bool
            if True, using prognostic aerosol (default - essentially, the purpose of this model).
            if False, using diagnostic INP, i.e., total activated INP numbers are calcuated while considering
            tau_act (singular) or Jhet in current time step (ABIFM).
        prognostic_ice: bool
            If True, using prognostic ice, i.e., ice particles have INP memory, thereby enabling sublimation
            such that particle INPs are restored (requires setting prognostic_inp to True).
            If False, ice particles have no memory, and therefore, no sublimation, for example.
            Note that prognostic_ice requires more computation time. Memory is only allocated for ice snapshot
            as in INAS.
            Requires: prognostic_inp == True.
        dt_out: np.ndarray, float, int, or None
            array specifying times at which prognostic variables will be saved.
            Using a constant value if float or int
            Saving every time step if None
            Requires prognostic_ice == True.
        relative_sublim: bool
            If True, using the relative reduction of Ni with height (based on LES).
            If False, using abosulte reduction.
            Requires prognostic_ice == True.
        aer_info: list of dict
            Used to initialize the aerosol arrays. Each element of the list describes a single population
            type providing its composition, concentration, and PSD, e.g., can use a single log-normal population
            of Illite, or two Illite PSDs with different mode diameter and geometric SD combined with a Kaolinite
            population.
            Each dictionary (i.e., an 'aerosol_attrs' list element) must contain the keys:

                1. n_init_max: [float] total concentration [m-3].

                2. psd: [dict] choose a 'type' key between several options (parentheses denote required dict key
                names; units are SI by default; for concentration and/or diameter values, other units can be
                specified using 'input_conc_units' and/or 'input_conc_units' input parameters):
                    - "mono": fixed-size population, i.e., a single particle diameter should be provided
                      (diam [m]).
                    - "logn": log--normal: provide geometric mean diameter (diam_mean [m]), geometric SD
                      (geom_sd), number of PSD bins or an alternative diameter bin array (n_bins), minimum diameter
                      (diam_min [m]; can be a 2-element tuple and then the 2nd is the maximum diameter cutoff),
                      and bin-to-bin mass ratio (m_ratio). Note that the effective bin-to-bin diameter ratio
                      equals m_ratio**(1/3).
                    - "multi_logn": multi-modal log-normal: as in "logn" but diam_mean, geom_sd, and n_init_max
                      need to be specified as lists or np.ndarrays with the same length (each characterizing
                      a single mode (bin array is identical and represents the sum of modes).
                    - "custom": custom size distribution with maunally specified bin values and PSD shape.
                      Provide the PSD diameter array (diam) and the number concentration per bin
                      (dn_dlogD). Optional input key includes normalization to n_init (norm_to_n_init_max)
                      that normalizes dn_dlogD such that such sum(dn_dlogD) = n_init_max.
                    - "default": (parameters not required) using a log-normal PSD with mean diameter
                      of 1e-6 m, geometric SD of 2.5, 35 PSD bins with minimum diameter of 0.01e-6 m and mass
                      ratio of 2, resulting in max diameter of ~26e-6 m.
            optional keys:
                1. name: [str] population name (or tag). A default string using nucleus type is used if not
                provided.

                2. nucleus_type: [str; --ABIFM--]  name of substance (e.g., Al2O3) - to initialize Jhet (must be
                specified for ABIFM).

                3. diam_cutoff: [float or tuple; --singular--] minimum particle diameter to consider.
                Using a value of 0.5e-6 as in D2010 if not specified. Use a 2-element tuple to specify a range of
                diameters to consider.

                4. T_array: [list or np.ndarray; --singular--] discrete temperature array. If not specified, using
                temperatures between the smallest LES-informed temperature (or -40 C)  and 0 with logarithmically-
                increasing delta_t.

                5. singular_fun: [lambda func. or str; --singular--] INP parametrization (typically as a function
                of T).
                str: use "D2010" to use eq. 1 in DeMott et al., 2010, "D2015" to use eq. 2 in DeMott et al.,
                2015, "D2010fit" to use the temperature dependence fit from fig. 2 in DeMott et al., 2010,
                "ND2012" for surface area temperature-based fit (eq. 5) in Niemand et al., JAS, 2012,
                "SC2020" for surface area temperature-based fit (eq. 5) in Schill et al., PNAS, 202,
                and "AT2013" for surface area temperature_based fit (eq.6) in Atkinson et al., NATURE, 2013.
                The D2015 has default values of the five coeff. from eq. 2 (cf - calibration correction factor,
                alpha, beta, gamma, delta); these might be coded as optional input for the AER class in
                the future.
                Note that "D2010fit" does not consider aerosol PSDs.
                Use "D2010" (default) if None.

                6. singular_scale: [float] Scale factor for 'singular_fun' or Jhet (1 by default).

                7. n_init_weight_prof: [dict] a dict with keys "height" and "weight". Each key contains
                a list or np.ndarray of length s (s > 1) determining PSD heights [m] and weighting profiles.
                Weights are applied on n_init such that n_init(z) = n_init_max * weighting_factor(z), i.e., a
                weighted_aer_prof filled with ones means that n_init(z) = n_init_max.
                if weights > 1 are specified, the profile is normalized to max value == 1. heights are interpolated
                between the specified heights, and the edge values are used for extrapolation (can be used to set
                different aerosol source layers at model initialization, and combined with turbulence weighting,
                allows the emulation of cloud-driven mixing.

                8. entrain_psd: [dict] PSD for entrained aerosol - similar to the aer_info dict for specifying the
                PSD parameters of the entrained aerosol (can be surface aerosol fluxes if entrain_from_cth=0, for
                example). The 'type' key value must be the same as the aer_info dict.
                optional keys:
                    1. src_weight_time: [dict] a dict with keys "time" and "weight" for entrainment source.
                9. entrain_to_cth: [bool or int] as in the 'entrain_to_cth' in the ci_model class attributes, the
                case of which will result in determining this attribute value only for this specific aerosol
                population.
                If not specified, using the default option, i.e., the initial PSD ('dn_dlogD) with a weight of 1.,
                which in likely most scenarios represent the free-tropospheric (or PBL top) as was the case until
                the Sep 6, 2020 commits.
        input_conc_units: str or None
            An str specifies the input aerosol concentration units that will be converted to SI in pre-processing.
            Relevant input parameters are: n_init_max and dn_dlogD (custom).
        input_diam_units: str or None
            An str specifies the input aerosol diameter units that will be converted to SI in pre-processing.
            Relevant input parameters are: diam (mono, custom) diam_mean (logn, multi_logn), diam_min
            (logn, multi_logn), and diam_cutoff.
        input_heatrate_units: str or None
            An str specifies the input heating rate units that will be converted to SI in pre-processing.
            The relevant input parameters is: heat_rate.
        do_act: bool
            determines whether aerosol (INP) activation will be performed.
        do_entrain: bool
            determines whether aerosols entrainment will be performed.
        do_mix_aer: bool
            determines whether mixing of aerosols will be performed.
        do_mix_ice: bool
            determines whether mixing of ice will be performed.
        do_sedim: bool
            determines whether ice sedimentation will be performed.
        do_sublim: bool
            determines whether ice sublimation will be performed (based on dNi/dz from LES).
        output_budgets: bool
            If True, then activation, entrainment, and mixing budgest are provided in the model output.
        output_aer_decay: bool
            If True, then generating an output field of the relative fraction of PBL aerosol relative to
            initial value, as well as the decay rate between consecutive time steps.
        run_model: bool
            True - run model once initialization is done.

        Other Parameters
        ----------------------
        les_out_path: str or None
            LES output path (can be relative to running directory). Use default if None.
        les_out_filename: str or None
            LES output filename. Use default file if None.
        les_bin_phys: bool
            IF True, using bin microphysics output namelist for harvesting LES data.
            If False, using bulk microphysics output namelist for harvesting LES data.
        t_harvest: scalar, 2- or 3-element tuple, list (or ndarray), or None
            If scalar then using the nearest time (assuming units of seconds) to initialize the model
            (single profile).
            If a tuple, cropping the range defined by the first two elements (increasing values) using a
            slice object. If len(t_harvest) == 3 then using the 3rd element as a time offset to subtract from
            the tiem array values.
            If a list, cropping the times specified in the list (can be used take LES output profiles every
            delta_t seconds.
            NOTE: default in the ci_model class (10800 s) is different than in the DHARMA init method (None).
        fields_to_retain: list or None
            Fieldnames to crop from the LES output (required to properly run the model).
            If None, then cropping the minimum number of required fields using DHARMA's namelist convention
            (Temperature [K], q_liq [kg/kg], RH [fraction], precipitation flux [mm/h], and ice number
            concentration [cm^-3]).
        height_ind_2crop: list, str, or None
            Indices of heights to crop from the model output (e.g., up to the PBL top).
            if str then different defitions for PBL:
                - if == "ql_pbl" then cropping all values within the PBL defined here based on the
                'q_liq_pbl_cut' attribute. If more than a single time step exist in the dataset, then cropping
                the highest index corresponding to the cutoff.
                - OTHER OPTIONS TO BE ADDED.
            If None then not cropping.
            Method to determine cloud base with:
                - if == "ql_thresh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        """
        # count processing time
        Now = time()

        # Set some simulation attributes.
        self.vars_harvested_from_les = ["RH", "ql", "T", "Ni", "prec", "rho"]  # processed vars used by the model
        self.final_t = final_t
        self.use_ABIFM = use_ABIFM
        self.in_cld_q_thresh = in_cld_q_thresh  # kg/kg
        self.nuc_RH_thresh = nuc_RH_thresh  # fraction value
        self.prognostic_inp = prognostic_inp
        if np.logical_and(not self.prognostic_inp, prognostic_ice):
            print("prognostic_inp is False while prognostic_ice, which requires True prognostic_inp, is False - "
                  "setting prognostic_ice = False")
            prognostic_ice = False
        self.prognostic_ice = prognostic_ice
        if isinstance(dt_out, (float, int)):
            print(f"Setting output time increments to {dt_out} s")
            dt_out = np.arange(0., self.final_t + 1e-10, dt_out)
        elif dt_out is None:
            print(f"Setting output time increments to 1 time step of {delta_t} s (none were specified)")
            dt_out = np.arange(0., self.final_t + 1e-10, delta_t)  # By default output every time step
        self.dt_out = dt_out

        # assign a unit registry and define percent units.
        self.ureg = pint.UnitRegistry()
        self.ureg.define(pint.definitions.UnitDefinition('percent', 'pct', (),
                         pint.converters.ScaleConverter(1 / 100.0)))

        # Load LES output
        if les_name == "DHARMA":
            les = LES.DHARMA(les_out_path=les_out_path, les_out_filename=les_out_filename, t_harvest=t_harvest,
                             fields_to_retain=fields_to_retain, height_ind_2crop=height_ind_2crop,
                             cbh_det_method=cbh_det_method, q_liq_pbl_cut=in_cld_q_thresh,
                             les_bin_phys=les_bin_phys)
            les.ds["rho"] = les.ds["rho"].isel({"time": 0})  # density is constant with time (per Exner function)
        else:
            raise NameError("Can't process LES model output from '%s'" % les_name)
        self.LES_attributes = {"LES_name": les_name,
                               "les_out_path": les.les_out_path,
                               "les_out_filename": les.les_out_filename,
                               "les_bin_phys": les.les_bin_phys,
                               "t_averaged_les": t_averaged_les,
                               "t_harvest": t_harvest,
                               "fields_to_retain": fields_to_retain,
                               "height_ind_2crop": height_ind_2crop,
                               "cbh_det_method": cbh_det_method}

        # time-averaged LES variable profile option
        if t_averaged_les:
            les_units = {}
            for key in self.vars_harvested_from_les:
                les_units.update({key: les.ds[key].attrs["units"]})
            Mean_time = les.ds["time"].mean()
            les.ds = les.ds.mean(dim="time")
            les.ds = les.ds.assign_coords({"time": Mean_time})
            les.ds = les.ds.expand_dims("time").transpose(*("height", "time"))
            for key in self.vars_harvested_from_les:  # restore attributes lost during averaging.
                les.ds[key].attrs["units"] = les_units[key]

            # Redetermine cloud bounds with the time-averaged profile for model consistency (entrainment, etc.).
            tmp_ds = xr.Dataset()  # first, use a temporary xr.Dataset to retain t-averaged precip rates.
            tmp_ds["P_Ni"], tmp_ds["Pcb_per_Ni"] = les.ds["P_Ni"].copy(), les.ds["Pcb_per_Ni"].copy()
            les._find_and_calc_cb_precip(self.LES_attributes["cbh_det_method"])
            tmp_fields = [x for x in les.ds.keys()]
            les.ds["P_Ni"].values, les.ds["Pcb_per_Ni"].values = tmp_ds["P_Ni"].values, tmp_ds["Pcb_per_Ni"].values

            # crop updated dataset (temporarily change les object attributes to invoke internal method)
            tmp_attrs = {"ql": les.q_liq_field, "height_dim": les.height_dim}
            les.q_liq_field["name"], les.q_liq_field["scaling"], les.height_dim = "ql", 1, "height"
            les._crop_fields(tmp_fields, height_ind_2crop)
            les.q_liq_field["name"], les.q_liq_field["scaling"], les.height_dim = \
                tmp_attrs["ql"]["name"], tmp_attrs["ql"]["scaling"], tmp_attrs["height_dim"]

        # Make self.les point at the LES object's xr.Dataset for accessibility
        self.LES_obj = les
        self.les = self.LES_obj.ds

        # Make sure ice does not sediment more than 1 vertical cell per time step. In that case change delta_t
        if isinstance(v_f_ice, dict):
            max_sediment_vel = np.max(v_f_ice["value"])
        else:
            max_sediment_vel = np.max(v_f_ice)
        max_sediment_dist = max_sediment_vel * delta_t  # maximum ice sedimentation distance per time step
        if custom_vert_grid is not None:
            height = custom_vert_grid.astype(np.float32)
            height = height[np.logical_and(height <= self.les["height"].max().values,
                                           height >= self.les["height"].min().values)]
            if len(height) < len(custom_vert_grid):
                print("Some heights were omitted because they are outside the processed LES dataset grid")
        else:
            height = self.les["height"].values
        if max_sediment_dist > np.min(np.diff(height)):
            delta_t = np.floor(np.min(np.diff(height)) / max_sediment_vel)
            print("∆t was modified to the largest integer preventing ice sedimentation of more than 1 " +
                  "grid cell (%d s)" % delta_t)
        self.delta_t = delta_t
        self.mod_nt = int(final_t / delta_t) + 1  # number of time steps
        self.mod_nt_out = len(dt_out)  # number of output time steps
        self.mod_nz = len(height)  # number of vertical layers

        # allocate xarray DataSet for model atmospheric state and prognosed variable fields
        self.ds = xr.Dataset()
        self.ds = self.ds.assign_coords({"height": height})
        self.ds = self.ds.assign_coords({"time": np.arange(self.mod_nt) * self.delta_t})
        self.ds = self.ds.assign_coords({"t_out": dt_out})
        delta_z = np.diff(self.ds["height"])
        self.ds["delta_z"] = xr.DataArray(np.concatenate((delta_z, np.array([delta_z[-1]]))),
                                          dims=("height"), attrs={"units": "$m$"})
        extrap_locs_tail = self.ds["time"] >= self.les["time"].max()
        extrap_locs_head = self.ds["time"] <= self.les["time"].min()
        x, y = np.meshgrid(self.les["height"], self.les["time"])
        for key in self.vars_harvested_from_les:

            # Linear interp (two 1D interpolations - fastest) if LES temporal evolution is to be considered.
            if self.les["time"].size > 1:
                self._set_1D_or_2D_var_from_AERut(self.les[key], key)
            else:
                # Use LES bounds (min & max) outside the available range (redundant step - could be useful later).
                key_array_tmp = np.zeros((self.mod_nz, self.mod_nt))
                if extrap_locs_head.sum() > 0:
                    key_array_tmp[:, extrap_locs_head.values] = np.tile(np.expand_dims(
                        np.interp(self.ds["height"], self.les["height"],
                                  self.les[key].sel({"time": self.les["time"].min()})),
                        axis=1), (1, np.sum(extrap_locs_head.values)))
                if extrap_locs_tail.sum() > 0:
                    key_array_tmp[:, extrap_locs_tail.values] = np.tile(np.expand_dims(
                        np.interp(self.ds["height"], self.les["height"],
                                  self.les[key].sel({"time": self.les["time"].max()})),
                        axis=1), (1, np.sum(extrap_locs_tail.values)))
                self.ds[key] = xr.DataArray(key_array_tmp, dims=("height", "time"))
            self.ds[key].attrs = self.les[key].attrs

        # init entrainment
        self.w_e_ent = w_e_ent
        self.entrain_to_cth = entrain_to_cth
        self.implicit_ent = implicit_ent
        self._set_1D_or_2D_var_from_AERut(w_e_ent, "w_e_ent", "$m/s$", "Cloud-top entrainment rate")
        if self.les["time"].size > 1:
            self._set_1D_or_2D_var_from_AERut({"time": self.les["time"].values,
                                               "value": self.les["lowest_cbh"].values},
                                              "lowest_cbh", "$m$", "Lowest cloud base height")
            self._set_1D_or_2D_var_from_AERut({"time": self.les["time"].values,
                                               "value": self.les["lowest_cth"].values},
                                              "lowest_cth", "$m$", "Lowest cloud top height")
        else:
            self._set_1D_or_2D_var_from_AERut(self.les["lowest_cbh"].item(),
                                              "lowest_cbh", "$m$", "Lowest cloud base height")
            self._set_1D_or_2D_var_from_AERut(self.les["lowest_cth"].item(),
                                              "lowest_cth", "$m$", "Lowest cloud top height")

        # init vertical mixing and generate a mixing layer mask for the model
        self.tau_mix = tau_mix
        self.mixing_bounds = mixing_bounds
        self._set_1D_or_2D_var_from_AERut(tau_mix, "tau_mix", "$s$", "Boundary-layer mixing time scale")
        if mixing_bounds is None:
            self.ds["mixing_mask"] = xr.DataArray(np.full((self.mod_nz, self.mod_nt),
                                                          True, dtype=bool), dims=("height", "time"))
        else:
            if isinstance(mixing_bounds[0], str):
                if mixing_bounds[0] == "ql_thresh":
                    self.ds["mixing_base"] = xr.DataArray(np.interp(
                        self.ds["time"], self.les["time"], self.les["lowest_cbh"]), dims=("time"))
                    self.ds["mixing_base"].attrs["units"] = "$m$"
            else:
                self._set_1D_or_2D_var_from_AERut(mixing_bounds[0], "mixing_base", "$m$", "Mixing layer base")
            if isinstance(mixing_bounds[1], str):
                if mixing_bounds[1] == "ql_thresh":
                    self.ds["mixing_top"] = xr.DataArray(np.interp(
                        self.ds["time"], self.les["time"], self.les["lowest_cth"]), dims=("time"))
                    self.ds["mixing_top"].attrs["units"] = "$m$"
            else:
                self._set_1D_or_2D_var_from_AERut(mixing_bounds[1], "mixing_top", "$m$", "Mixing layer top")
            mixing_mask = np.full((self.mod_nz, self.mod_nt), False, dtype=bool)
            for t in range(self.mod_nt):
                rel_ind = np.arange(
                    np.argmin(np.abs(self.ds["height"].values - self.ds["mixing_base"].values[t])),
                    np.argmin(np.abs(self.ds["height"].values - self.ds["mixing_top"].values[t])) + 1)  # inc. top
                mixing_mask[rel_ind, t] = True
            self.ds["mixing_mask"] = xr.DataArray(mixing_mask, dims=("height", "time"))
        self.ds["mixing_mask"].attrs["long_name"] = "Mixing-layer mask (True --> mixed)"

        # init number weighted ice fall velocity
        self.v_f_ice = v_f_ice
        self._set_1D_or_2D_var_from_AERut(v_f_ice, "v_f_ice", "$m/s$", "Number-weighted ice crystal fall velocity")

        # init and apply heating rates (prior to calculating delta_aw and/or other activation-related variables)
        self.heat_rate, self.input_heatrate_units = heat_rate, input_heatrate_units
        if self.heat_rate is not None:
            self._set_1D_or_2D_var_from_AERut(heat_rate, "heat_rate", r"$K\ s^{-1}$", "Atmospheric heating rate")
            if self.input_heatrate_units is not None:
                self.ds["heat_rate"].values = \
                    (self.ds["heat_rate"].values * self.ureg(self.input_heatrate_units)).to("K * s^{-1}").magnitude
            for t in range(1, self.mod_nt):
                self.ds["T"].values[:, t:] += self.ds["heat_rate"].isel({"time": [t]}).values * delta_t

        # set singular activation parameters.
        if isinstance(tau_act, (float, int)):
            self.use_tau_act = True
            self.tau_act = tau_act
        else:
            self.use_tau_act = False
            self.tau_act = None
        self.implicit_act = implicit_act

        # init sublimation
        self.relative_sublim = relative_sublim
        self.implicit_sublim = implicit_sublim

        # calculate delta_aw
        self._calc_delta_aw()

        # allocate aerosol population Datasets
        self.aer = {}
        self.aer_info = copy.deepcopy(aer_info)  # save the aerosol info dict for reference in a deep copy.
        self.input_conc_units, self.input_diam_units = input_conc_units, input_diam_units
        self._convert_input_to_SI()  # Convert input concentration and/or diameter parameters to SI (if requested).
        optional_keys = ["name", "nucleus_type", "diam_cutoff", "T_array",  # optional aerosol class input params.
                         "n_init_weight_prof", "singular_fun", "singular_scale",
                         "entrain_psd", "entrain_to_cth"]
        for ii in range(len(self.aer_info)):
            param_dict = {"use_ABIFM": use_ABIFM}  # tmp dict for aerosol attributes to send to class call.
            if np.all([x in self.aer_info[ii].keys() for x in ["n_init_max", "psd"]]):
                param_dict["n_init_max"] = self.aer_info[ii]["n_init_max"]
                param_dict["psd"] = self.aer_info[ii]["psd"]
            else:
                raise KeyError('aerosol information requires the keys "n_init_max", "psd"')
            if not self.aer_info[ii]["psd"]["type"] in ["mono", "logn", "multi_logn", "custom", "default"]:
                raise ValueError('PSD type must be one of: "mono", "logn", "multi_logn", "custom", "default"')
            for key in optional_keys:
                param_dict[key] = self.aer_info[ii][key] if key in self.aer_info[ii].keys() else None

            # set aerosol population arrays
            tmp_aer_pop = self._set_aer_obj(param_dict)
            self.aer[tmp_aer_pop.name] = tmp_aer_pop

        # allocate nucleated ice DataArrays
        if not self.prognostic_ice:
            self.ds["ice_snap"] = xr.DataArray(np.zeros(self.ds["height"].size), dims=("height"))
            self.ds["ice_snap"].attrs["units"] = "$m^{-3}$"
            self.ds["ice_snap"].attrs["long_name"] = "Diagnostic ice number concentration (snapshot)"
        self.ds["Ni_nuc"] = xr.DataArray(np.zeros((self.mod_nz,
                                                   self.mod_nt_out)), dims=("height", "t_out"))
        self.ds["Ni_nuc"].attrs["units"] = "$m^{-3}$"
        self.ds["Ni_nuc"].attrs["long_name"] = "Nucleated ice"
        self.ds["nuc_rate"] = xr.DataArray(np.zeros((self.mod_nz,
                                           self.mod_nt_out)), dims=("height", "t_out"))
        self.ds["nuc_rate"].attrs["units"] = r"$m^{-3}\:s^{-1}$"
        self.ds["nuc_rate"].attrs["long_name"] = "Ice nucleation rate"

        print("Model initalization done! Total processing time = %f s" % (time() - Now))

        # Set additional coordinates and attributes
        self.ds["height"].attrs["units"] = "$m$"
        self.ds["time"].attrs["units"] = "$s$"
        self.ds["height_km"] = self.ds["height"].copy() / 1e3  # add coordinates for height in km.
        self.ds = self.ds.assign_coords(height_km=("height", self.ds["height_km"].values))
        self.ds["height_km"].attrs["units"] = "$km$"
        self.ds["time_h"] = self.ds["time"].copy() / 3600  # add coordinates for time in h.
        self.ds = self.ds.assign_coords(time_h=("time", self.ds["time_h"].values))
        self.ds["time_h"].attrs["units"] = "$h$"
        self.ds = self.ds.assign_coords({"t_out": self.dt_out})
        self.ds["t_out"].attrs["units"] = "$s$"
        self.ds["t_out_h"] = self.ds["t_out"].copy() / 3600  # add coordinates for time in h.
        self.ds = self.ds.assign_coords(t_out_h=("t_out", self.ds["t_out_h"].values))
        self.ds["t_out_h"].attrs["units"] = "$h$"
        self.time_dim = "time"
        self.height_dim = "height"
        self.t_out_dim = "t_out"
        self.T_dim = "T"  # setting the T dim even though it is only set when allocating an AER object.
        self.diam_dim = "diam"  # setting the diam dim even though it is only set when allocating an AER object.

        # Run the model and reassign coordinate unit attributes (typically lost in xr.DataArray manipulations)
        if np.logical_and(not self.prognostic_ice, do_sublim):
            print("prognostic_ice is False while do_sublim is True, but do_sublim requires prognostic ice - "
                  "setting do_sublim = False")
            do_sublim = False
        self.do_act = do_act
        self.do_entrain = do_entrain
        self.do_mix_aer = do_mix_aer
        self.do_mix_ice = do_mix_ice
        self.do_sedim = do_sedim
        self.do_sublim = do_sublim
        self.time_splitting = time_splitting
        self.ent_then_act = ent_then_act
        self.output_budgets = output_budgets
        self.output_aer_decay = output_aer_decay
        if run_model:
            Run(self)
            self.ds["time_h"].attrs["units"] = "$h$"
            self.ds["time"].attrs["units"] = "$s$"
            self.ds["t_out"].attrs["units"] = "$s$"
            self.ds["t_out_h"].attrs["units"] = "$h$"
            self.ds["height_km"].attrs["units"] = "$km$"
            self.ds["height"].attrs["units"] = "$m$"
            for key in self.aer.keys():
                self.aer[key].ds["time_h"].attrs["units"] = "$h$"
                self.aer[key].ds["time"].attrs["units"] = "$s$"
                self.aer[key].ds["t_out_h"].attrs["units"] = "$h$"
                self.aer[key].ds["t_out"].attrs["units"] = "$s$"
                self.aer[key].ds["height_km"].attrs["units"] = "$km$"
                self.aer[key].ds["height"].attrs["units"] = "$m$"

    @staticmethod
    def calc_a_ice_w(T):
        """
        calculate a_w(ice) using eq. 7 in Koop and Zobrist (2009, https://doi.org/10.1039/B914289D.

        Parameters
        ----------
        T: np.ndarray or xr.DataArray
            Temperature

        Returns
        -------
        a_ice_w: np.ndarray or xr.DataArray
            water activity for ice nucleation
        """
        a_ice_w = \
            (np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) -
                    0.00728332 * T) /
             (np.exp(54.842763 - 6763.22 / T -
              4.210 * np.log(T) + 0.000367 * T +
              np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)))
             )
        return a_ice_w

    def _calc_delta_aw(self):
        """
        calculate the ∆aw field and S_ice for ABIFM using:
        1. eq. 1 in Knopf and Alpert (2013, https://doi.org/10.1039/C3FD00035D) combined with:
        2. eq. 7 in Koop and Zobrist (2009, https://doi.org/10.1039/B914289D) for a_w(ice)
        Here we assume that our droplets are in equilibrium with the environment at its given RH, hence, RH = a_w.
        """
        a_ice_w = self.calc_a_ice_w(self.ds['T'])
        self.ds["delta_aw"] = self.ds['RH'] - a_ice_w
        self.ds["S_ice"] = self.ds['RH'] / a_ice_w
        self.ds['delta_aw'].attrs['units'] = ""
        self.ds["S_ice"].attrs['units'] = ""

    def _set_1D_or_2D_var_from_AERut(self, var_in, var_name, units_str=None, long_name_str=None):
        """
        set a 1D xr.DataArray from a scalar or a dictionary containing "time" and "value" keys.
        If 'var_in' is a scalar then generating a uniform time series.
        Values are linearly interpolated onto the model temporal grid (values outside the provided
        range are extrapolated.
        The method can also operate on an xr.DataArray. In that case it interpolates the input
        variable (containing "time" and "height" coordinates) onto the ci_model object's grid
        and also extrapolates using edge values (two-1D linear interpolations are performed).

        Parameters
        ---------
        var_in: xr.DataArray, dict, or scalar.
            if xr.DataArray, must have "time" and "height" coordinates and dims.
            if dict then using the "time" and "value" keys of the variable.
        var_name: str
            Name of DataArray variable.
        units_str: str
            string for the units attribute.
        long_name_str: str
            string for the long_name attribute.
        """
        if isinstance(var_in, (float, int)):
            self.ds[var_name] = xr.DataArray(np.ones(self.mod_nt) * var_in, dims=("time"))
        elif isinstance(var_in, dict):  # 1D linear interpolation
            if not np.all([x in var_in.keys() for x in ["time", "value"]]):
                raise KeyError('variable time series requires the keys "time" and "value"')
            if not np.logical_and(len(var_in["time"]) > 1,
                                  len(var_in["time"]) == len(var_in["value"])):
                raise ValueError("times and values must have the same length > 1")
            self.ds[var_name] = xr.DataArray(np.interp(self.ds["time"],
                                             var_in["time"], var_in["value"]), dims=("time"))
        elif isinstance(var_in, xr.DataArray):  # 2D linear interpolation
            if not np.all([x in var_in.coords for x in ["time", "height"]]):
                raise KeyError('2D variable processing requires the "time" and "height" coordinates!')
            if not np.logical_and(len(var_in["time"]) > 1, len(var_in["height"]) > 1):
                raise ValueError("times and height coordinates must be longer than 1 for interpolation!")
            key_array_tmp = np.zeros((self.mod_nz, self.mod_nt))
            key_1st_interp = np.zeros((var_in["height"].size, self.mod_nt))
            for hh in range(var_in["height"].size):
                key_1st_interp[hh, :] = np.interp(self.ds["time"].values, var_in["time"].values,
                                                  var_in.isel({"height": hh}))
            for tt in range(self.mod_nt):
                key_array_tmp[:, tt] = np.interp(self.ds["height"].values, var_in["height"].values,
                                                 key_1st_interp[:, tt])
            self.ds[var_name] = xr.DataArray(key_array_tmp, dims=("height", "time"))
        else:
            raise TypeError("Input variable must be of type float, int, dict, or xr.DataArray!")
        if units_str is not None:
            self.ds[var_name].attrs["units"] = units_str
        if long_name_str is not None:
            self.ds[var_name].attrs["long_name"] = long_name_str

    def _set_aer_obj(self, param_dict):
        """
        Invoke an AER class call and use the input parameters provided. Using a full dictionary key call to
        maintain consistency even if some AER class input variable order will be changed in future updates.

        Parameters
        ----------
        param_dict: dict
            Keys include all possible input parameters for the AER sub-classes.

        Returns
        -------
        tmp_aer_pop: AER class object
            AER class object that includes the AER array with dims height x time x diameter (ABIFM) or
            height x time x temperature (singular).
        """
        if param_dict["psd"]["type"] == "mono":
            tmp_aer_pop = AER.mono_AER(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       entrain_psd=param_dict["entrain_psd"],
                                       entrain_to_cth=param_dict["entrain_to_cth"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "logn":
            tmp_aer_pop = AER.logn_AER(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       entrain_psd=param_dict["entrain_psd"],
                                       entrain_to_cth=param_dict["entrain_to_cth"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "multi_logn":
            tmp_aer_pop = AER.multi_logn_AER(use_ABIFM=param_dict["use_ABIFM"],
                                             n_init_max=param_dict["n_init_max"],
                                             psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                             name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                             T_array=param_dict["T_array"],
                                             singular_fun=param_dict["singular_fun"],
                                             entrain_psd=param_dict["entrain_psd"],
                                             entrain_to_cth=param_dict["entrain_to_cth"],
                                             singular_scale=param_dict["singular_scale"],
                                             n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "custom":
            tmp_aer_pop = AER.custom_AER(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                         psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                         name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                         T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                         entrain_psd=param_dict["entrain_psd"],
                                         entrain_to_cth=param_dict["entrain_to_cth"],
                                         singular_scale=param_dict["singular_scale"],
                                         n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "default":
            param_dict["psd"].update({"diam_mean": 1e-6, "geom_sd": 2.5, "n_bins": 35, "diam_min": 0.01e-6,
                                      "m_ratio": 2.})  # default parameters.
            tmp_aer_pop = AER.logn_AER(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       entrain_psd=param_dict["entrain_psd"],
                                       entrain_to_cth=param_dict["entrain_to_cth"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)

        return tmp_aer_pop

    def _convert_input_to_SI(self):
        """
        Convert one or more input parameters to SI if other units were specified.
        """
        if self.input_conc_units is not None:  # assuming input_conc_units is an str with valid conc. units
            self._do_input_conversion(["n_init_max", "dn_dlogD"], self.input_conc_units, "m^{-3}")
        if self.input_diam_units is not None:  # assuming input_diam_units is an str with valid length units
            self._do_input_conversion(["diam", "diam_mean", "diam_min", "diam_cutoff"], self.input_diam_units, "m")

    def _do_input_conversion(self, param_list, from_units, to_units):
        """
        Search for input parameters in the aer_info input list of dicts and convert units to SI.
        Quantity type is parsed by pint (for all valid unit strings see:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt).

        Parameters
        ----------
        param_list: list
            Elements include all possible (and relevant) input parameters for conversion, so define wisely.
        from_units: str
            Units to convert from (input units).
        to_units: str
            Units to convert to.
        """
        for ii in range(len(self.aer_info)):
            for param in param_list:
                if param in self.aer_info[ii]["psd"].keys():
                    param_val = (self.aer_info[ii]["psd"][param] * self.ureg(from_units)).to(to_units).magnitude
                    if type(self.aer_info[ii]["psd"][param]) == tuple:
                        self.aer_info[ii]["psd"][param] = tuple(param_val)
                    elif type(self.aer_info[ii]["psd"][param]) == list:
                        self.aer_info[ii]["psd"][param] = list(param_val)
                    else:  # scalar or np.ndarray
                        self.aer_info[ii]["psd"][param] = param_val
                    print("'%s' (in aer_info's 'psd' keys) was input in %s units; now converted to %s (SI)" %
                          (param, from_units, to_units))
                if param in self.aer_info[ii].keys():
                    param_val = (self.aer_info[ii][param] * self.ureg(from_units)).to(to_units).magnitude
                    if type(self.aer_info[ii][param]) == tuple:
                        self.aer_info[ii][param] = list(param_val)
                    elif type(self.aer_info[ii][param]) == list:
                        self.aer_info[ii][param] = list(param_val)
                    else:  # scalar or np.ndarray
                        self.aer_info[ii][param] = param_val
                    print("'%s' (in aer_info) was input in %s units; now converted to %s (SI)" %
                          (param, from_units, to_units))

    def _convert_quantity_units(self, to_units):
        """
        Convert a quantity units (e.g., volume, concentration) in all relevant arrays (e.g., from 1/m^3 to L-1).

        Parameters
        ---------
        to_units: str
            Units to convert to. Quantity type is parsed by pint (for all valid unit strings see:
            https://github.com/hgrecco/pint/blob/master/pint/default_en.txt).
        """
        Converted = []  # converted fields
        for DA in self.ds.keys():
            if isinstance(self.ds[DA].data, pint.Quantity):
                if self.ds[DA].data.check(to_units):
                    Converted.append("The units of '%s' converted from %s to %s" %
                                     (DA, self.ds[DA].attrs["units"], to_units))
                    self.ds[DA].data = self.ds[DA].data.to(to_units)
                    self.ds[DA].attrs["units"] = r"$%s$" % to_units
        for key in self.aer.keys():
            for DA in self.aer[key].ds.keys():
                if isinstance(self.aer[key].ds[DA].data, pint.Quantity):
                    if self.aer[key].ds[DA].data.check(to_units):
                        Converted.append("The units of '%s' in the '%s' popolation converted from %s to %s" %
                                         (DA, key, self.aer[key].ds[DA].attrs["units"], to_units))
                        self.aer[key].ds[DA].data = self.aer[key].ds[DA].data.to(to_units)
                        self.aer[key].ds[DA].attrs["units"] = r"$%s$" % to_units
        if Converted:
            for Conv_str in Converted:
                print(Conv_str)
        else:
            print("No fields with units able to convert to %s " % to_units)

    def _swap_height_dim_to_from_km(self):
        """
        If the height dim is in m changing to km and vice versa.
        """
        if "height" in self.ds.dims:
            print("Converting height dimension units from meters to kilometers")
            self.ds = self.ds.swap_dims({"height": "height_km"})
            self.height_dim = "height_km"
            for key in self.aer.keys():
                self.aer[key].ds = self.aer[key].ds.swap_dims({"height": "height_km"})
        else:
            print("Converting height dimension units from kilometers to meters")
            self.ds = self.ds.swap_dims({"height_km": "height"})
            self.height_dim = "height"
            for key in self.aer.keys():
                self.aer[key].ds = self.aer[key].ds.swap_dims({"height_km": "height"})

    def _swap_time_dim_to_from_hr(self):
        """
        If the time dim is in seconds changing to hours and vice versa.
        """
        if "time" in self.ds.dims:
            print("Converting time dimension units from seconds to hours")
            self.ds = self.ds.swap_dims({"time": "time_h"})
            self.time_dim = "time_h"
            for key in self.aer.keys():
                self.aer[key].ds = self.aer[key].ds.swap_dims({"time": "time_h"})
        else:
            print("Converting time dimension units from hours to seconds")
            self.ds = self.ds.swap_dims({"time_h": "time"})
            self.time_dim = "time"
            for key in self.aer.keys():
                self.aer[key].ds = self.aer[key].ds.swap_dims({"time_h": "time"})
        if "t_out" in self.ds.dims:
            print("Converting output time dimension units from seconds to hours")
            self.ds = self.ds.swap_dims({"t_out": "t_out_h"})
            self.t_out_dim = "t_out_h"
            for key in self.aer.keys():
                self.aer[key].ds = self.aer[key].ds.swap_dims({"t_out": "t_out_h"})
        else:
            print("Converting output time dimension units from hours to seconds")
            self.ds = self.ds.swap_dims({"t_out_h": "t_out"})
            self.t_out_dim = "t_out"
            for key in self.aer.keys():
                self.aer[key].ds = self.aer[key].ds.swap_dims({"t_out_h": "t_out"})

    def _swap_diam_dim_to_from_um(self):
        """
        If the diam dim is in m changing to um and vice versa.
        """
        for key in self.aer.keys():
            if "diam" in self.aer[key].ds.dims:
                print("Converting diameter dimension units for %s from meters to micrometers" % key)
                self.aer[key].ds = self.aer[key].ds.swap_dims({"diam": "diam_um"})
                self.diam_dim = "diam_um"
            else:
                print("Converting diameter dimension units for %s from micrometers to meters" % key)
                self.aer[key].ds = self.aer[key].ds.swap_dims({"diam_um": "diam"})
                self.diam_dim = "diam"

    def _swap_T_dim_to_from_C(self):
        """
        If the T dim is in Kelvin changing to Celsius and vice versa (singular).
        """
        if not self.use_ABIFM:
            for key in self.aer.keys():
                if "T" in self.aer[key].ds.dims:
                    print("Converting diameter dimension units for %s from Kelvin to Celsius" % key)
                    self.aer[key].ds = self.aer[key].ds.swap_dims({"T": "T_C"})
                    self.T_dim = "T_C"
                else:
                    print("Converting diameter dimension units for %s from Celsius to Kelvin" % key)
                    self.aer[key].ds = self.aer[key].ds.swap_dims({"T_C": "T"})
                    self.T_dim = "T"

    def ci_model_ds_to_netcdf(self, out_prefix='AC_1D_out'):
        """
        export datasets from a model simulation. Each dataset is stored in a different file.
        Files are generated for the main ci_model object and each aerosol population.

        Parameters
        ----------
        out_prefix: str
            filename prefix and path from which to load ci_model's datasets.
            A "_main.nc" suffix is added to the filename of the NetCDF file containing the main
            ci_model dataset, while for each dataset of an aerosol population xxxx, an
            'aer_pop_xxxx.nc' suffix is added.
        """
        out_filenames = []
        ds_4_out = self.ds.copy(deep=True)
        ds_4_out = self.strip_units(ds_4_out)
        out_filenames.append(out_prefix + "_main.nc")
        ds_4_out.to_netcdf(out_filenames[-1])
        for aer_key in self.aer.keys():
            ds_4_out = self.aer[aer_key].ds.copy(deep=True)
            ds_4_out = self.strip_units(ds_4_out)
            out_filenames.append(out_prefix + f"_aer_pop_{aer_key}.nc")
            ds_4_out.to_netcdf(out_filenames[-1])
        print("Exporting ci_model xr.Dataset to the following files\n")
        print(out_filenames + "\n")

    def ci_model_ds_from_netcdf(self, out_prefix='AC_1D_out'):
        """
        Load datasets from a model simulation. Each dataset is stored in a different file.
        Assumes files were generated for the main ci_model object and each aerosol population.

        Parameters
        ----------
        out_prefix: str
            filename prefix and path from which to load ci_model's datasets.
            A "_main.nc" suffix is added to the filename of the NetCDF file containing the main
            ci_model dataset, while for each dataset of an aerosol population xxxx, an
            'aer_pop_xxxx.nc' suffix is added.
        """
        ds_4_out = xr.open_dataset(out_prefix + "_main.nc")
        ds_4_out = self.reassign_units(ds_4_out)
        self.ds = ds_4_out
        for aer_key in self.aer.keys():
            ds_4_out = xr.open_dataset(out_prefix + f"_aer_pop_{aer_key}.nc")
            ds_4_out = self.reassign_units(ds_4_out)
            self.aer[aer_key].ds = ds_4_out
        print(f"Loading ci_model xr.Datasets from the {out_prefix} files done!\n")

    @staticmethod
    def strip_units(ds_4_out):
        """
        Strip units from fields in an xr.Dataset enabling export to NetCDF files
        (convert pint.Quantity data fields to np.ndarray while saving stripping info).

        Parameters
        ----------
        ds_4_out: xr.Dataset
            Dataset from which to strip units

        Returns
        -------
        ds_4_out: xr.Dataset
            Dataset with to stripped units.
        """
        for key in ds_4_out.keys():
            if isinstance(ds_4_out[key].data, pint.quantity.Quantity):
                print(f"Stripping units from '{key}'")
                ds_4_out[key].data = ds_4_out[key].data.magnitude
                ds_4_out[key].attrs["stripped_units"] = 1
            else:
                ds_4_out[key].attrs["stripped_units"] = 0
        return ds_4_out

    def reassign_units(self, ds_4_out):
        """
        Reassign units to fields in an xr.Dataset loaded from a NetCDF file assuming that
        a 'stripped_units' attribute exists.
        (convert np.ndarray data fields to pint.Quantity and delete stripping info).

        Parameters
        ----------
        ds_4_out: xr.Dataset
            Dataset with to stripped units.

        Returns
        -------
        ds_4_out: xr.Dataset
            Dataset with with units added to fields.
        """
        for key in ds_4_out.keys():
            if ds_4_out[key].attrs["stripped_units"]:
                print(f"Restoring units to '{key}'")
                ds_4_out[key].data *= self.ureg(ds_4_out[key].attrs["units"])
                del ds_4_out[key].attrs["stripped_units"]
        return ds_4_out

    def _recalc_cld_and_mixing(self):
        """
        Recalculate Jhet (ABIFM) and  LES-harvested parameters following changes to LES ouput (essentially,
        cloud depth) in order for the model to consider in simulation. Mixing bounds are updated only if they
        are cloud-dependent (e.g., using 'ql_thresh').
        NOTE: no other change is made to the grid or cropped fields, so these parameters should be specified in
        the first call to init_model.
        NOTE: in the case of ABIFM, 'inp_cum_init' and 'inp_pct' are not recalculated.
        ALSO, do not change units from SI before calling this method.
        """
        print("recalculating cloud depth and mixing layer depth")
        # find all cloud bases and the precip rate in the lowest cloud base in every time step (each profile).
        if self.LES_attributes["cbh_det_method"] == "ql_thresh":
            cbh_all = np.diff(self.ds["ql"].values >= self.in_cld_q_thresh, prepend=0, axis=0) == 1
            cth_all = np.diff(self.ds["ql"].values >= self.in_cld_q_thresh, append=0, axis=0) == -1
        else:
            print("Unknown cbh method string - skipping cbh detection function")
            return
        self.ds["lowest_cbh"].values = np.full(self.ds.dims["time"], np.nan)
        self.ds["lowest_cth"].values = np.full(self.ds.dims["time"], np.nan)
        for tt in range(self.ds.dims["time"]):
            cbh_lowest = np.argwhere(cbh_all[:, tt]).flatten()
            if len(cbh_lowest):
                cth_lowest = np.argwhere(cth_all[:, tt]).flatten()
                self.ds["lowest_cbh"].values[tt] = self.ds["height"].values[cbh_lowest[0]]
                self.ds["lowest_cth"].values[tt] = self.ds["height"].values[cth_lowest[0]]

        # redetermine mixing bounds and mixing mask
        if self.mixing_bounds is not None:
            if isinstance(self.mixing_bounds[0], str):
                if self.mixing_bounds[0] == "ql_thresh":
                    self.ds["mixing_base"].values = np.copy(self.ds["lowest_cbh"].values)
            if isinstance(self.mixing_bounds[1], str):
                if self.mixing_bounds[1] == "ql_thresh":
                    self.ds["mixing_top"].values = np.copy(self.ds["lowest_cth"].values)
            mixing_mask = np.full((self.mod_nz, self.mod_nt), False, dtype=bool)
            for t in range(self.mod_nt):
                rel_ind = np.arange(
                    np.argmin(np.abs(self.ds["height"].values - self.ds["mixing_base"].values[t])),
                    np.argmin(np.abs(self.ds["height"].values - self.ds["mixing_top"].values[t])) + 1)  # inc. top
                mixing_mask[rel_ind, t] = True
            self.ds["mixing_mask"].values = mixing_mask

        # Recalculate delta_aw
        print("recalculating delta_aw")
        self._calc_delta_aw()  # recalculate delta_aw

        if self.use_ABIFM:
            # Recalculate Jhet for ABIFM (NOTE that 'inp_cum_init' and 'inp_pct' are not recalculated)
            print("recalculating Jhet (use_ABIFM == True)")
            for key in self.aer.keys():
                self.aer[key].ds["Jhet"] = 10.**(self.aer[key].Jhet.c + self.aer[key].Jhet.m *
                                                 self.ds["delta_aw"]) * 1e4  # calc Jhet
                if self.aer[key].singular_scale != 1.:
                    self.aer[key].ds["Jhet"].values *= self.aer[key].singular_scale
                self.aer[key].ds["Jhet"].attrs["units"] = "$m^{-2} s^{-1}$"
                self.aer[key].ds["Jhet"].attrs["long_name"] = "Heterogeneous ice nucleation rate coefficient"
        else:
            # allocate aerosol population Datasets (required since the T array might have changed)
            self.aer = {}
            optional_keys = ["name", "nucleus_type", "diam_cutoff", "T_array",  # opt. aerosol class input params
                             "n_init_weight_prof", "singular_fun", "singular_scale",
                             "entrain_psd", "entrain_to_cth"]
            for ii in range(len(self.aer_info)):
                param_dict = {"use_ABIFM": self.use_ABIFM}  # tmp dict for aerosol attributes to send to class call
                if np.all([x in self.aer_info[ii].keys() for x in ["n_init_max", "psd"]]):
                    param_dict["n_init_max"] = self.aer_info[ii]["n_init_max"]
                    param_dict["psd"] = self.aer_info[ii]["psd"]
                else:
                    raise KeyError('aerosol information requires the keys "n_init_max", "psd"')
                if not self.aer_info[ii]["psd"]["type"] in ["mono", "logn", "multi_logn", "custom", "default"]:
                    raise ValueError('PSD type must be one of: "mono", "logn", "multi_logn", "custom", "default"')
                for key in optional_keys:
                    param_dict[key] = self.aer_info[ii][key] if key in self.aer_info[ii].keys() else None

                # set aerosol population arrays
                tmp_aer_pop = self._set_aer_obj(param_dict)
                self.aer[tmp_aer_pop.name] = tmp_aer_pop

    @staticmethod
    def generate_figure(**kwargs):
        """
        A method for generating a figure object.
        """
        return plotting.generate_figure(**kwargs)

    def plot_curtain(self, **kwargs):
        """
        A method for curtain plots based on the object's xr.DataSet
        """
        return plotting.plot_curtain(self, **kwargs)

    def plot_tseries(self, **kwargs):
        """
        A method for time series plots based on the object's xr.DataSet
        """
        return plotting.plot_tseries(self, **kwargs)

    def plot_profile(self, **kwargs):
        """
        A method for profile plots based on the object's xr.DataSet
        """
        return plotting.plot_profile(self, **kwargs)

    def plot_psd(self, **kwargs):
        """
        A method for PSD plots based on the object's xr.DataSet
        """
        return plotting.plot_psd(self, **kwargs)
