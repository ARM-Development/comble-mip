"""
This module is used for quick calculations of INP, Jhet, etc.
"""
import xarray as xr
import numpy as np
import copy
import AER
from init_model import ci_model


class ci_quickcalc(ci_model):
    """
    Quick aerosol and INP calculations using the ci_model class
    """
    def __init__(self, aer_info_dict, T_in, use_ABIFM=True, RH_in=None, P_in=None, ABIFM_delta_t=10.,
                 input_conc_units=None, input_diam_units=None):
        """
        Model namelists and unit conversion coefficient required for the quick calculation.

        Parameters
        ----------
        aer_info_dict: dict
            A dict describing a single aerosol population providing its composition, concentration, and PSD,
            The dict must contain the keys:

                1. n_init_max: [float] total concentration [m-3].

                2. psd: [dict] choose a 'type' key between several options (parentheses denote required dict key
                names; units are SI by default; for concentration and/or diameter values, other units can be
                specified using 'input_conc_units' and/or 'input_conc_units' input parameters):
                    - "mono": fixed-size population, i.e., a single particle diameter should be provided
                      (diam [m]).
                    - "logn": log--normal: provide geometric mean diameter (diam_mean [m]), geometric SD
                      (geom_sd), number of PSD bins (n_bins), minimum diameter (diam_min [m]; can also be  a
                      2-element tuple and then the 2nd is the maximum diameter cutoff), and
                      bin-to-bin mass ratio (m_ratio). Note that the effective bin-to-bin diameter ratio
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
        T_in: float
            Input temperature [K]
        use_ABIFM: bool
            True - use ABIFM, False - use singular.
        RH_in: float or None [ABIFM]
            Input relative humidity wrt liquid water [%]. Required only for ABIFM for water activity calculation.
        P_in: float or None [singular INN]
            Input pressure [Pa]. Required only for singular INN for conversion from SL-1 to L-1
        ABIFM_delta_t: float [ABIFM]
            Delta_t to use to calculate activated INP when using ABIFM (10 s by default matching the CFDC).
        input_conc_units: str or None
            An str specifies the input aerosol concentration units that will be converted to SI in pre-processing.
            Relevant input parameters are: n_init_max and dn_dlogD (custom).
        input_diam_units: str or None
            An str specifies the input aerosol diameter units that will be converted to SI in pre-processing.
            Relevant input parameters are: diam (mono, custom) diam_mean (logn, multi_logn), diam_min
            (logn, multi_logn), and diam_cutoff.
        """
        # First, setting flags,arrays, and scalars required for AER module operation.
        self.prognostic_ice, self.prognostic_inp, self.entrain_to_cth, \
            self.use_ABIFM, self.dt_out, self.in_cld_q_thresh = \
            False, True, True, use_ABIFM, np.array([0.]), 0.

        # set dummy dataset
        self.Rd = 287.052874  # J kg-1 K-1
        self.delta_t = ABIFM_delta_t
        if use_ABIFM:
            RH_in /= 100.  # Conversion to fractional RH
        elif RH_in is None:
            RH_in = np.nan
        if P_in is None:
            P_in = np.nan
        self.ds = xr.Dataset(
            data_vars={"T": (["height", "time"], np.full((1, 1), T_in)),
                       "RH": (["height", "time"], np.full((1, 1), RH_in)),
                       "rho": (["height", "time"], np.full((1, 1), P_in / (self.Rd * T_in))),
                       "ql": (["height", "time"], np.full((1, 1), 1.))},
            coords={"height": [0.], "time": [0.], "t_out": [0.]}
        )

        # calc Delta_aw
        self._calc_delta_aw()

        # allocate aerosol population Datasets
        self.aer = {}
        self.aer_info_dict = copy.deepcopy(aer_info_dict)  # save aerosol info dict for reference in a deep copy
        self.input_conc_units, self.input_diam_units = input_conc_units, input_diam_units
        self._convert_input_to_SI()  # Convert input concentration and/or diameter parameters to SI (if requested).
        optional_keys = ["name", "nucleus_type", "diam_cutoff", "T_array",  # optional aerosol class input params.
                         "n_init_weight_prof", "singular_fun", "singular_scale",
                         "entrain_psd", "entrain_to_cth"]
        param_dict = {"use_ABIFM": use_ABIFM}  # tmp dict for aerosol attributes to send to class call.
        if np.all([x in self.aer_info_dict.keys() for x in ["n_init_max", "psd"]]):
            param_dict["n_init_max"] = self.aer_info_dict["n_init_max"]
            param_dict["psd"] = self.aer_info_dict["psd"]
        else:
            raise KeyError('aerosol information requires the keys "n_init_max", "psd"')
        if not self.aer_info_dict["psd"]["type"] in ["mono", "logn", "multi_logn", "custom", "default"]:
            raise ValueError('PSD type must be one of: "mono", "logn", "multi_logn", "custom", "default"')
        for key in optional_keys:
            param_dict[key] = self.aer_info_dict[key] if key in self.aer_info_dict.keys() else None

        # set aerosol population arrays
        tmp_aer_pop = self._set_aer_obj(param_dict)
        self.aer[tmp_aer_pop.name] = tmp_aer_pop


def quickcalc(aer_info_dict, T_in, use_ABIFM=True, RH_in=None, P_in=None, ABIFM_delta_t=10.,
              input_conc_units=None, input_diam_units=None):
    """
    The method initializes a quickcalc class object but then performs additional processing
    to return a summary xr.Dataset.

    Parameters
    ----------
    As in quickcalc.__init__

    Returns
    -------
    ds_out: xr.Dataset
        Quick INP calculation dataset
    """
    qct = ci_quickcalc(aer_info_dict, T_in, use_ABIFM, RH_in, P_in, ABIFM_delta_t,
                       input_conc_units, input_diam_units)

    key = [key for key in qct.aer.keys()]
    qct.aer = qct.aer[key[0]]
    if isinstance(qct.aer, (AER.multi_logn_AER, AER.logn_AER)):
        qct.aer.ds["diam_bin_edges"] = xr.DataArray(qct.aer.raw_diam, coords={"diam_edge": qct.aer.raw_diam},
                                                    attrs={"units": r"$m$",
                                                           "long_name": "Diameter bin array edges"})
    if qct.use_ABIFM:
        qct.aer.ds.attrs["Parameterization"] = "ABIFM"
        qct.aer.ds["inp_tot"] = xr.DataArray(qct.aer.ds["inp_pct"] * qct.aer.ds["dn_dlogD"].sum() / 100.,
                                             attrs={"long_name": "total INP activated in %.1f s" % qct.delta_t,
                                                    "units": r"$m^{-3}$"})
        qct.aer.ds["inp"] = xr.DataArray(qct.aer.ds["Jhet"] * (qct.aer.ds["dn_dlogD"] * qct.aer.ds["surf_area"]),
                                         attrs={"long_name": "INP activated in %.1f s" % qct.delta_t,
                                                "units": r"$m^{-3}$"})
        qct.aer.ds = qct.aer.ds.assign(variables={"S_ice": qct.ds["S_ice"], "Delta_aw": qct.ds["delta_aw"],
                                                  "T_in": qct.ds["T"], "RH_in": qct.ds["RH"]})
        qct = qct.aer.ds.drop(
            ["dn_dlogD_src", "n_aer", "n_aer_snap", "n_aer_src", "inp_cum_init", "T"]).squeeze(drop=True)
    else:
        qct.aer.ds["inp_tot"] = xr.DataArray(qct.aer.ds["inp_pct"] * qct.aer.ds["dn_dlogD"].sum() / 100.,
                                             attrs={"long_name": "total Diagnostic INP",
                                                    "units": r"$m^{-3}$"})
        qct.aer.ds["inp"] = xr.DataArray(qct.aer.ds["inp_snap"],
                                         attrs={"long_name": "Diagnostic INP",
                                                "units": r"$m^{-3}$"})
        qct.aer.ds = qct.aer.ds.assign(variables={"T_in": qct.ds["T"]})
        if qct.aer.is_INAS:
            qct.aer.ds.attrs["Parameterization"] = "INAS"
            qct.aer.ds = qct.aer.ds.drop(["inp_init"])
        else:
            qct.aer.ds.attrs["Parameterization"] = "INN"
            qct.aer.ds = qct.aer.ds.assign(variables={"P_in": qct.ds["rho"] * qct.Rd * qct.ds["T"]})
        qct = qct.aer.ds.drop(["dn_dlogD_src", "n_aer", "n_aer_snap", "n_aer_src", "inp_cum_init",
                               "inp_src", "inp_snap"]).squeeze(drop=True)
    ds_out = qct
    return ds_out
