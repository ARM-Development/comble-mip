"""
This module includes the AER (aerosol) population class and its sub-classes for different PSDs.
In addition, it includes the Jhet class.
"""
import xarray as xr
import numpy as np
import pandas as pd
import os


class Jhet():
    """
    Class to load Jhet LUT and assign c and m coefficient values based on requested aerosol type.
    """
    def __init__(self, nucleus_type="Illite", coeff_filename=None):
        """
        set the ABIFM linear fit coefficients for the aerosol type.

        Parameters
        ---------
        nucleus_type: str
            aerosol type to use (must match the LUT (not case sensitive).
        coeff_filename: str
            path and filename of Jhet coefficients' Table. By default using the values from Table 1 in Knopf and
            Alpert, 2013, DOI: 10.1039/C3FD00035D.
        """
        self.Jhet_coeff_table = self._load_Jhet_coeff(coeff_filename=coeff_filename)
        self._set_Jhet_coeff(nucleus_type=nucleus_type)

    def _load_Jhet_coeff(self, coeff_filename=None):
        """
        Loads Jhet coefficients tables assuming that the columns represent (from left to right): aerosol type
        (substance), c coefficient, c SD, lower and upper confidence levels for c (respectively), m coefficient,
        m SD, and lower and upper confidence levels for m.

        Parameters
        ---------
        coeff_filename: str
            path and filename of Jhet coefficients' Table. By default using the values from Table 1 in Knopf and
            Alpert, 2013, DOI: 10.1039/C3FD00035D.

        Returns
        -------
        Jhet_coeff_table: DataFrame
        The Jhet coefficients including c (slope) and m (intercept) required for the Jhet calculation.

        """
        if coeff_filename is None:
            coeff_filename = os.path.dirname(__file__) + "/Jhet_coeff.csv"
        Jhet_coeff_table = pd.read_csv(coeff_filename, names=["nucleus_type", "c", "sigma_c", "LCL_c", "UCL_c",
                                                              "m", "sigma_m", "LCL_m", "UCL_m"], index_col=0)
        return Jhet_coeff_table

    def _set_Jhet_coeff(self, nucleus_type="Illite"):
        """
        set the ABIFM linear fit coefficients for the specified aerosol type.
        """
        if nucleus_type.lower() in self.Jhet_coeff_table.index.str.lower():  # allowing case errors
            subs_loc = self.Jhet_coeff_table.index.str.lower() == nucleus_type.lower()
            self.c, self.m = np.float64(self.Jhet_coeff_table.loc[subs_loc, ["c", "m"]].values)[0]
        else:
            raise NameError("aerosol type '%s' not found in Jhet table" % nucleus_type)


class AER_pop():
    """
    class for aerosol population
    """
    def __init__(self, use_ABIFM=None, n_init_max=None, nucleus_type=None, diam=None, dn_dlogD=None, name=None,
                 diam_cutoff=None, T_array=None, singular_fun=None, singular_scale=None, psd={},
                 n_init_weight_prof=None, entrain_psd=None, entrain_to_cth=None, ci_model=None):
        """
        aerosol population namelist

        Parameters
        ----------
        self.is_INAS: bool
            True - using INAS (aerosol are prognosed like in ABIFM).
        use_ABIFM: bool
            True - use ABIFM, False - use singular.
        scheme: str
            "ABIFM" or "singular" or None when no input is provided.
        nucleus_type: str
            type of nucleus required for Jhet calculation (--ABIFM--)
        Jhet: Jhet class object
            To use with ABIFM (--ABIFM--)
        diam_cutff: float or tuple
            If float, minimum particle diameter to consider (0.5 by default, i.e., consider all sizes).
            If tuple, then lower and upper diameter limits (--singular--).
        T_array: list or ndarray
            discrete temperature [K] array for INP parametrization (--singular--).
        singular_fun: str or lambda function
            If str, then:
                1. use "D2010" to use eq. 1 in DeMott et al., 2010.
                2. "D2015" to use eq. 2 in DeMott et al..
                3. "D2010fit" to use the temperature dependence fit from fig. 2 caption in DeMott et al., 2010.
                4. "ND2012 to use surface area temperature-based fit (eq. 5) in Niemand et al., JAS, 2012.
                5. "SC2020" to use surface area temperature-based fit (eq. 5) in Schill et al., PNAS, 2020.
                6. "AT2013" to use surface area temperature_based fit (eq.6) in Atkinson et al., NATURE, 2013.
                7. "AL2022" to use fit in Peter et al., Science Advances, 2022.
                8. "SW2017" to use fit in Swarup China et al., JGR, 2017.
                9. "MC2018" to use surface area temperature_based fit fig. 8 caption in McCluskey et al., JGR, 2018.
                Use a lambda function for INP parametrization typically as a function of T (--singular--).
            Use "D2010" (default) if None.
            Notes:
            The D2015 has default values of the five coefficients from eq. 2 (cf - calibration correction factor,
            alpha, beta, gamma, delta); these might be coded as optional input parameters for the AER class in
            the future.
            "D2010fit" does not consider aerosol PSDs.
        singular_scale: float
            "Artificial" scale factor for 'singular_fun' or for Jhet (1 by default).
        n_init_max: float
            total initial aerosol concentration [m-3].
        diam: list or ndarray or scalar
            discrete particle diameter array [um]
        dn_dlogD: list or ndarray or scalar
            discrete particle number per size bin (sums to n_init_max) [m-3]
        psd_type: str
            population type e.g., "mono", "logn", "multi_logn", "custom".
        psd: dict
            dictionary providing psd parameter information enabling full psd reproduction.
        entrain_psd: dict
            dictionary providing psd parameter for entrained aerosol.
        entrain_to_cth: bool or int or None
            determines where to entrain aerosol (cloud top / mixing layer base / specific height index).
            If specified, then overrides the ci_model input value.
            If None, using the ci_model object value.
        name: str
            population name (or tag).
        n_init_weight_prof: dict or None
               a dict with keys "height" and "weight". Each key contains a list or np.ndarray of length s (s > 1)
               determining PSD heights [m] and weighting profiles. Weights are applied on n_init such that
               n_init(z) = n_init_max * weighting_factor(z), i.e., a weighted_aer_prof filled with ones means
               that n_init(z) = n_init_max.
               Weights are generally expected to have values between 0 and 1. If at least one weight value > 1,
               then the profile is normalized such that the maximum value equals 1. heights are interpolated
               between the specified heights, and the edge values are used for extrapolation (can be used to set
               different aerosol source layers at model initialization, and combined with turbulence weighting,
               allows the emulation of cloud-driven mixing.
        ci_model: ci_model class
            Containing variables such as the requested domain size, LES time averaging option
            (ci_model.t_averaged_les), custom or LES grid information (ci_model.custom_vert_grid),
            and LES xr.DataSet object(ci_model.les) after being processed.
            All these required data are automatically set when a ci_model class object is assigned
            during model initialization.
        ds: Xarray Dataset
            will be shaped and incorporate all aerosol population in the domain:
            ABIFM: height x time x diameter
            singular: height x time x T
        """
        # set attributes
        self.is_INAS = False  # Only becomes True in case of INAS
        if use_ABIFM is not None:
            if not np.logical_or(not use_ABIFM,
                                 np.logical_and(nucleus_type is not None, diam is not None)):
                raise RuntimeError("nucleus type and diameter must be specified in ABIFM")
            if use_ABIFM:
                self.scheme = "ABIFM"
                self.nucleus_type = nucleus_type
                self.Jhet = Jhet(nucleus_type=nucleus_type)
            else:
                self.scheme = "singular"
                if diam_cutoff is None:
                    self.diam_cutoff = 0.5e-6
                else:
                    self.diam_cutoff = diam_cutoff
                if singular_fun is None:
                    singular_fun = "D2010"  # set by default to DeMott et al. (2010)
            if singular_scale is None:
                self.singular_scale = 1.
            else:
                self.singular_scale = singular_scale
        else:
            self.scheme = None
        self.n_init_weight_prof = n_init_weight_prof
        self.n_init_max = n_init_max
        if isinstance(diam, (float, int)):
            self.diam = [diam]
        else:
            self.diam = diam
        if isinstance(dn_dlogD, (float, int)):
            self.dn_dlogD = [dn_dlogD]
        else:
            self.dn_dlogD = dn_dlogD
        self.psd = psd
        self.entrain_psd = entrain_psd
        if "type" in psd.keys():  # assuming that the __init__ method is invoked from an inhertied classes.
            self.psd_type = psd["type"]
        if name is None:
            self._random_name()
        else:
            self.name = name
        if 'src_weight_time' in entrain_psd.keys():
            self.src_weight_time = entrain_psd['src_weight_time']
        else:
            self.src_weight_time = None
        if entrain_to_cth is None:
            self.entrain_to_cth = True
        else:
            self.entrain_to_cth = entrain_to_cth

        # Assign aerosol dataset
        self.ds = xr.Dataset()
        self.ds = self.ds.assign_coords({"diam": np.ones(1) * diam})
        self.ds["dn_dlogD"] = xr.DataArray(np.ones(1) * dn_dlogD, dims=self.ds["diam"].dims)
        self.ds["dn_dlogD"].attrs["units"] = "$m^{-3}$"
        self.ds["dn_dlogD"].attrs["long_name"] = "Particle number concentration per diameter bin"
        self.ds["dn_dlogD_src"] = xr.DataArray(np.ones(1) * entrain_psd['dn_dlogD'], dims=self.ds["diam"].dims)
        self.ds["dn_dlogD_src"].attrs["units"] = "$m^{-3}$"
        self.ds["dn_dlogD_src"].attrs["long_name"] = "Source (entrained) number concentration per diameter bin"
        self._calc_surf_area()

        # Use the ci_model class object if provided to init the aerosol array (start with height-time coords).
        # Also determine entrainment height based on ci_model if entrain_to_cth is None.
        if ci_model is not None:
            if entrain_to_cth is None:
                self.entrain_to_cth = ci_model.entrain_to_cth
            self.ds = self.ds.assign_coords({"height": ci_model.ds["height"].values,
                                             "time": ci_model.ds["time"].values,
                                             "t_out": ci_model.ds["t_out"].values})
            if T_array is None:
                self._set_T_array(ci_model)  # set T bin array with ∆T that follows a geometric progression.
            else:
                self.T_array = T_array
            if use_ABIFM is True:
                self._init_aer_Jhet_ABIFM_arrays(ci_model)
            elif use_ABIFM is False:
                self._set_aer_conc_fun(singular_fun)
                self._init_aer_singular_array(ci_model)
            self.ds["height"].attrs["units"] = "$m$"
            self.ds["time"].attrs["units"] = "$s$"
            self.ds["t_out"].attrs["units"] = "$s$"
            self.ds["height_km"] = self.ds["height"].copy() / 1e3  # add coordinates for height in km.
            self.ds = self.ds.assign_coords(height_km=("height", self.ds["height_km"].values))
            self.ds["height_km"].attrs["units"] = "$km$"
            self.ds["time_h"] = self.ds["time"].copy() / 3600  # add coordinates for time in h.
            self.ds = self.ds.assign_coords(time_h=("time", self.ds["time_h"].values))
            self.ds["time_h"].attrs["units"] = "$h$"
            self.ds["t_out_h"] = self.ds["t_out"].copy() / 3600  # add coordinates for time in h.
            self.ds = self.ds.assign_coords(t_out_h=("t_out", self.ds["t_out_h"].values))
            self.ds["t_out_h"].attrs["units"] = "$h$"

            if ci_model.prognostic_ice:
                self.ds["ice_snap"].attrs["units"] = "$m^{-3}$"
                self.ds["ice_snap"].attrs["long_name"] = "prognosed ice number concentration (snapshot)"
                self.ds["Ni_nuc"] = xr.DataArray(
                    np.full((*ci_model.dt_out.shape, *self.ds["ice_snap"].shape), np.nan),
                    dims=(*self.ds["t_out"].dims, *self.ds["ice_snap"].dims))

        else:

            print("'ci_model' object not provided - not assigning aerosol concentration array")

        # Set coordinate attributes
        self.ds["diam"].attrs["units"] = r"$m$"
        self.ds["diam"].attrs["long_name"] = "Bin-middle particle diameter"
        self.ds["diam_um"] = self.ds["diam"].copy() * 1e6  # add coordinates for diameter in microns
        self.ds = self.ds.assign_coords(diam_um=("diam", self.ds["diam_um"].values))
        self.ds["diam_um"].attrs["units"] = r"$\mu m$"

    def _random_name(self):
        """
        Generate random string name for population
        """
        self.name = self.psd_type + \
            "_%05d" % np.random.randint(1e4 - 1)  # generate random population number if not provided.

    def _calc_surf_area(self):
        """
        Calculate surface area per particle [m2] corresponding to each diameter to use with Jhet [m-2 * s-1]
        """
        self.ds["surf_area"] = xr.DataArray(np.pi * self.ds["diam"] ** 2, dims=self.ds["diam"].dims)
        self.ds["surf_area"].attrs["units"] = "$m^2$"
        self.ds["surf_area"].attrs["long_name"] = "Surface area per particle diameter"

    def _set_T_array(self, ci_model, dT0=0.1, dT_exp=1.05, T_max=268.15):
        """
        Sets the temperature array for singular using geometric progression bins (considering that n_AER(T)
        parameterizations typically follow a power law).
        The minimum temperature (leftmost bin edge) is set based on the minimum temperature of the model
        domain (floored to the 1st decimal).

        Parameters
        ---------
        dT0: float
            ∆T between the first and second temperature bin edges
        dT_exp: float
            exponent for ∆T (the ratio of ∆T between consecutive bins).
        T_max: float
            maximum temperature (in K) for T array (the edge of the final bin can be larger than T_max).
        """
        if ci_model.ds["T"].min() >= T_max:
            raise RuntimeError('Minimum LES-informed temperature must be larger than %.2f K in'
                               ' singular mode to allow any aerosol to activate' % T_max)
        T_min = 0. + np.maximum(ci_model.ds["T"].min().values, 233.15)
        T_array = np.array([T_min])
        while T_array[-1] < T_max:
            T_array = np.append(T_array, [T_array[-1] + dT0 * dT_exp ** (len(T_array) - 1)])
        self.T_array = T_array

    def _set_aer_conc_fun(self, singular_fun):
        """
        Set the INP initialization function for the singular approach.

        Parameters
        ---------
        singular_fun: str or lambda function
            If str, then:
                1. use "D2010" to use eq. 1 in DeMott et al., 2010.
                2. "D2015" to use eq. 2 in DeMott et al..
                3. "D2010fit" to use the temperature dependence fit from fig. 2 caption in DeMott et al., 2010.
                4. "ND2012" to use surface area temperature-based fit (eq. 5) in Niemand et al., JAS, 2012.
                5. "SC2020" to use surface area temperature-based fit (eq. 5) in Schill et al., PNAS, 2020.
                6. "AT2013" to use surface area temperature_based fit (eq.6) in Atkinson et al., NATURE, 2013.
                7. "AL2022" to use fit in Peter et al., Science Advances, 2022.
                8. "SW2017" to use fit in Swarup China et al., JGR, 2017.
                9. "MC2018" to use surface area temperature_based fit fig. 8 caption in McCluskey et al., JGR, 2018.
            Use a lambda function for INP parametrization typically as a function of T (--singular--).
            Use "D2010" (default) if None.
            Notes:
            The D2015 has default values of the five coefficients from eq. 2 (cf - calibration correction factor,
            alpha, beta, gamma, delta); these might be coded as optional input parameters for the AER class in
            the future.
            "D2010fit" does not consider aerosol PSDs.
        """
        if isinstance(singular_fun, str):
            if singular_fun == "D2010":
                self.singular_fun = lambda Tk, n_aer05: 0.0000594 * (273.16 - Tk) ** 3.33 * (n_aer05 * 1e-6) ** \
                    (0.0264 * (273.16 - Tk) + 0.0033) * 1e3  # DeMott et al. (2010)
            elif singular_fun == "D2015":
                self.singular_fun = \
                    lambda Tk, n_aer05, cf=3., alpha=0., beta=1.25, gamma=0.46, delta=-11.6: \
                    cf * (n_aer05 * 1e-6) ** (alpha * (273.16 - Tk) + beta) * \
                    np.exp(gamma * (273.16 - Tk) + delta) * 1e3  # DeMott et al. (2015)
            elif singular_fun == "D2010fit":
                self.singular_fun = \
                    lambda Tk: 0.117 * np.exp(-0.125 * (Tk - 273.2)) * 1e3  # DeMott et al. (2010) fig. 2 fit
            elif singular_fun == "ND2012":
                self.singular_fun = lambda Tk, s_area: \
                    np.exp(-0.517 * (Tk - 273.15) + 8.934) * s_area  # INAS Niemand et al. (2012)
            elif singular_fun == "SC2020":
                # INAS soot Schill et al. (2020)
                self.singular_fun = lambda Tk, s_area: \
                    np.exp(1.844 - 0.687 * (Tk - 273.15) - 0.00597 * (Tk - 273.15)**2) * s_area
            elif singular_fun == "AT2013":
                # K feldspar Atkinson et al.(2013; valid between 248 and 268 K)
                self.singular_fun = lambda Tk, s_area: \
                    np.exp(-1.038 * Tk + 275.26) * s_area * 1e4
            elif singular_fun == "AL2022":
                # SSA ns fitting from Peter et al. (2022)
                self.singular_fun = lambda Tk, s_area: \
                    10**(24.02526 * (1-(np.exp(9.550426 - 5723.265 / Tk + 3.53068 * np.log(Tk) -
                    0.00728332 * Tk) /
             (np.exp(54.842763 - 6763.22 / Tk -
              4.210 * np.log(Tk) + 0.000367 * Tk +
              np.tanh(0.0415 * (Tk - 218.8)) * (53.878 - 1331.22 / Tk - 9.44523 * np.log(Tk) + 0.014025 * Tk)))
             ))-2.26105)* 1e4* s_area
            elif singular_fun == "SW2017":
             # Oragnic ns fitting from Swarup China et al. (2017)
                self.singular_fun = lambda Tk, s_area: \
                    10**(66.90259 * (1-(np.exp(9.550426 - 5723.265 / Tk + 3.53068 * np.log(Tk) -
                    0.00728332 * Tk) /
             (np.exp(54.842763 - 6763.22 / Tk -
              4.210 * np.log(Tk) + 0.000367 * Tk +
              np.tanh(0.0415 * (Tk - 218.8)) * (53.878 - 1331.22 / Tk - 9.44523 * np.log(Tk) + 0.014025 * Tk)))
             ))-12.322)* 1e4* s_area
            elif singular_fun == "MC2018":
                self.singular_fun = lambda Tk, s_area: \
                    np.exp(-0.545 * (Tk-273.15) + 1.0125) * s_area 
            elif singular_fun == "0.5PT2022":
                # SSA ns fitting from Peter et al. (2022)
                self.singular_fun = lambda Tk, s_area: \
                    10**(24.02526 * (1-(np.exp(9.550426 - 5723.265 / Tk + 3.53068 * np.log(Tk) -
                    0.00728332 * Tk) /
             (np.exp(54.842763 - 6763.22 / Tk -
              4.210 * np.log(Tk) + 0.000367 * Tk +
              np.tanh(0.0415 * (Tk - 218.8)) * (53.878 - 1331.22 / Tk - 9.44523 * np.log(Tk) + 0.014025 * Tk)))
             ))-2.26105)* 1e4*0.5*s_area
            elif singular_fun == "0.5MC2018":
                self.singular_fun = lambda Tk, s_area: \
                    np.exp(-0.545 * (Tk-273.15) + 1.0125) *0.5* s_area     
            else:
                raise NameError("The singular treatment %s is not implemented in the model. Check the \
                                input string." % singular_fun)
        else:  # assuming lambda function
            self.singular_fun = singular_fun

    def _init_aer_singular_array(self, ci_model, std_L_to_L=True):
        """
        initialize the aerosol and INP concentration arrays as well as other diagnostic arrays for singular
        (height x time x temperature) or (height x time x diam).

        Parameters
        ---------
        ci_model: ci_model class object
            Cloud-ice nucleation model object including all model initialization and prognosed field datasets.
        std_L_to_L: bool [singular]
            True - converting number concentration parameterization from standard liter to SI liter
        """
        if ci_model.prognostic_inp:
            self.ds = self.ds.assign_coords({"T": self.T_array})
            tmp_inp_array, tmp_inp_array_src = \
                np.zeros((self.ds["height"].size, self.ds["T"].size)), np.zeros((self.ds["T"].size))
        if self.singular_fun.__code__.co_argcount > 1:
            if 'n_aer05' in self.singular_fun.__code__.co_varnames:  # 2nd argument is aerosol conc. above cutoff
                if isinstance(self.diam_cutoff, float):
                    input_2 = np.sum(self.ds["dn_dlogD"].sel({"diam": slice(self.diam_cutoff, None)}).values)
                    input_2_src = \
                        np.sum(self.ds["dn_dlogD_src"].sel({"diam": slice(self.diam_cutoff, None)}).values)
                else:  # assuming 2-element tuple
                    input_2 = np.sum(self.ds["dn_dlogD"].sel({"diam": slice(self.diam_cutoff[0],
                                                                            self.diam_cutoff[1])}).values)
                    input_2_src = np.sum(self.ds["dn_dlogD_src"].sel({"diam": slice(self.diam_cutoff[0],
                                                                                    self.diam_cutoff[1])}).values)
                self.n_aer05_frac = input_2 / np.sum(self.ds["dn_dlogD"].values)
                if ci_model.prognostic_inp:
                    input_2 = np.ones((self.ds["height"].size, self.ds["T"].size)) * input_2
                    tmp_n_inp = np.flip(self.singular_fun(np.tile(np.expand_dims(self.ds["T"].values, axis=0),
                                        (self.ds["height"].size, 1)), input_2), axis=1)  # start at max temperature
                    input_2_src = np.ones((self.ds["T"].size)) * input_2_src
                    tmp_n_inp_src = np.flip(self.singular_fun(self.ds["T"].values, input_2_src), axis=0)

                    # weight array vertically.
                    if self.n_init_weight_prof is not None:
                        tmp_n_inp = np.tile(np.expand_dims(self._weight_aer_h_or_t(False), axis=1),
                                            (1, self.ds["T"].size)) * tmp_n_inp

                    # Convert INN parameterization from SL-1 to L-1 (using lowest T & rho in time 0 for src)
                    if std_L_to_L:
                        print("Converting INP from standard liter to SI liter")
                        tmp_n_inp = self.convert_SL_to_L(
                            np.tile(np.expand_dims(ci_model.ds["rho"].values[:, 0], axis=1),
                                    (1, self.ds["T"].size)),
                            np.tile(np.expand_dims(ci_model.ds["T"].values[:, 0], axis=1), (1, self.ds["T"].size)),
                            conc_field_in=tmp_n_inp)
                        tmp_n_inp_src = self.convert_SL_to_L(
                            ci_model.ds["rho"].values[0, 0], ci_model.ds["T"][0, 0].values,
                            conc_field_in=tmp_n_inp_src)

            elif 's_area' in self.singular_fun.__code__.co_varnames:  # 2nd argument is surface area
                self.is_INAS = True
                self._init_aer_Jhet_ABIFM_arrays(ci_model)  # Also alocate aerosol concentration array
                if ci_model.prognostic_inp:
                    tmp_inp_array = np.zeros((self.ds["height"].size, self.ds["diam"].size, self.ds["T"].size))
                    tmp_inp_array_src = np.zeros((self.ds["diam"].size, self.ds["T"].size))
                    tmp_n_inp = self.singular_fun(np.tile(np.expand_dims(np.flip(self.ds["T"].values), axis=0),
                                                          (self.ds["diam"].size, 1)),
                                                  np.tile(np.expand_dims((self.ds["surf_area"] *
                                                                          self.ds["dn_dlogD"]).values, axis=1),
                                                          (1, self.ds["T"].size)))
                    tmp_n_inp = np.tile(np.expand_dims(tmp_n_inp, axis=0), (self.ds["height"].size, 1, 1))
                    tmp_n_inp_src = self.singular_fun(np.tile(np.expand_dims(np.flip(self.ds["T"].values), axis=0),
                                                              (self.ds["diam"].size, 1)),
                                                      np.tile(np.expand_dims(
                                                          (self.ds["surf_area"] *
                                                           self.ds["dn_dlogD_src"]).values,
                                                          axis=1), (1, self.ds["T"].size)))

                    # weight array vertically.
                    if self.n_init_weight_prof is not None:
                        tmp_n_inp = np.tile(np.expand_dims(self._weight_aer_h_or_t(False), axis=(1, 2)),
                                            (1, self.ds["diam"].size, self.ds["T"].size)) * tmp_n_inp

        elif ci_model.prognostic_inp:  # single input (temperature)
            tmp_n_inp = np.tile(np.expand_dims(np.flip(self.singular_fun(self.ds["T"].values)), axis=0),
                                (self.ds["height"].size, 1))  # start at highest temperatures
        if ci_model.prognostic_inp:
            if self.singular_scale != 1.:
                tmp_n_inp *= self.singular_scale
                tmp_n_inp_src *= self.singular_scale
            if not self.is_INAS:
                self.ds["inp_cum_init"] = xr.DataArray(np.flip(tmp_n_inp, axis=-1), dims=("height", "T"))
                tmp_inp_array[:, 0] = tmp_n_inp[:, 0]
                tmp_inp_array_src[0] = tmp_n_inp_src[0]
                for ii in range(1, self.ds["T"].size):
                    tmp_inp_array[:, ii] = tmp_n_inp[:, ii] - tmp_inp_array[:, :ii].sum(axis=1)
                    tmp_inp_array_src[ii] = tmp_n_inp_src[ii] - tmp_inp_array_src[:ii].sum()
            else:
                self.ds["inp_cum_init"] = xr.DataArray(np.flip(tmp_n_inp, axis=-1), dims=("height", "diam", "T"))
                tmp_inp_array[:, :, 0], tmp_inp_array_src[:, 0] = tmp_n_inp[:, :, 0], tmp_n_inp_src[:, 0]
                for ii in range(1, self.ds["T"].size):
                    tmp_inp_array[:, :, ii] = tmp_n_inp[:, :, ii] - tmp_inp_array[:, :, :ii].sum(axis=2)
                    tmp_inp_array_src[:, ii] = tmp_n_inp_src[:, ii] - tmp_inp_array_src[:, :ii].sum(axis=1)

            self.ds["T_C"] = self.ds["T"].copy() - 273.15  # add coordinates for temperature in Celsius
            self.ds = self.ds.assign_coords(T_C=("T", self.ds["T_C"].values))
            self.ds["T_C"].attrs["units"] = "$° C$"
            self.ds["T"].attrs["units"] = "$K$"  # set coordinate attributes.

        if self.src_weight_time is not None:
            src_weight_tseries = \
                self._weight_aer_h_or_t(False, use_height=False, alternative_dict=self.src_weight_time)
        if not self.is_INAS:  # singular
            self.ds["n_aer_snap"] = xr.DataArray(
                np.full(self.ds["height"].size, np.sum(self.ds["dn_dlogD"].values)), dims=("height"))
            if self.n_init_weight_prof is not None:
                self.ds["n_aer_snap"] = \
                    self._weight_aer_h_or_t(False) * self.ds["n_aer_snap"]
            self.ds["n_aer_snap"].attrs["units"] = "$m^{-3}$"
            self.ds["n_aer_snap"].attrs["long_name"] = "aerosol number concentration (snapshot)"
            self.ds["n_aer"] = xr.DataArray(
                np.zeros((*self.ds["n_aer_snap"].shape, *ci_model.dt_out.shape)),
                dims=(*self.ds["n_aer_snap"].dims, *self.ds["t_out"].dims))
            self.ds["n_aer"].attrs["units"] = "$m^{-3}$"
            self.ds["n_aer"].attrs["long_name"] = "aerosol number concentration"

            self.ds["n_aer_src"] = xr.DataArray(np.tile(np.sum(self.ds["dn_dlogD_src"].values),
                                                        (self.ds["time"].size)), dims=("time"))
            if self.src_weight_time is not None:
                self.ds["n_aer_src"].values *= src_weight_tseries
            self.ds["n_aer_src"].attrs["units"] = "$m^{-3}$"
            self.ds["n_aer_src"].attrs["long_name"] = "aerosol source number concentration per diameter bin"
            if ci_model.prognostic_inp:
                self.ds["ns_raw"] = xr.DataArray(self.singular_fun(ci_model.ds["T"],
                                                                   np.tile(np.expand_dims(input_2[:, 0],
                                                                                          axis=1),
                                                                           (1, ci_model.ds["time"].size))) /
                                                 (self.ds["dn_dlogD"] * self.ds["surf_area"]).sum(),
                                                 dims=("height", "time"))
                self.ds["ns_raw"].attrs["long_name"] = "INAS ns-equivalent singular treatment"
                self.ds["inp_snap"] = xr.DataArray(tmp_inp_array, dims=("height", "T"))
                self.ds["inp_snap"].values = np.flip(self.ds["inp_snap"].values, axis=-1)
                self.ds["inp_snap"].attrs["units"] = "$m^{-3}$"
                self.ds["inp_snap"].attrs["long_name"] = "prognosed INP number concentration (snapshot)"
                self.ds["inp_src"] = xr.DataArray(np.flip(np.tile(np.expand_dims(
                    tmp_inp_array_src, axis=0), (self.ds["time"].size, 1)), axis=-1), dims=("time", "T"))
                if self.src_weight_time is not None:
                    self.ds["inp_src"].values *= np.tile(
                        np.expand_dims(src_weight_tseries, axis=-1), (1, self.ds["T"].size))
                self.ds["inp_pct"] = xr.DataArray(self.singular_fun(ci_model.ds["T"],
                                                                    np.tile(np.expand_dims(input_2[:, 0],
                                                                                           axis=1),
                                                                            (1, ci_model.ds["time"].size))) /
                                                  self.ds["dn_dlogD"].sum() * 100., dims=("height", "time"))
                if ci_model.prognostic_ice:
                    self.ds["ice_snap"] = xr.DataArray(np.zeros(self.ds["inp_snap"].shape),
                                                       dims=("height", "T"))
        elif ci_model.prognostic_inp:  # INAS
            self.ds["ns_raw"] = xr.DataArray(self.singular_fun(ci_model.ds["T"], 1), dims=("height", "time"))
            self.ds["ns_raw"].attrs["long_name"] = "INAS ns"
            self.ds["inp_snap"] = xr.DataArray(tmp_inp_array, dims=("height", "diam", "T"))
            self.ds["inp_snap"].values = np.flip(self.ds["inp_snap"].values, axis=-1)
            self.ds["inp_snap"].attrs["units"] = "$m^{-3}$"
            self.ds["inp_src"] = xr.DataArray(np.flip(np.tile(np.expand_dims(tmp_inp_array_src, axis=0),
                                                              (ci_model.ds["time"].size, 1, 1)),
                                                      axis=-1), dims=("time", "diam", "T"))
            if self.src_weight_time is not None:
                self.ds["inp_src"].values *= np.tile(
                    np.expand_dims(src_weight_tseries, axis=(1, 2)), (1, self.ds["diam"].size, self.ds["T"].size))
            self.ds["inp_snap"].attrs["long_name"] = "prognosed INP number concentration (snapshot)"
            self.ds["inp_init"] = self.ds["inp_snap"].copy()  # copy of initial INP (might be used for entrainment)
            self.ds["inp_init"].attrs["long_name"] = "prognosed INP number concentration (initial)"
            self.ds["inp_pct"] = self.ds["ns_raw"] * (self.ds["dn_dlogD"] * self.ds["surf_area"]).sum() / \
                self.ds["dn_dlogD"].sum() * 100.
            if ci_model.prognostic_ice:
                self.ds["ice_snap"] = xr.DataArray(np.zeros(self.ds["inp_snap"].shape),
                                                   dims=("height", "diam", "T"))
        if ci_model.prognostic_inp:
            self.ds["inp_src"].attrs["units"] = "$m^{-3}$"
            self.ds["inp_src"].attrs["long_name"] = "INP source number concentration"
        if ci_model.prognostic_inp:
            self.ds["ns_raw"].values = np.where(ci_model.ds["ql"].values >= ci_model.in_cld_q_thresh,
                                                self.ds["ns_raw"].values, 0)  # crop in-cloud pixels
            self.ds["inp_pct"].values = np.where(ci_model.ds["ql"].values >= ci_model.in_cld_q_thresh,
                                                 self.ds["inp_pct"].values, 0)  # crop in-cloud pixels
            self.ds["inp_pct"].attrs["units"] = "$percent$"
            self.ds["inp_pct"].attrs["long_name"] = ("INP parameterization percentage relative to total initial"
                                                     " aerosol concentrations")
            self.ds["inp_cum_init"].attrs["units"] = "$m^{-3}$"
            self.ds["inp_cum_init"].attrs["long_name"] = \
                "Initial cumulative (over T) INP array"
            self.ds["ns_raw"].attrs["units"] = "$m^{-2}$"

    def _init_aer_Jhet_ABIFM_arrays(self, ci_model, pct_const=None):
        """
        initialize the aerosol array (height x time x diameter) and the Jhet arrays (for ABIFM) assuming that
        dn_dlogD has been calculated and that the ci_model object was already generated (with delta_aw, etc.).

        Parameters
        ----------
        ci_model: ci_model class object
            Cloud-ice nucleation model object including all model initialization and prognosed field datasets.
        pct_const: float or None  (--ABIFM--)
            time constant to use for the diagnostic INP percentage calculation.
            If None (default), using the model's delta_t
        """
        self.ds["n_aer_snap"] = xr.DataArray(np.tile(np.expand_dims(self.ds["dn_dlogD"].values, axis=0),
                                                     (self.ds["height"].size, 1)),
                                             dims=("height", "diam"))
        self.ds["n_aer_snap"].attrs["units"] = "$m^{-3}$"
        self.ds["n_aer_snap"].attrs["long_name"] = "aerosol number concentration per diameter bin (snapshot)"
        self.ds["n_aer"] = xr.DataArray(
            np.zeros((self.ds["n_aer_snap"].shape[0], *ci_model.dt_out.shape, self.ds["n_aer_snap"].shape[1])),
            dims=(*self.ds["height"].dims, *self.ds["t_out"].dims,
                  *self.ds["diam"].dims))
        self.ds["n_aer"].attrs["units"] = "$m^{-3}$"
        self.ds["n_aer"].attrs["long_name"] = "aerosol number concentration"
        self.ds["n_aer_src"] = xr.DataArray(np.tile(np.expand_dims(self.ds["dn_dlogD_src"].values, axis=0),
                                                    (self.ds["time"].size, 1)), dims=("time", "diam"))
        if self.src_weight_time is not None:
            src_weight_tseries = \
                self._weight_aer_h_or_t(False, use_height=False, alternative_dict=self.src_weight_time)
            self.ds["n_aer_src"].values *= np.tile(
                np.expand_dims(src_weight_tseries, axis=(-1)), (1, self.ds["diam"].size))
        self.ds["n_aer_src"].attrs["units"] = "$m^{-3}$"
        self.ds["n_aer_src"].attrs["long_name"] = "aerosol source number concentration per diameter bin"
        if ci_model.use_ABIFM:
            self.ds["Jhet"] = 10.**(self.Jhet.c + self.Jhet.m * ci_model.ds["delta_aw"]) * 1e4  # calc Jhet
            if self.singular_scale != 1.:
                self.ds["Jhet"].values *= self.singular_scale
            self.ds["Jhet"].attrs["units"] = "$m^{-2} s^{-1}$"
            self.ds["Jhet"].attrs["long_name"] = "Heterogeneous ice nucleation rate coefficient"
            if pct_const is None:
                pct_const = ci_model.delta_t
            self.ds["inp_pct"] = self.ds["Jhet"] * \
                (self.ds["dn_dlogD"] * self.ds["surf_area"]).sum() * pct_const / self.ds["dn_dlogD"].sum() * 100.
            self.ds["inp_pct"].attrs["units"] = "$percent$"
            self.ds["inp_pct"].attrs["long_name"] = \
                ("INP percentage relative to total initial aerosol (using a time"
                 " constant of %.0f s)" % pct_const)
            if self.n_init_weight_prof is not None:
                self._weight_aer_h_or_t()

            if ci_model.prognostic_ice:
                self.ds["ice_snap"] = xr.DataArray(np.zeros(self.ds["n_aer_snap"].shape),
                                                   dims=("height", "diam"))

            # Generate a diagnostic initial  INP equivalent to the singular approaches (for comparison purposes)
            self.ds = self.ds.assign_coords({"T": self.T_array})
            aw = 1  # liquid water activity at water saturation
            delta_aw = aw - ci_model.calc_a_ice_w(self.T_array)  # delta_aw equivalent to the singular T array
            Jhet_tmp = 10.**(self.Jhet.c + self.Jhet.m * delta_aw) * 1e4  # corresponding Jhet in SI units
            INP_array = np.tile(np.expand_dims(Jhet_tmp, axis=(0, 1)),
                                (*self.ds["n_aer_snap"].shape, 1)) * self.singular_scale * \
                np.tile(np.expand_dims(self.ds["n_aer_snap"] * self.ds["surf_area"], axis=2),
                        (1, 1, self.ds["T"].size)) * pct_const
            self.ds["inp_cum_init"] = xr.DataArray(INP_array, dims=("height", "diam", "T"))
            self.ds["inp_cum_init"].attrs["units"] = "$m^{-3}$"
            self.ds["inp_cum_init"].attrs["long_name"] = \
                "Initial cumulative (over T) INP array "
            "(equivalent to singular approaches; using a time constant of %.0f s)" % pct_const

    def _weight_aer_h_or_t(self, set_internally=True, use_height=True, alternative_dict=None):
        """
        apply weights on initial aerosol profile (weighting on n_init_max). If using singular then returning the
        weights profile

        Parameters
        ---------
        set_internally: bool
            True - set n_aer internally (use ABIFM is True), False - return weighted array (for singular).
            If False, simply returning the weighted profile or time series.
        alternative_dict: dict or None
            Alternative dict defining the height/time and weight to be interpolated.
            Using self.n_init_weight_prof if None.

        Returns
        -------
        weight_arr_interp: np.ndarray (--singular--)
            weight profile with height coordinates
        """
        if use_height:
            Dim = "height"
        else:
            Dim = "time"
        if alternative_dict is None:
            alternative_dict = self.n_init_weight_prof
        if not np.all([x in alternative_dict.keys() for x in [Dim, "weight"]]):
            raise KeyError(f'Weighting the aerosol profiles requires the keys {Dim} and "weight"')
        if not np.logical_and(len(alternative_dict[Dim]) > 1,
                              len(alternative_dict[Dim]) == len(alternative_dict["weight"])):
            raise ValueError(f"weights and {Dim}s must have the same length > 1")
        if np.any(alternative_dict["weight"] < 0.):
            raise ValueError("weight values must by > 0 (at least one negative value was entered)")
        if np.any(alternative_dict["weight"] > 1.):
            print("At least one specified weight > 1 (max value = %.1f); normalizing weights such that \
                  max weight == 1" % alternative_dict["weight"].max())
            alternative_dict["weight"] = alternative_dict["weight"] / \
                alternative_dict["weight"].max()
        weight_arr_interp = np.interp(self.ds[Dim], alternative_dict[Dim], alternative_dict["weight"])
        if set_internally:
            self.ds["n_aer_snap"] = np.tile(np.expand_dims(weight_arr_interp, axis=1),
                                            (1, self.ds["diam"].size)) * self.ds["n_aer_snap"]
        else:  # Relevant for singular when considering particle diameters (e.g., D2010, D2015).
            return weight_arr_interp

    @staticmethod
    def convert_SL_to_L(rho_in, T_in, post_1982=True, conc_field_in=None):
        """
        Convert a number concentration field from standard liter to SI liter (useful for INP in INN params).

        Parameters
        ---------
        rho_in: np.ndarray or float
            input density [kg m-3]
        T_in: np.ndarray or float
            input temperature [K]
        post_1982: bool
            True - P0 post 1982 (100000 Pa), False - P0 pre 1982 (101325 Pa)
        conc_field_in: np.ndarray
            number concentration field in standard liter

        Returns
        -------
        conc_field_out: np.ndarray
            number concentration field in SI liter
        """
        Rd = 287.052874  # J kg-1 K-1
        if post_1982:
            P0 = 100000  # in Pa
        else:
            P0 = 101325  # in Pa
        P = rho_in * Rd * T_in  # calculating pressure [Pa]
        conc_field_out = conc_field_in * (T_in / 273.15) * (P0 / P)
        return conc_field_out


class mono_AER(AER_pop):
    """
    Uniform (fixed) aerosol diameter.
    """
    def __init__(self, use_ABIFM, n_init_max, nucleus_type=None, name=None,
                 diam_cutoff=None, T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, entrain_psd=None, entrain_to_cth=None, ci_model=None):
        """
        Parameters as in the 'AER_pop' class (fixed diameter can be specified in the 'psd' dict under the 'diam'
        key or in the diam).
        """
        psd.update({"type": "mono"})  # require type key consistency
        if "diam" not in psd.keys():
            raise KeyError('mono-dispersed PSD processing requires the "diam" fields')
        diam = psd["diam"]
        dn_dlogD = np.array(n_init_max)
        if entrain_psd is None:  # Entrained aerosol PSD
            entrain_psd = {"dn_dlogD": np.copy(dn_dlogD)}
        else:
            entrain_psd["dn_dlogD"] = np.array(entrain_psd["n_init_max"])

        super().__init__(use_ABIFM=use_ABIFM, n_init_max=n_init_max, nucleus_type=nucleus_type, diam=diam,
                         dn_dlogD=dn_dlogD, name=name, diam_cutoff=diam_cutoff, T_array=T_array,
                         singular_fun=singular_fun, singular_scale=singular_scale, psd=psd,
                         n_init_weight_prof=n_init_weight_prof, entrain_psd=entrain_psd,
                         entrain_to_cth=entrain_to_cth, ci_model=ci_model)


class logn_AER(AER_pop):
    """
    Log-normal aerosol PSD.
    """
    def __init__(self, use_ABIFM, n_init_max, nucleus_type=None, name=None,
                 diam_cutoff=None, T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, correct_discrete=True,
                 entrain_psd=None, entrain_to_cth=None, ci_model=None):
        """
        Parameters as in the 'AER_pop' class

        Parameters
        ----------
        diam_mean: float
            geometric mean diameter [um]
        geom_sd: float
            geometric standard deviation
        n_bins: int
            if int, specify the number of bins in psd array.
            if list or np.ndarray, specifies diameter bin edges thereby bypassing the diameter array setup
        diam_min: float or 2-element tuple
            minimum diameter [um]. If a 2-element tuple, then the 1st element is the minimum diameter
            and the 2nd is the maximum diameter cutoff (large diameters will not be considered).
        m_ratio: float
            bin-tp-bin mass ratio (smaller numbers give more finely resolved grid).
            Effectively, the diameter ratio between consecutive bins is m_ratio**(1/3).
        correct_discrete: bool
            normalize dn_dlogD so that sum(dn_dlogD) = n_init_tot (typically, a correction of
            ~0.5-1.0% due to discretization)
        """
        psd.update({"type": "logn"})  # require type key consistency
        if not np.all([x in psd.keys() for x in ["diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"]]):
            raise KeyError('log-normal PSD processing requires the fields' +
                           '"diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"')
        diam, dn_dlogD, dF, nF = self._calc_logn_diam_dn_dlogd(psd, n_init_max)
        self.raw_diam, self.raw_dn_dlogD, self.unnorm_dn_dlogD = dF, nF, dn_dlogD  # bin edge, dn_dlogD, bin unnorm
        if correct_discrete:
            dn_dlogD = self._normalize_to_n_tot(n_init_max, dn_dlogD)  # correct for discretization
        if entrain_psd is None:  # Entrained aerosol PSD
            entrain_psd = {"dn_dlogD": np.copy(dn_dlogD)}
        else:
            _, dn_dlogD_ent, _, _ = \
                self._calc_logn_diam_dn_dlogd(entrain_psd, entrain_psd["n_init_max"], diam_in=dF)
            if correct_discrete:
                dn_dlogD_ent = self._normalize_to_n_tot(
                    entrain_psd["n_init_max"], dn_dlogD_ent)  # correct for discretization
            entrain_psd["dn_dlogD"] = dn_dlogD_ent

        super().__init__(use_ABIFM=use_ABIFM, n_init_max=n_init_max, nucleus_type=nucleus_type, diam=diam,
                         dn_dlogD=dn_dlogD, name=name, diam_cutoff=diam_cutoff, T_array=T_array,
                         singular_fun=singular_fun, singular_scale=singular_scale, psd=psd,
                         n_init_weight_prof=n_init_weight_prof, entrain_psd=entrain_psd,
                         entrain_to_cth=entrain_to_cth, ci_model=ci_model)

    def _calc_logn_diam_dn_dlogd(self, psd, n_init_max, integrate_dn_dlogD=True, diam_in=None):
        """
        Assign particle diameter array and calculate dn_dlogD for log-normal distribution.
        Then integrate using trapezoidal rule to get total concentration per bin.

        Parameters
        ---------
        psd: dict
            Log-normal PSD parameters.
        n_init_max: float
            total initial aerosol concentration [m-3].
        integrate_dn_dlogD: bool
            True - integrate dn_dlogD using the trapezoidal rule, False - normalize (origianl code prototype).
        diam_in: np.ndarray or None
            Input diameter array instead of generating one using 'diam_min', 'n_bins', and 'm_ratio'.
            An alternative to diam_in (as currently implemented in the call of this method from init_model is
            to specify the input diameter array in the 'n_bins' input variable.
            Ignored if None.

        Returns
        -------
        diam_bin_mid: np.ndarray
            Particle diameter array (log-scale middle of integrated bin converted back to linear).
        dn_dlogD_bin: np.ndarray
            Particle number concentration (integrated) per diameter bin.
        diam: np.ndarray
            Particle diameter array corresponding to dn_dlogD.
        dn_dlogD: np.ndarray
            Particle number concentration per diameter (PSD value in units of m-3)
        """
        if diam_in is None:
            if isinstance(psd["n_bins"], (np.ndarray, list)):  # specified diam edge array as in custom_AER
                diam = psd["n_bins"]
            else:
                if isinstance(psd["diam_min"], float):
                    diam = np.ones(psd["n_bins"]) * psd["diam_min"]
                elif isinstance(psd["diam_min"], tuple):
                    diam = np.ones(psd["n_bins"]) * psd["diam_min"][0]
                diam = diam * (psd["m_ratio"] ** (1. / 3.)) ** (np.cumsum(np.ones(psd["n_bins"])) - 1)
                if isinstance(psd["diam_min"], tuple):  # remove diameters larger than cutoff
                    diam = diam[diam <= psd["diam_min"][1]]
        else:
            diam = diam_in
        denom = np.sqrt(2 * np.pi) * np.log(psd["geom_sd"])
        if integrate_dn_dlogD:
            diam_bin_mid = np.exp((np.log(diam[:-1]) + np.log(diam[1:])) / 2)  # bin middle in log scale
            argexp = np.log(diam_bin_mid / psd["diam_mean"]) / np.log(psd["geom_sd"])
            dn_dlogD = (n_init_max / denom) * np.exp(-0.5 * argexp**2)
            dn_dlogD_bin = np.diff(np.log(diam)) * dn_dlogD
        else:  # original code
            argexp = np.log(diam / psd["diam_mean"]) / np.log(psd["geom_sd"])
            dn_dlogD = (1 / denom) * np.exp(-0.5 * argexp**2)
            dn_dlogD = dn_dlogD / dn_dlogD.sum() * n_init_max
            dn_dlogD_bin = dn_dlogD[:]  # in this case same as normalized dn_dlogD
            diam_bin_mid = diam[:]  # in this case (no integration) represents a bin value.
        return diam_bin_mid, dn_dlogD_bin, diam, dn_dlogD

    def _normalize_to_n_tot(self, n_init_max, dn_dlogD):
        """
        normalize dn_dlogD so that sum(dn_dlogD) = n_init_tot (typically, a correction of ~0.1-0.5% due to
        discretization).

        Parameters
        ----------
        n_init_max: float
            total initial aerosol concentration [m-3].
        dn_dlogD_bin: np.ndarray
            Particle number concentration (integrated) per diameter bin.

        Returns
        -------
        dn_dlogD: np.ndarray
            Particle number concentration per diameter (PSD value in units of m-3)
        """
        dn_dlogD = dn_dlogD / np.sum(dn_dlogD) * n_init_max
        return dn_dlogD


class multi_logn_AER(logn_AER):
    """
    Multiple log-normal aerosol PSD.
    """
    def __init__(self, use_ABIFM, n_init_max, nucleus_type=None, name=None,
                 diam_cutoff=None, T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, correct_discrete=True, entrain_psd=None,
                 entrain_to_cth=None, ci_model=None):
        """
        Parameters as in the 'AER_pop' class. Note that n_init_max should be a list or np.ndarray
        of values for each mode with the same length as diam_mean and geom_sd. Array bins are specified
        using scalars.

        Parameters
        ----------
        diam_mean: list or np.ndarray of float
            geometric mean diameter [um] for each model
        geom_sd: list or np.ndarray of float
            geometric standard deviation for each mode
        n_bins: int
            number of bins in psd array
        diam_min: float or 2-element tuple
            minimum diameter [um]. If a 2-element tuple, then the 1st element is the minimum diameter
            and the 2nd is the maximum diameter cutoff (large diameters will not be considered).
        m_ratio: float
            bin-tp-bin mass ratio (smaller numbers give more finely resolved grid).
            Effectively, the diameter ratio between consecutive bins is m_ratio**(1/3).
        correct_discrete: bool
            normalize dn_dlogD so that sum(dn_dlogD) = n_init_tot (typically, a correction of
            ~0.5-1.0% due to discretization)
        """
        psd.update({"type": "multi_logn"})  # require type key consistency
        if not np.all([x in psd.keys() for x in ["diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"]]):
            raise KeyError('log-normal PSD processing requires the fields' +
                           '"diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"')
        if not len(np.unique((len(n_init_max), len(psd["diam_mean"]), len(psd["geom_sd"])))) == 1:
            raise IndexError("'n_init_max', 'diam_mean', and 'geom_sd' must have the same length (one " +
                             "value for each mode)")
        for ii in range(len(n_init_max)):
            psd_tmp = psd.copy()
            psd_tmp["diam_mean"] = psd_tmp["diam_mean"][ii]
            psd_tmp["geom_sd"] = psd_tmp["geom_sd"][ii]
            diam_tmp, dn_dlogD_tmp, dF_tmp, nF_tmp = super()._calc_logn_diam_dn_dlogd(psd_tmp, n_init_max[ii])
            if ii == 0:
                diam, dn_dlogD, dF, nF = diam_tmp, dn_dlogD_tmp, dF_tmp, nF_tmp
            else:
                dn_dlogD += dn_dlogD_tmp
                nF += nF_tmp
        self.unnorm_dn_dlogD = dn_dlogD
        self.raw_diam, self.raw_dn_dlogD, self.unnorm_dn_dlogD = dF, nF, dn_dlogD  # bin edge, dn_dlogD, bin unnorm
        if correct_discrete:
            dn_dlogD = super()._normalize_to_n_tot(np.sum(n_init_max), dn_dlogD)  # correct for discretization
        if entrain_psd is None:  # Entrained aerosol PSD
            entrain_psd = {"dn_dlogD": np.copy(dn_dlogD)}
        else:
            for ii in range(len(entrain_psd["n_init_max"])):
                psd_tmp = entrain_psd.copy()
                psd_tmp["diam_mean"] = psd_tmp["diam_mean"][ii]
                psd_tmp["geom_sd"] = psd_tmp["geom_sd"][ii]
                _, dn_dlogD_tmp, _, _ = super()._calc_logn_diam_dn_dlogd(
                    psd_tmp, entrain_psd["n_init_max"][ii], diam_in=dF)
                if ii == 0:
                    dn_dlogD_ent = dn_dlogD_tmp
                else:
                    dn_dlogD_ent += dn_dlogD_tmp
            if correct_discrete:
                dn_dlogD_ent = self._normalize_to_n_tot(
                    np.sum(entrain_psd["n_init_max"]), dn_dlogD_ent)  # correct for discretization
            entrain_psd["dn_dlogD"] = dn_dlogD_ent

        super(logn_AER, self).__init__(use_ABIFM=use_ABIFM, n_init_max=np.sum(n_init_max),
                                       nucleus_type=nucleus_type, diam=diam, dn_dlogD=dn_dlogD, name=name,
                                       diam_cutoff=diam_cutoff, T_array=T_array, singular_fun=singular_fun,
                                       singular_scale=singular_scale, psd=psd,
                                       n_init_weight_prof=n_init_weight_prof, entrain_psd=entrain_psd,
                                       entrain_to_cth=entrain_to_cth, ci_model=ci_model)


class custom_AER(AER_pop):
    """
    custom aerosol PSD ('dn_dlogD' and 'diam' with optional normalization to n_init_max).
    """
    def __init__(self, use_ABIFM, n_init_max=None, nucleus_type=None, name=None,
                 diam_cutoff=None, T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, entrain_psd=None, entrain_to_cth=None,
                 ci_model=None):
        """
        Parameters as in the 'AER_pop' class

        Parameters
        ----------
        norm_to_n_init_max: bool
            If True then dn_dlogD is normalized such that sum(dn_dlogD) = n_init_max. In that case, n_init_max
            must be specified.
        """
        psd.update({"type": "custom"})  # require type key consistency
        if not np.all([x in psd.keys() for x in ["diam", "dn_dlogD"]]):
            raise KeyError('custom PSD processing requires the fields "diam", "dn_dlogD"')
        diam = psd["diam"]
        dn_dlogD = psd["dn_dlogD"]
        if len(dn_dlogD) != len(diam):
            raise ValueError("The 'diam' and 'dn_dlogD' arrays must have the same size!")
        if "norm_to_n_init_max" in psd.keys():
            if psd["norm_to_n_init_max"]:
                dn_dlogD = dn_dlogD / np.sum(dn_dlogD) * n_init_max
        if entrain_psd is None:  # Entrained aerosol PSD
            entrain_psd = {"dn_dlogD": np.copy(dn_dlogD)}
        else:
            if np.logical_or(len(entrain_psd["diam"]) != len(diam),
                             entrain_psd["dn_dlogD"] != len(entrain_psd["diam"])):
                raise ValueError("The 'diam', entrain 'diam', and entrain 'dn_dlogD' arrays are not same size!")
            dn_dlogD_ent = entrain_psd["dn_dlogD"]
            if "norm_to_n_init_max" in entrain_psd.keys():
                if entrain_psd["norm_to_n_init_max"]:
                    dn_dlogD_ent = dn_dlogD_ent / np.sum(entrain_psd["dn_dlogD"]) * entrain_psd["n_init_max"]

        super().__init__(use_ABIFM=use_ABIFM, n_init_max=n_init_max, nucleus_type=nucleus_type, diam=diam,
                         dn_dlogD=dn_dlogD, name=name, diam_cutoff=diam_cutoff, T_array=T_array,
                         singular_fun=singular_fun, singular_scale=singular_scale, psd=psd,
                         n_init_weight_prof=n_init_weight_prof, entrain_psd=entrain_psd,
                         entrain_to_cth=entrain_to_cth, ci_model=ci_model)
