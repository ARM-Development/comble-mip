Aerosol-Cloud Column Model (AC-1D)
==================================

The goal of this simplified 1D model is to facilitate the comparison and evaluation of different parameterizations and approaches for ice nucleation via immersion freezing from the literature against each other and observational constraints. The model is informed from LES case study output. The default Arctic case study from the SHEBA field campaign is described and simulated by Fridlind et al. (2012).

Assumptions
^^^^^^^^^^^

1. Cloud conditions are independent of ice nucleation (no interactions). For the default case study, the cloud conditions are assumed at steady-state (unaffected by weak ice formation).
2. All aerosol are assumed to be activated in updrafts and restored in downdrafts, and hence, all in-cloud aerosol (including INP) are within a droplet suitable for immersion freezing (no interstatial state).

Step-by-step (model operation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Profiles of temperature, RH, and cloud water mixing ratio are read in from LES output (thermodynamic profiles can also be modified after model object initialization and prior to model run).
2. Model predicts evolution of Naer(z,t,D), Ninp(z,t,D,T) (INAS) or Ninp(z,t,T) (INN), and Nice(z,t) (optionally prognostic with the same dims as Ninp).
3. Precipitation rates are estimated from pre-specified number-weighted value. For example, numbers based on LES output at cloud base.
4. Predicted nucleation rate profiles are saved and can be plotted.  

Requirements
^^^^^^^^^^^^

* Numpy (https://numpy.org)
* Matplotlib (https://matplotlib.org)
* Pandas (https://pandas.pydata.org)
* Xarray (http://xarray.pydata.org)
* pint (https://pint.readthedocs.io/en/stable/)

Documentation
-----------------

For API documentation and an example Jupyter Notebook see: https://isilber.github.io/cld_INP_1D_model/.


Authors
-------

Code was written by `Israel Silber <ixs34@psu.edu>`_ (Pennsylvania State University) and Ann Fridlind (NASA GISS). 

References
----------
Fridlind, A.M., B. van Diedenhoven, A.S. Ackerman, A. Avramov, A. Mrowiec, H. Morrison, P. Zuidema, and M.D. Shupe, 2012: A FIRE-ACE/SHEBA case study of mixed-phase Arctic boundary-layer clouds: Entrainment rate limitations on rapid primary ice nucleation processes. J. Atmos. Sci., 69, 365-389, doi:10.1175/JAS-D-11-052.1.

Knopf, D.A., P.A. Alpert, 2013: A water activity based model of heterogeneous ice nucleation kinetics for freezing of water and aqueous solution droplets, Faraday Discuss., 165, 513-534, doi:10.1039/C3FD00035D.
