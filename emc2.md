# The Earth Model Column Collaboratory (EMC<sup>2</sup>)

To faciliate direct observational-model evaluations, we will utilize the EMC<sup>2</sup>, which is an open-source ground- and satellite-based lidar and radar instrument simulator and subcolumn generator. While it specifically targets large-scale model outputs (e.g., climate models), EMC<sup>2</sup> is also applicable to high-resolution model output (e.g., LES). More information about EMC<sup>2</sup> may be found in [Silber et al., 2022](https://doi.org/10.5194/gmd-15-901-2022).

An example application of EMC<sup>2</sup> using LES outputs from the COMBLE-MIP case to simulate the Ka-band ARM Zenith Radar (KAZR) and micropulse lidar (MPL), which were deployed at the AMF1 site at Andenes, Norway, may be found in Fig. 1 below.

```{figure} figures/emc2_1.png

Direct comparison of the (left) observed KAZR reflectivity, doppler velocity, and spectral width, as well as attenuated backscatter from the MPL and liquid water path with (right) simulated quantities applying EMC{sup}`2` to the NASA DHARMA LES model.
```

EMC<sup>2</sup> can also be applied to the COMBLE-MIP LES outputs to simulate satellite-based CALIOP attenuated backscatter (Fig. 2).

```{figure} figures/emc2_2.png

Direct comparison of the (top) observed attenuated backscatter from CALIOP with (bottom) simulated attenuated backscatter applying EMC{sup}`2` to the NASA DHARMA LES model.
```
