# COMBLE Model-Observation Intercomparison Project Cookbook

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=ARM+JupyterHub&message=ACE+Environment&color=blue)](https://jupyterhub.arm.gov/hub/user-redirect/git-pull?repo=https%3A//github.com/ARM-Development/comble-mip&urlpath=lab/tree/comble-mip/../user-data-home/comble-mip/notebooks&branch=main)

```{figure} figures/13march_case_overview.png

(Left) MODIS visible satellite image over the Norwegian Sea region on 13 March 2020. Colored lines show backward trajectories from 18 UTC at Andenes, Norway (denoted by the blue star) at altitudes of 500, 1000, and 2000 m ASL in cyan, yellow, and blue, respectively. (Right) Vertically pointing radar, lidar, microwave radiometer, and meteorological measurements at Andenes highlight the convective nature of cellular clouds, characterized by high reflectivity, strong vertical motions, liquid water pockets, and intense turbulence structures. 
```

## Background & Motivation

This website hosts information about the Cold-Air Outbreaks in the Marine Boundary Layer Experiment (COMBLE) Model-Observation Intercomparison Project (MIP) for large-eddy simulation (LES) and single-column models (SCMs). The intercomparison, which was initiated in late 2021, focuses on a cold-air outbreak event observed over the Norwegian Sea on 13 March 2020 (Fig. 1) during the US Department of Energy's COMBLE field campaign [(Geerts et al., 2023)](https://journals.ametsoc.org/view/journals/bams/103/5/BAMS-D-21-0044.1.xml). Despite the frequent occurrence of Arctic CAOs, numerical weather prediction models often have difficulty representing the convective boundary layer structure and turbulence, as well as mixed-phase cloud properties. Moreover, earth system models face similar challenges with respect to predicting mixed-phase cloud feedbacks and equilibrium climate sensitivity.

In mid-2022, a [COMBLE MIP white paper](https://www.gewexevents.org/wp-content/uploads/COMBLE_white_paper.pdf) was accepted by the GEWEX Atmospheric System Study (GASS) panel, describing a two-part project for LES and SCM models. In part one, droplet and ice concentrations are specified. In part two, the specification adds an observation-informed initial aerosol size distribution profile with modally specified composition and hygroscopicity, adequate for prognosing droplet activation and heterogeneous ice formation. The COMBLE MIP effort has been an early adopter of the [DOE ARM Data Workbench](https://armcrf.servicenowservices.com/kb?id=kb_search&kb_knowledge_base=d932abb71ba9d590094aebdbac4bcb8b) to support each stage of participation, in addition to providing access to the [DOE Cumulus HPC](https://www.arm.gov/capabilities/computing-resources) for computationally intensive forward-simulation for model evaluation against ground-based lidar and radar measurements at Andenes.

We welcome you to explore the website to learn more about the COMBLE-MIP project, including goals & hypotheses, modeling framework, and workflows.

## Model Inputs

A [file](https://github.com/ARM-Development/comble-mip/blob/main/notebooks/forcing/COMBLE_INTERCOMPARISON_FORCING_V2.4.nc) containing model intialization and forcing information is provided in the [DEPHY-SCM standard format](https://github.com/GdR-DEPHY/DEPHY-SCM), suitable for application to both LES and SCM setups. The most recent version of the DEPHY standards may be found [here](https://docs.google.com/document/d/1eAWY-ELL5Ua6a9WIsv4ODHmLXvfgla5TNQAuAwNASo0).

Detailed information about our requested model configuration may be found [here](https://arm-development.github.io/comble-mip/main_configuration.html).

## Python Notebooks

We've developed several Jupyter Notebooks to reduce the burden on COMBLE MIP participants:

### Input Conversion Examples

Example scripts to convert the DEPHY standard format to ASCII or other netCDF formats needed to drive specific models.

* [DEPHY forcing &rarr; DHARMA LES and ModelE3 SCM](https://arm-development.github.io/comble-mip/notebooks/conversion/convert_comble_dephy_forcing_to_DHARMA_LES_and_ModelE3_SCM_forcing.html)
* [DEPHY forcing &rarr; WRF-LES](https://arm-development.github.io/comble-mip/notebooks/conversion/convert_comble_dephy_forcing_to_WRF_LES_forcing.html)

### Output Conversion Examples

Example scripts to convert specific model results back to the DEPHY standard format prior to submission.

* [DHARMA LES output &rarr; DEPHY](https://arm-development.github.io/comble-mip/notebooks/conversion_output/convert_DHARMA_LES_output_to_dephy_format.html)
* [ModelE3 SCM output &rarr; DEPHY](https://arm-development.github.io/comble-mip/notebooks/conversion_output/convert_ModelE3_SCM_output_to_dephy_format.html)
* [WRF-LES output &rarr; DEPHY](https://arm-development.github.io/comble-mip/notebooks/conversion_output/convert_WRF_LES_output_to_dephy_format.html)

### Visualization Tools

Example scripts to compare your results with those submitted from other models as well as with observations.

* [Check your LES results](https://arm-development.github.io/comble-mip/notebooks/plotting/check_les.html)
* [Check your SCM results](https://arm-development.github.io/comble-mip/notebooks/plotting/check_scm.html)
* [Compare with observations](https://arm-development.github.io/comble-mip/notebooks/plotting/example_plotting.html)

```{note}
If you would like to contribute to the COMBLE MIP website by adding example scripts to convert to/from your host model and/or enhancing our plotting scripts, then please see the [Contributors Guide](https://arm-development.github.io/comble-mip/CONTRIBUTING.html).
```

## Authors

Tim Juliano, Florian Tornow, Ann Fridlind - Intercomparison development

Abigail Williams, Lynn Russell, Yijia Sun, Daniel Knopf - Aerosol analysis

Max Grover, Scott Collis, Kyle Dumas, Monica Ihli - Infrastructure development

Please contact Tim Juliano (tjuliano@ucar.edu) and Florian Tornow (ft2544@columbia.edu) for comments or questions about the model intercomparison project.

```{attention}
If you are interested in participating in the LES/SCM COMBLE intercomparison, then please [sign up here](https://docs.google.com/spreadsheets/d/1h0BDDCCJTfIsdvHHNFyA17bpsNAL7405GG69IkC8qJs/edit?usp=sharing).
```
