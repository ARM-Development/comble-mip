# COMBLE Model-Observation Intercomparison Project Cookbook

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=ARM+JupyterHub&message=ACE+Environment&color=blue)](https://jupyterhub.arm.gov/hub/user-redirect/git-pull?repo=https%3A//github.com/ARM-Development/comble-mip&urlpath=lab/tree/comble-mip/../user-data-home/comble-mip/notebooks&branch=main)

```{figure} figures/13march_case_overview.png

(Left) MODIS visible satellite image over the Norwegian Sea region on 13 March 2020. Colored lines show backward trajectories from 18 UTC at Andenes, Norway (denoted by the blue star) at altitudes of 500, 1000, and 2000 m ASL in cyan, yellow, and blue, respectively. (Right) Vertically pointing radar, lidar, microwave radiometer, and meteorological measurements at Andenes highlight the convective nature of cellular clouds, characterized by high reflectivity, strong vertical motions, liquid water pockets, and intense turbulence structures. 
```

## Motivation

This website hosts information about the Cold-Air Outbreaks in the Marine Boundary Layer Experiment (COMBLE) Model-Observation Intercomparison Project (MIP) for large-eddy simulation (LES) and single-column models (SCMs). The intercomparison focuses on a cold-air outbreak event observed over the Norwegian Sea on 13 March 2020 (Fig. 1) during the US Department of Energy's COMBLE field campaign [(Geerts et al., 2023)](https://journals.ametsoc.org/view/journals/bams/103/5/BAMS-D-21-0044.1.xml). We welcome you to explore the website to learn more about the COMBLE-MIP framework and workflow.

## Model Inputs

A file containing model intialization and forcing information is provided in the [DEPHY-SCM standard format](https://github.com/GdR-DEPHY/DEPHY-SCM) (see document link on README.md), suitable for application to both LES and SCM setups.

## Input Conversion Examples

Example scripts to convert the DEPHY standard format to ASCII or other netCDF formats needed to drive specific models.

## Output Conversion Examples

Example scripts to convert specific model results back to the DEPHY standard format prior to submission.

## Visualization Tools

Plotting scripts to compare your results with those submitted from other models and observations.

## Authors

Tim Juliano, Florian Tornow, Ann Fridlind - Intercomparison development

Abigail Williams, Lynn Russell, Yijia Sun, Daniel Knopf - Aerosol analysis

Max Grover, Scott Collis, Monica Ihli - Infrastructure development

Please contact Tim Juliano: tjuliano ((at)) ucar.edu OR Florian Tornow: ft2544 ((at)) columbia.edu for comments or questions about the model intercomparison project.

```{attention}
If you are interested in participating in the LES/SCM COMBLE intercomparison, please [sign up here](https://docs.google.com/spreadsheets/d/1h0BDDCCJTfIsdvHHNFyA17bpsNAL7405GG69IkC8qJs/edit?usp=sharing).
```
