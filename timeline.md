# Timeline & File Naming Convention

```{attention}
Ready to make your model outputs accessible to other MIP participants? Please refer to [this page](https://arm-development.github.io/comble-mip/CONTRIBUTING.html) to learn how to upload your model outputs to the repository.
```

## Phase 1
| Product                                                                                                                                                      | Simulation Name                                                                        | Due Date                                                                                    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| SCM/small-domain LES, liquid-only<br>SCM/small-domain LES with ice<br>SCM/small-domain LES with ice, default roughness lengths*<br>Large domain LES with ice | Lx25_dx100_FixN_noice<br>Lx25_dx100_FixN<br>Lx25_dx100_FixN_def_z0<br>Lx125_dx100_FixN | Dec. 15, 2023<br>Dec. 15, 2023<br>Dec. 15, 2023<br><span style="color:red">**TBD**</span>** |

## Phase II
| Product                                                                                                                                                      | Simulation Name                                                  | Due Date                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------|
| SCM/small-domain LES, prognostic aerosol, diagnostic ice<br>SCM/small-domain LES, prognostic aerosol and ice<br>Large domain LES, prognostic aerosol and ice | Lx25_dx100_ProgNa<br>Lx25_dx100_ProgNaNi<br>Lx125_dx100_ProgNaNi | Jan. 15, 2024<br>Jan. 15, 2024<br><span style="color:red">**TBD**</span>** |

## Phase II (optional)
| Product                                                                                                                      | Simulation Name                                                                                | Due Date                                                     |
|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| Prognostic droplet, no ice<br>Prognostic droplet, diagnostic ice<br>Prognostic droplet and ice<br>Prognostic aerosol, no ice | Lx25_dx100_ProgNc_noice<br>Lx25_dx100_ProgNc<br>Lx25_dx100_ProgNcNi<br>Lx25_dx100_ProgNa_noice | Apr. 1, 2024<br>Apr. 1, 2024<br>Apr. 1, 2024<br>Apr. 1, 2024 |

## Notes
*This simulation will use the default roughness length formulations in each host model. All other simulations will use the fixed roughness lengths as defined on the [Main Model Configuration page](https://arm-development.github.io/comble-mip/main_configuration.html).
<br>
**Given its high computational cost, we suggest that participants refrain from running this setup in case there are any last minute tweaks to the model forcing; for instance, large-scale forcing required to generate realistic roll structures.
