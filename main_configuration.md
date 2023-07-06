# Main Model Configuration

| Model component                   | Setting                                                                                                                                   |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Horizontal grid cell spacing      | Dx=Dy=100 m                                                                                                                               |
| Horizontal domain dimensions/size | Nx=Ny=256; Lx=Ly=25.6 km                                                                                                                  |
| Vertical grid                     | According to input forcing file specifications                                                                                            |
| Domain top                        | 7 km                                                                                                                                      |
| Start/end times                   | 22 UTC on 12 March 2020<br>18 UTC on 13 March 2020                                                                                        |
| Initial profiles                  | Thermodynamic and kinematic soundings provided, with the following variable names to be used as initial conditions: u, v, temp, theta, qv |
| Initial perturbations for LES     | Theta only: 0.1 K below 250 m                                                                                                             |
| Surface forcing                   | 2–20h: specified SST in time with roughness height following Charnock                                                                     |
