# Global non-divergent Barotropic Vorticity Equation (BVE) solver

> [!WARNING]
> **ARCHIVED REPOSITORY**
> 
> This repository has ben **archived** and `BVE-global-model` is now part of the [BVE-models](https://github.com/Arnau-V-B/BVE-models.git) repository, together with two more BVE solvers:
> - `BVE-global-linear-model`: which also solves the non-divergent BVE globally but does it linearizing the equation over a mean zonal flow (for idealized experiments)
> - `BVE-beta-channel-model`: which solves the non-divergent BVE in a beta-channel ($$\beta=ctt.$$) linearizing it over a mean zonal flow (for analytic Rossby wave dispersion analyses)
>
> Please visit the new repository for all future updates.

This program solves the non-divergent BVE on the sphere using spherical harmonics transforms. It is designed for global atmospheric dynamics simulations, such as 500 hPa flow evolution, with realistic initial conditions of single-level relative vorticity fields from reanalysis data (e.g., ERA5). The workflow of the code is based on [technical documentation](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/barotropic.pdf) from the NOAA Geophysical Fluid Dynamics Laboratory (GFDL).

The model uses a spectral transform method with triangular truncation, supports both equally sampled (Driscoll–Healy) and Gauss–Legendre quadrature grids (see SHTOOLS [documentation](https://shtools.github.io/SHTOOLS/)), and includes implicit hyperdiffusion for numerical stability. The time integration follows a leapfrog scheme with a Robert–Asselin–Williams (RAW) filter to control its associated computational mode and improve the accuracy to third order (i.e. $$O(\Delta t^3)$$), and the time step is set to 30 min (but it can be lowered if numerical instability appears).

## Mathematical formalism

For a barotropic atmosphere with constant height, the evolution of relative vorticity can be described pretty well with the non-divergent barotropic vorticity equation:

$$\frac{\partial \zeta}{\partial t} = - \vec{v}\cdot\vec{\nabla}(\zeta + f) \quad \longleftrightarrow \quad \frac{\partial}{\partial t}(\nabla^2\psi) = - J(\psi, \nabla^2\psi + f);$$

where $$\vec{v}$$ is the horizontal wind velocity, $$\zeta$$ is the relative vorticity (defined as $$\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$), $\psi$ is the streamfuncion (defined as $$\vec{v} = \hat{k} \times \vec{\nabla}\psi$$ or $$\zeta = \nabla^2\psi$$), $$f = 2\Omega\sin(\theta)$$ is the Coriolis parameter and $$J(A,B) = \frac{\partial A}{\partial x} \frac{\partial B}{\partial y} - \frac{\partial A}{\partial y}\frac{\partial B}{\partial x}$$ is the Jacobi operator.

On the spherical Earth surface, fields are periodic in longitude $\lambda$ (i.e. azimuthal angle $\phi$) and enclosed in latitude $\theta$ (i.e. copolar angle $-\Theta$), so they can be represented as series of spherical harmonics up to a given order $l_{max}$ (the truncation wavenumber):

$$\psi(\lambda,\theta) = \sum_{l=0}^{l_{max}}\sum_{m=-l}^{l} \psi_{lm} P_{lm}(sin\theta) e^{im\lambda};$$

where $$P_{lm}$$ is the associated Legendre polynomial of order $$m$$ and degree $$l$$. In this "spectral" space, the spatial derivatives ($$\frac{\partial}{\partial x} \equiv \frac{1}{R\cos(\theta)}\frac{\partial}{\partial \lambda}$$ and $$\frac{\partial}{\partial y} \equiv \frac{1}{R}\frac{\partial}{\partial \theta}$$) become analytic and can be solved through recursive relations such as:

$$\frac{\partial \hat{\psi}_{lm}}{\partial \lambda} = i m \hat{\psi}_{lm}$$

$$\cos(\theta) \frac{\partial \hat{\psi}_{lm}}{\partial \theta} = (l+2) \epsilon_{l+1,m} \hat{\psi}_{l+1,m} - (l-1) \epsilon_{l,m} \hat{\psi}_{l-1,m} \quad \mathrm{; where} \quad \epsilon_{lm} = \sqrt{\frac{l^2 - m^2}{4l^2 - 1}}$$

To prevent the simulation from being contaminated with aliasing of the highest modes (i.e. the smallest waves) resulting from the products between fields, an extra term has to be added to the BVE:

$$\frac{\partial \zeta}{\partial t} = - \vec{v}\cdot\vec{\nabla}(\zeta + f) - \eta \nabla^4 \zeta \quad \mathrm{; with} \quad \eta = \frac{R^4}{\tau (l_{max}(l_{max} + 1))^2};$$

where $$R$$ is the Earth radius, $$\tau$$ the decay rate of the target waves and $$l_{max}$$ the highest wavenumber. This is known as the hyperdiffusion term and only dissipates the response at the smallest unresolved scales, so that they disappear soon after they are formed and also transfer of eddy energy from large to small scales is partially simulated. By default, the decay rate is set to 3 hours, but it can be increased to for better resolution or decreased for even more smoothing.

Finally, to integrate in time, a first order Euler scheme is used at the first iteration:

$$\zeta_{i+1} = \zeta_i + \Delta t RHS_i,$$

and then a third order scheme based on a leapfrog with RAW filter is used for the following iterations:

$$\zeta_{i+1} = \zeta_{i-1} + 2\Delta t RHS_i$$

$$\zeta_i = \zeta_i + \frac{\nu \alpha}{2}(\zeta_{i+1} - 2\zeta_i + \zeta_{i-1})$$

$$\zeta_{i+1} = \zeta_{i+1} - \frac{\nu (1-\alpha)}{2}(\zeta_{i+1} - 2\zeta_i + \zeta_{i-1});$$

where $$\nu=0.1$$ and $$\alpha=0.5$$ seem to conserve kinetic energy, enstrophy and mean vorticity the most.

## Program structure

The execution of the program relies on two files that need to be in the same directory:
1. `config.py`. Here all the initial conditions and parameters of the simulation are specified. By default, it tries to read an ERA5 reanalysis relative vorticity field in netCDF format, so such file or a similar one has to be provided by the user. Nevertheless, the script can be easily modified to accomodate it to the user needs.
2. `main_BVE_glob.py`. This is the main program. It contains all the internal functions defined, the execution pipeline and the saving and plotting routines of the output results. The final plotting routine can be disabled to save time if more customized plots are desired.

Once the initial conditions are properly set in `config.py`, to run the simulation script `main_BVE_glob.py` has to be executed. For example, on Windows 11 this can be done by typing in the following line in the CMD terminal:
```
python main_BVE_glob.py
```

During the execution an `output/` folder is created in the same directory where all the simulation output is saved. In it a file named `params_exp_().txt` contains the `config.py` copy with the initial conditions used in the simulation, and two more folders are created `data/` and `figures/`. The former contains two netCDF files with the register of conserved values (i.e. knietic energy, enstrophy and mean vorticity) in `conserved_values_exp_().nc` and the streamfunction and relative vorticity fields evolution in `fields_evolution_exp_().nc`. In the later, a figure showing the evolution of the three conserved values (`conserved_values_exp_().png`), two GIFs with the evolution of the fields (`.gif`) and the single frames used for both of them, are saved. 

## Requirements

All the code has been written in Python v3.13.7 on a Windows 11 laptop with both clarity and efficienty in mind, so there should be no problem running it in any computer hardware as long as it has Python installed. For the program to run, the following Python libraries need to be installed:
- [numpy (v2.3.3)](https://pypi.org/project/numpy/)
- [scipy (v1.16.2)](https://pypi.org/project/scipy/)
- [xarray (v2025.10.1)](https://docs.xarray.dev/en/stable/)
- [pyshtools (v4.14.1)](https://pypi.org/project/pyshtools/)
- [matplotlib (v3.10.6)](https://pypi.org/project/matplotlib/)
- [imageio (v2.37.2)](https://pypi.org/project/ImageIO/)

> [!NOTE]
> The versions shown are only orientative, as newer or older ones will most likely work just fine.

> [!WARNING]
> This program has only been tested with Windows 11, so Windows 10, macOS and Linux users could possibly experience some problems running it.

## License

This program is completely open source and was made in colaboration with Universitat de Barcelona as part of a master's degree final thesis.
