"""
Configuration file for the global non-divergent barotropic vorticity equation (BVE) simulation.
All the initial conditions of the simulation are defined in this file. The code is prepared to read the
netCDF file of an ERA5 reanalysis relative vorticity field, but it can easily be adapted for any other
dataset options.
This script has to be placed in the same direcotry as the main program main_BVE_glob.py
"""


import xarray as xr
import numpy as np


# EARTH PARAMETERS ==========================================================================================
R = 6371 * 10**3                      # Earth radius (in m)
Omega = 2.0 * np.pi / (24*3600)       # Earth rangular velocity (in rad/s)
g0 = 9.806                            # Mean gravitational acceleration (in m/s^2)


# TIME INTEGRATION ==========================================================================================
ti = 0                                # Initial time (in s)
tf = 48 * 3600		                  # Final time (in s)
dt = 1800                             # Time step (in s)
tau_d = 3 * 3600                      # Hyperdiffusion time scale (in s)

# RAW filter
nu = 0.1                              # Damping factor   
alpha = 0.5                           # Displacement factor


# OUTPUT ====================================================================================================
output_name = "exp_test"              # Name of the output folder
save_time = 3 * 3600                  # Time interval between saves (in s) 


# INITIAL CONDITIONS ========================================================================================

# Initial relative vorticity field
dataset_name = "forecast_15-04-2026_glob.nc"     # Name of the dataset file

# We open and read the relative vorticity dataset
ds = xr.open_dataset(dataset_name, engine='netcdf4')
start_date = ds['valid_time'].values[0]
press_lvl = ds['pressure_level'].values[0]

# We get the initial relative vortiticy field
zeta0 = ds['vo'].sel(valid_time=start_date, pressure_level=press_lvl).values

# We get the latitude and longitude coordinates (in ERA5: nlat = N+1, nlon = 2N)
lat = ds['latitude'].values
lon = ds['longitude'].values

ds.close()


# SPECTRAL SCHEME ===========================================================================================
gridtype = 'GLQ'     # Type of grid quadrature: Discroll-Healy (DH) --> Equally sampled Nx2N
                     #                          Gauss-Legendre (GLQ) --> Gaussian latitudes (N+1)x(2N+1)

# We define the grid sampling and triangular truncation
if gridtype == 'DH':
    sampl = (len(lat) - 1) // 2 - 1         # Number of samples for grid representation
    lmax = sampl                            # Maximum wavenumber for spectral triangular truncation
                                            # lmax <= sampl
elif gridtype == 'GLQ':
    sampl = 128
    lmax = 85