"""
Configuration file for barotropic vorticity equation simulation
"""

import xarray as xr
import numpy as np


# ===== EARTH PARAMETERS =====
R = 6371 * 10**3                      # Earth radius (in m)
Omega = 2.0 * np.pi / (24*3600)       # Earth rangular velocity (in rad/s)
g0 = 9.806                            # Mean gravitational acceleration (in m/s^2)
H = 8.5 * 10**3                       # Atmospheric scale height (in m)


# ===== TIME INTEGRATION =====
ti = 0                  # Initial time (in s)
tf = 12 * 3600		# Final time (in s)
dt = 150                # Time step (in s)

# RAW filter
nu = 0.1           # Damping factor   
alpha = 0.5        # Displacement factor


# ===== OUTPUT =====
output_name = "exp_1"       # Name of the output folder
save_time = 3 * 3600        # Time interval between saves (in s) 


# ===== INITIAL CONDITIONS =====

# Initial horizontal velocity fields
dataset_name = "uv_19-02-2026_0000_glob.nc"     # Name of the dataset file

# We open and read the horizontal velocity dataset
ds = xr.open_dataset(dataset_name, engine='netcdf4')
time_value = ds['valid_time'].values[0]
press_lvl = ds['pressure_level'].values[0]

# We get the horizontal velocity components
u0 = ds['u'].sel(valid_time=time_value, pressure_level=press_lvl).values
v0 = ds['v'].sel(valid_time=time_value, pressure_level=press_lvl).values

# We get the latitude and longitude coordinates (nlat = N+1, nlon = 2N)
lat = ds['latitude'].values
lon = ds['longitude'].values