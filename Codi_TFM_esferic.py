import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pyshtools as pysh
import time
import sys
import os
import matplotlib.pyplot as plt


def grid2spec(field, grid):
	"""
	This function takes a 2D gridded field and transforms it into a set of real spherical harmonics
	coefficients until a maximum order of triangular truncation lmax.

	It accepts both Discroll-Healy (DH) (equally sampled) and Gauss-Legendre quadrature (GLQ)
	(gaussian latitudes) grids.

	PARAMETERS:
	(input) -->
		field : 2D array (nlat, nlon) of grid field
		grid : type of grid quadrature: DH or GLQ
	(internal) -->
		lmax : maximum wavenumber for spectral triangular truncation
		glq_weights : weights for each of the roots of Legendre polynomial of roder lmax
		glq_nodes : roots of Legendre polynomial of order lmax
	(output) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients
					[0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	"""

	if grid == 'DH':
		field_spec = pysh.expand.SHExpandDH(field, sampling=2, lmax_calc=lmax)

	elif grid == 'GLQ':
		field_spec = pysh.expand.SHExpandGLQ(field, w=glq_weights, zero=glq_nodes, lmax_calc=lmax)

	return field_spec


def spec2grid(field_spec, grid):
	"""
	This function takes a 3D array of real spherical harmonics with triangular truncation lmax
	and shape (2,lmax+1,lmax+1) and transforms it into a 2D gridded field with sampling sampl.

	It accepts both Discroll-Healy (DH) (equally sampled) and Gauss-Legendre quadrature (GLQ)
	(gaussian latitudes) grids.

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients
					[0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
		grid : type of grid quadrature: DH or GLQ
	(internal) -->
		sampl : sampling of the resulting gridded field
		glq_nodes : roots of Legendre polynomial of order lmax
	(output) -->
		field : 2D array (nlat, nlon) of the gridded field
	"""

	if grid == 'DH':
		field = pysh.expand.MakeGridDH(field_spec, sampling=2, lmax=sampl)

	elif grid == 'GLQ':
		field = pysh.expand.MakeGridGLQ(field_spec, zero=glq_nodes, lmax=sampl)

	return field


def lambda_derivative(field_spec):
	"""
	This function computes the derivative in longitude for a given field in the real spherical
	harmonics spectral space. The spectral field coefficients of triangular truncation lmax need to be
	shaped as (2,lmax+1,lmax+1).

	The following recursion relation between spectral coefficients is used to compute the derivative:
	-->	d/dlambda f_lm = i * m * f_lm
		Cosine coeffs (C): d/dlambda C_lm = m * S_lm
		Sine coeffs (S): d/dlambda S_lm = - m * C_lm

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients
					[0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	(internal) -->
		R : Earth radius (for correct scaling)
	(output) -->
		dlambda : 3D array (2, lmax+1, lmax+1) with the coefficients of the derivative in longitude
	"""

	# We obtain the spectral truncation of the coefficients
	lmax = field_spec.shape[1] - 1

	# We generate a vector of m's from 0 to lmax
	m = np.arange(lmax + 1)
	
	# We compute the derivative with the recursion relation (applied to all orders l and m)
	dlambda_C = m[None, :] * field_spec[1]
	dlambda_S = - m[None, :] * field_spec[0]

	# Finally we combine both cosine and sine coefficients into a (2,lmax+1,lmax+1) array
	# And divide by the Earth radius to get the correct units
	return np.stack((dlambda_C, dlambda_S), axis=0) / R


def theta_derivative(field_spec):
	"""
	This function computes the derivative in latitude for a given field in the real spherical
	harmonics spectral space multiplied by the cosine of latitude. The spectral field coefficients of
	triangular truncation lmax need to be shaped as (2,lmax+1,lmax+1).

	The following recursion relation between spectral coefficients is used to compute the derivative:
	-->	cos(theta) * d/dtheta f_lm = ((l+2) * eps_l+1,m * f_l+1,m - (l-1) * eps_lm * f_l-1,m)
		eps_lm = sqrt((l^2 - m^2) / (4*l^2 - 1))

	PARAMETERS:
	(input) -->
		field_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients
					[0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	(internal) -->
		R : Earth radius (for correct scaling)
	(output) -->
		dtheta : 3D array (2, lmax+1, lmax+1) with the coefficients of the derivative in latitude * cos(latitude)
	"""

	# We obtain the spectral truncation of the coefficients
	lmax = field_spec.shape[1] - 1

	# We generate an empty array to store the derivative coefficients
	dtheta = np.zeros_like(field_spec)

	# We generate a vector of m's from 0 to lmax
	m = np.arange(lmax + 1)

	# We iterate over all wavenumbers l from 0 to lmax
	for l in range(lmax + 1):
		# We get the m orders from 0 to l+1
		m_valid = m[:l+1]

		# We precompute the epsilon values
		eps_l = np.sqrt((l**2 - m_valid**2) / (4*l**2 - 1))
		eps_lp1 = np.sqrt(((l+1)**2 - m_valid**2) / (4*(l+1)**2 - 1))

		# We apply the recursion relation
		if l == 0:
			dtheta[:, l, :l+1] = (l+2) * eps_lp1 * field_spec[:, l+1, :l+1]
		elif l == lmax:
			dtheta[:, l, :l+1] = - (l-1) * eps_l * field_spec[:, l-1, :l+1]
		else:
			dtheta[:, l, :l+1] = ((l+2) * eps_lp1 * field_spec[:, l+1, :l+1]
								- (l-1) * eps_l * field_spec[:, l-1, :l+1])

	# Finally we divide by the Earth radius to get the correct units
	return dtheta / R


def compute_adv(u, v, vort):
	"""
	This function computes the advection term of the BVE in the spectral space multiplied by 
	cos(latitude) given the horizontal velocity fields and the vorticity field in the grid space.
	---> cos(theta) * adv = -1/R * d/dlambda (u * (zeta+f)) - cos(theta)/R * d/dtheta (v * (zeta+f))
		Where: V * Nabla(zeta+f) = Nabla * (V * (zeta+f)); as flow is non divergent (Nabla*V = 0)

	To avoid convolutions, the product is computed in the grid and then transformed to spectral
	space to perform the horizontal divergence of the product.
	
	PARAMETERS:
	(input) -->
		u : 2D array (nlat, nlon) of grid zonal velocity field
		v : 2D array (nlat, nlon) of grid meridional velocity field
		vort : 2D array (nlat, nlon) of grid vorticity field
	(internal) -->
		f : 2D array (nlat, nlon) of grid Coriolis parameter (f=2*Omega*sin(latitude))
	(output) -->
		adv_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the advection term
	"""

	# We first compute the product V(zeta+f) in the spectral space
	u_zeta = u * (vort + f)
	v_zeta = v * (vort + f)

	# Then we transform the product into the spectral space
	u_zeta_spec = grid2spec(u_zeta, gridtype)
	v_zeta_spec = grid2spec(v_zeta, gridtype)

	# We compute the horizontal divergence
	u_zeta_lambda_spec = lambda_derivative(u_zeta_spec)
	v_zeta_theta_spec = theta_derivative(v_zeta_spec)

	# And we obtain the advection term multiplied by cos(latitude)
	adv_spec = - (u_zeta_lambda_spec + v_zeta_theta_spec)

	return adv_spec


def compute_vel(stream_spec):
	"""
	This function computes the horizontal velocity fields in the spectral space multiplied by cos(latitude)
	given the spectral coefficients of the stream function. The spectral field coefficients of triangular
	truncation lmax need to be shaped as (2,lmax+1,lmax+1).

	To do so, it computes the curl of the stream function:
	--> V = Nabla x psi
		cos(theta) * u = - cos(theta)/R * d/dtheta psi
		cos(theta) * v = 1/R * d/dlambda psi

	PARAMETERS:
	(input) -->
		stream_spec : 3D array (2, lmax+1, lmax+1) of real spherical harmonics coefficients
					[0,:,:] for cosine coefficients and [1,:,:] for sine coefficients
	(output) -->
		u_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the zonal velocity field multiplied by cos(latitude)
		v_spec : 3D array (2, lmax+1, lmax+1) with the spectral coefficients of the meridional velocity field multiplied by cos(latitude)
	"""

	# We compute the spectral curl of the stream function
	u_spec = - theta_derivative(stream_spec)
	v_spec = lambda_derivative(stream_spec)

	return u_spec, v_spec


if __name__ == '__main__':
	# We set a time counter to keep track of the total execution time of the code
	start_time = time.time()

	# We generate an output folder to save the results of the simulation
	output_dir = "output_spherical/"
	os.makedirs(output_dir, exist_ok=True)

	# We define all the main parameters that will be used in the simulation
	print("Obtaining model parameters ...\n")

	# We first import all the initial parameters defined in the config.py file
	from config import *

	# We first verify that truncation is lower than the grid sampling
	if lmax > sampl:
		raise ValueError("lmax must be smaller or equal to sampl")
	
	if gridtype == 'DH':
		# DH grid scheme has less spectral resolution: nlat_grid ~ 2*sampling
		nlat_grid = 2 * sampl + 2
		lat_grid = np.linspace(90, -90, nlat_grid)					# 90º to -90º
		lon_grid = np.linspace(0, 360, 2*nlat_grid, endpoint=False) # 0º to 360º (not included)
		lons_grid, lats_grid = np.meshgrid(lon_grid, lat_grid)
	elif gridtype == 'GLQ':
		# GLQ grid scheme has highest spectral resolution: nlat_grid ~ sampling
		nlat_grid = sampl
		glq_nodes, glq_weights = pysh.expand.SHGLQ(nlat_grid)	# Hermite polynomials zeros
		lat_grid = np.degrees(np.arcsin(glq_nodes))				# Gaussian latitudes
		lon_grid = np.linspace(0, 360, 2*nlat_grid+1)			# 0º to 360º (included)
		lons_grid, lats_grid = np.meshgrid(lon_grid, lat_grid)

	# We precompute some useful parameters
	f = 2 * Omega * np.sin(np.pi * lats_grid / 180)	# Coriolis parameter
	derfact = 1 / np.cos(np.pi * lats_grid / 180)		# Spherical scale factor
	derfact[0] = derfact[1]			# Avoid division by zero
	derfact[-1] = derfact[-2]		# Avoid division by zero

	# We build the laplacian operators in the spectral space (with desired truncation lmax)
	l = np.arange(lmax + 1).reshape(1, -1, 1)		# i.e. [1, lmax+1, 1]
	lap = - l * (l + 1) / R**2			# Normal laplacian
	lap2 = lap**2						# Squared laplacian (for hyperdiffiusion n=2)
	inv_lap = np.zeros_like(lap)		# Inverse laplacian
	inv_lap[l>0] = 1 / lap[l>0]		# Avoid division by zero

	# We compute the hyperdiffusion coefficient (scaled accordingly)
	kmax = (lmax * (lmax + 1)) / R**2		# Maximum wave mode
	eta = 1 / (tau_d * kmax**2)
	hyp_denom1 = 1 / (1 + dt * eta * lap2)			# Implicit operator for Euler scheme
	hyp_denom2 = 1 / (1 + 2 * dt * eta * lap2)		# Implicit operator for Leapfrog scheme

	# We generate empty lists to keep track of the conserved quantities:
	# kinetic energy, enstrophy and mean vorticity
	energies = []
	enstrophies = []
	mean_vorticities = []
	# And also to save the evolution fields
	times = []
	streamfunctions = []
	vorticities = []

	# We create a folder in 'output' to save the results of the specific experiment
	os.makedirs(output_dir + output_name, exist_ok=True)


	# Now, we generate the initial velocity fields
	print("Generating initial fields ...\n")

	# We configure the interpolator from regular grid to DH grid
	interp_z = RegularGridInterpolator((lat, lon), zeta0, bounds_error=False, fill_value=None)

	# We interpolate the original vorticity field to the DH grid
	zeta0_grid = interp_z((lats_grid, lons_grid))
	print(f'Initial grid shape: {zeta0_grid.shape}')

	# We convert it to spectral space
	zeta0_spec = grid2spec(zeta0_grid, gridtype)
	print(f'Spec shape: {zeta0_spec.shape}')

	# We obtain the stream function field from the relative vorticity
	psi0_spec = inv_lap * zeta0_spec
	psi0 = spec2grid(psi0_spec, gridtype)
	print(f'Final grid shape: {psi0.shape}')

	# And we extract the horizontal velocity fields from the stream function
	u0_spec, v0_spec = compute_vel(psi0_spec)
	u0 = spec2grid(u0_spec, gridtype) * derfact
	v0 = spec2grid(v0_spec, gridtype) * derfact




	# os.makedirs('temp/', exist_ok=True)

	# fig, ax = plt.subplots()
	# im = ax.pcolormesh(lons_dh,lats_dh,zeta0_grid)
	# cbar = fig.colorbar(im, ax=ax)
	# cbar.set_label('Vorticity (1/s)')
	# ax.set_title(f'Vorticity field at t = {0/3600:.2f}h')
	# fig.tight_layout()

	# fig_name = f"vorticity_field_{output_name}_t{0/3600:.2f}h.png"
	# fig.savefig('temp/' + fig_name)
	# plt.close(fig)

	# fig, ax = plt.subplots()
	# im = ax.pcolormesh(lons_dh,lats_dh,psi0)
	# cbar = fig.colorbar(im, ax=ax)
	# cbar.set_label('psi (m^2/s)')
	# ax.set_title(f'Stream function field at t = {0/3600:.2f}h')
	# fig.tight_layout()

	# fig_name = f"psi_field_{output_name}_t{0/3600:.2f}h.png"
	# fig.savefig('temp/' + fig_name)
	# plt.close(fig)

	# fig, ax = plt.subplots()
	# im = ax.pcolormesh(lons_dh,lats_dh,u0)
	# cbar = fig.colorbar(im, ax=ax)
	# cbar.set_label('Velocity (m/s)')
	# ax.set_title(f'U field at t = {0/3600:.2f}h')
	# fig.tight_layout()

	# fig_name = f"u_field_{output_name}_t{0/3600:.2f}h.png"
	# fig.savefig('temp/' + fig_name)
	# plt.close(fig)




	# We compute all the conserved values
	energy = np.mean(0.5 * (u0**2 + v0**2))
	enstrophy = np.sum((zeta0_grid + f)**2 / 2.0)
	zetamean = np.mean(zeta0_grid)
	# And save them in the lists
	energies.append(energy)
	enstrophies.append(enstrophy)
	mean_vorticities.append(zetamean)

	# We also save the initial fields
	times.append(ti)
	if gridtype == 'GLQ':		# Remove the extra 360º longitude band
		streamfunctions.append(psi0[:,:-1].copy())
		vorticities.append(zeta0_grid[:,:-1].copy())
	else: 
		streamfunctions.append(psi0.copy())
		vorticities.append(zeta0_grid.copy())

	# Finally we set the next save time
	next_save_time = save_time


	# We start the time integration
	print("Starting time integration ...")

	# We compute the advection term (spec --> grid --> spec)
	adv0_spec = compute_adv(u0, v0, zeta0_grid)

	# And perform a forward Euler step in time for the first integration
	zeta_spec = zeta0_spec
	zetaold_spec = zeta0_spec
	# We apply the hyperdiffusion implicitly (i.e. (1+dt*hyp)zeta_i+1 = rhs_i))
	zetanew_spec = (zeta_spec + dt * adv0_spec) * hyp_denom1
	# zetanew = spec2grid(zetanew_spec) * derfact
	zetanew = spec2grid(zetanew_spec, gridtype)

	# We can also extract the new stream function field
	psi_spec = inv_lap * zetanew_spec
	
	# And extract the new velocity fields
	u_spec, v_spec = compute_vel(psi_spec)
	u = spec2grid(u_spec, gridtype) * derfact
	v = spec2grid(v_spec, gridtype) * derfact




	# fig, ax = plt.subplots()
	# im = ax.pcolormesh(lons_dh,lats_dh,zetanew)
	# cbar = fig.colorbar(im, ax=ax)
	# cbar.set_label('Vorticity (1/s)')
	# ax.set_title(f'Vorticity field at t = {dt/3600:.2f}h')
	# fig.tight_layout()

	# fig_name = f"vorticity_field_{output_name}_t{dt/3600:.2f}h.png"
	# fig.savefig('temp/' + fig_name)
	# plt.close(fig)

	# fig, ax = plt.subplots()
	# psi = spec2grid(psi_spec)
	# im = ax.pcolormesh(lons_dh,lats_dh,psi)
	# cbar = fig.colorbar(im, ax=ax)
	# cbar.set_label('psi (m^2/s)')
	# ax.set_title(f'Stream function field at t = {dt/3600:.2f}h')
	# fig.tight_layout()

	# fig_name = f"psi_field_{output_name}_t{dt/3600:.2f}h.png"
	# fig.savefig('temp/' + fig_name)
	# plt.close(fig)

	# fig, ax = plt.subplots()
	# im = ax.pcolormesh(lons_dh,lats_dh,u)
	# cbar = fig.colorbar(im, ax=ax)
	# cbar.set_label('Velocity (m/s)')
	# ax.set_title(f'U field at t = {dt/3600:.2f}h')
	# fig.tight_layout()

	# fig_name = f"u_field_{output_name}_t{dt/3600:.2f}h.png"
	# fig.savefig('temp/' + fig_name)
	# plt.close(fig)




	# Again, we comptute the conserved values
	energy = np.mean(0.5 * (u**2 + v**2))
	enstrophy = np.sum((zetanew + f)**2 / 2.0)
	zetamean = np.mean(zetanew)
	# And save them in the lists
	energies.append(energy)
	enstrophies.append(enstrophy)
	mean_vorticities.append(zetamean)


	# Now, we can start the main integration loop
	for t in range(ti+2*dt, tf+dt, dt):

		# We show on screen an updating counter showing the elapsed time of computation and 
		# the simulation time to visualize the program progress
		elapsed = time.time() - start_time
		sys.stdout.write(f"\rElapsed time: {elapsed:.2f}s | Simulation time: {t/3600:.2f}h")
		sys.stdout.flush()

		# We update the vorticity fields from last iteration 
		zetaold_spec = zeta_spec
		zeta_spec = zetanew_spec
		zeta = zetanew
		
		# We compute the advection term
		adv_spec = compute_adv(u, v, zeta)

		# Now, a Leapfrog scheme is used to perform the time integration
		# Again, we apply the hyperdiffusion implicitly (i.e. (1+2dt*hyp)zeta_i+1 = rhs_i))
		zetanew_spec = (zetaold_spec + 2 * dt * adv_spec) * hyp_denom2

		# After the time step, we apply a Robert-Asselin-Williams filter to reduce the
		# computational mode amplitude while still mantaining nearly the second order precision
		# of the leapfrog scheme
		# We compute the correcting term (a centered difference)
		delta = zetanew_spec - 2.0*zeta_spec + zetaold_spec
		# And then we apply this correction to the current and new vorticity fields with a RAW filter
		# damping it with nu and displacing zeta forwards and zetanew backwards with alpha
		zeta_spec += nu*alpha/2.0 * delta
		zetanew_spec += - nu*(1-alpha)/2.0 * delta
		# zetanew = spec2grid(zetanew_spec) * derfact
		zetanew = spec2grid(zetanew_spec, gridtype)

		# Now we can extract the new stream function field
		psi_spec = inv_lap * zetanew_spec

		# And compute the new velocity fields
		u_spec, v_spec = compute_vel(psi_spec)
		u = spec2grid(u_spec, gridtype) * derfact
		v = spec2grid(v_spec, gridtype) * derfact

		# Finally, we comptute the conserved values
		energy = np.mean(0.5 * (u**2 + v**2))
		enstrophy = np.sum((zetanew + f)**2 / 2.0)
		zetamean = np.mean(zetanew)
		# And save them in the lists
		energies.append(energy)
		enstrophies.append(enstrophy)
		mean_vorticities.append(zetamean)




		# fig, ax = plt.subplots()
		# im = ax.pcolormesh(lons_dh,lats_dh,zetanew)
		# cbar = fig.colorbar(im, ax=ax)
		# cbar.set_label('Vorticity (1/s)')
		# ax.set_title(f'Vorticity field at t = {t/3600:.2f}h')
		# fig.tight_layout()

		# fig_name = f"vorticity_field_{output_name}_t{t/3600:.2f}h.png"
		# fig.savefig('temp/' + fig_name)
		# plt.close(fig)

		# fig, ax = plt.subplots()
		# psi = spec2grid(psi_spec)
		# im = ax.pcolormesh(lons_dh,lats_dh,psi)
		# cbar = fig.colorbar(im, ax=ax)
		# cbar.set_label('psi (m^2/s)')
		# ax.set_title(f'Stream function field at t = {t/3600:.2f}h')
		# fig.tight_layout()

		# fig_name = f"psi_field_{output_name}_t{t/3600:.2f}h.png"
		# fig.savefig('temp/' + fig_name)
		# plt.close(fig)

		# fig, ax = plt.subplots()
		# im = ax.pcolormesh(lons_dh,lats_dh,u)
		# cbar = fig.colorbar(im, ax=ax)
		# cbar.set_label('Velocity (m/s)')
		# ax.set_title(f'U field at t = {t/3600:.2f}h')
		# fig.tight_layout()

		# fig_name = f"u_field_{output_name}_t{t/3600:.2f}h.png"
		# fig.savefig('temp/' + fig_name)
		# plt.close(fig)





		# Each iteration we check if the simulation is diverging
		if np.isinf(zetanew).any():
			raise ValueError(f"Infinity detected in vorticity field at t = {t/3600:.2f}h")
		elif np.isnan(zetanew).any():
			raise ValueError(f"NaN detected in vorticity field at t = {t/3600:.2f}h")

		# Every time we get to the save interval, we save a snapshot of the psi and zeta fields
		if t >= next_save_time:
			times.append(t)
			psi = spec2grid(psi_spec, gridtype)
			if gridtype == 'GLQ':	# We remove the extra 360º longitude band
				streamfunctions.append(psi[:,:-1].copy())
				vorticities.append(zetanew[:,:-1].copy())
			else:
				streamfunctions.append(psi.copy())
				vorticities.append(zetanew.copy())
			next_save_time += save_time


	# In the end, we save the used config file in '.txt' format and the simulation data into
	# NetCDF4 Datasets
	print('')
	print("Saving simulation results...")

	# We first copy the config file used
	with (open("config.py", 'r') as file, 
		open(output_dir + output_name + f"/params_{output_name}.txt", 'w') as file_copy):
		file_copy.write(file.read())

	# Then, we start by creating the directory where the data will be saved
	data_dir = output_dir + output_name + "/data/"
	os.makedirs(data_dir, exist_ok=True)

	# First we save the conserved values
	cons = xr.Dataset(
		{
			'kinetic_energy': (['iteration'], energies),
			'enstrophy': (['iteration'], enstrophies),
			'mean_vorticity': (['iteration'], mean_vorticities)
		},
		coords={
			'iteration': np.arange(len(energies))
		}
	)

	cons.attrs['description'] = 'Evolution of the conserved values during the simulation'
	cons['kinetic_energy'].attrs = {
		'description': 'Mean kinetic energy of the of the field at each iteration',
		'units': 'm^2/s^2',
		'long_name': 'Kinetic energy'
	}
	cons['enstrophy'].attrs = {
		'description': 'Mean enstrophy of the of the field at each iteration',
		'units': '1/s^2',
		'long_name': 'Enstrophy'
	}
	cons['mean_vorticity'].attrs = {
		'description': 'Mean vorticity of the of the field at each iteration',
		'units': '1/s',
		'long_name': 'Mean vorticity',
		'positive': 'Cyclonic'
	}
	cons['iteration'].attrs['description'] = 'Iteration number in the simulation'
	
	cons_file = f"conserved_values_{output_name}.nc"
	cons.to_netcdf(data_dir + cons_file)

	# And then the vorticity and stream function snapshots

	# First, we generate the standard time, latitude and longitude coordinates
	if gridtype == 'GLQ':
		lon_grid = lon_grid[:-1]	# We remove the extra 360º longitude band

	time_steps = np.array(times, dtype='timedelta64[s]')
	date_times = start_date + time_steps	# Simulation dates

	time_coord = xr.DataArray(
		date_times,
		dims='time',
		attrs={
			'long_name': 'time',
			'standard_name': 'time'
		}
	)
	lat_coord = xr.DataArray(
		lat_grid,
		dims='lat',
		attrs={
			'units': 'degrees_north',
			'long_name': 'latitude',
			'standard_name': 'latitude',
			'stored_direction': 'decreasing'
		}
	)
	lon_coord = xr.DataArray(
		lon_grid,
		dims='lon',
		attrs={
			'units': 'degrees_east',
			'long_name': 'longitude',
			'standard_name': 'longitude'
		}
	)

	# Then we create the dataset and save the fields with their corresponding attributes
	evo = xr.Dataset(
		{
			'streamfunction': (['time', 'lat', 'lon'], np.stack(streamfunctions)),
			'vorticity': (['time', 'lat', 'lon'], np.stack(vorticities))
		},
		coords={
			'time': time_coord,
			'lat': lat_coord,
			'lon': lon_coord
		}
	)

	evo.attrs = {
		'description': 'Evolution of 500 hPa vorticity and stream function fields through BVE simulation',
		'Conventions': 'DF-1.7',
		'history': f'Created on {time.ctime()}',
		'source': 'Global barotropic vorticity equation simulation at 500 hPa in Python'
	}
	evo['streamfunction'].attrs = {
		'description': '2D simulated stream function field',
		'units': 'm**2 s**-1',
		'long_name': 'Stream function',
		'standard_name': 'streamfunction',
		'gridType': f'{gridtype} (T{lmax})'
	}
	evo['vorticity'].attrs = {
		'description': '2D simulated vorticity field',
		'units': 's**-1',
		'long_name': 'Vorticity (relative)',
		'standard_name': 'vorticity',
		'positive': 'Cyclonic',
		'gridType': f'{gridtype} (T{lmax})'
	}

	evo_file = f"fields_evolution_{output_name}.nc"
	evo.to_netcdf(data_dir + evo_file)


	# FIGURE PLOTTING ==============================================================================

	print("\nGenerating figures...")

	# We import the required libraries for the plots
	import matplotlib.pyplot as plt
	import matplotlib.ticker as ticker
	from matplotlib.colors import BoundaryNorm
	import imageio

	# We create the directory where the figures will be saved
	im_dir = output_dir + output_name + "/figures/"
	os.makedirs(im_dir, exist_ok=True)
	os.makedirs(im_dir + "temp_frames/", exist_ok=True)

	# 1) Evolution of the conserved values

	# We first read and extract the information in the corresponding Dataset
	cons = xr.open_dataset(data_dir + cons_file, engine='netcdf4')

	energies = cons['kinetic_energy']
	enstrophies = cons['enstrophy']
	vorticity_means = cons['mean_vorticity']
	iterations = cons['iteration']

	# We establish a fixed format for the y axis
	y_formatter = ticker.ScalarFormatter(useOffset=True, useMathText=True)

	# Then we plot the evolution of all the conserved magnitudes in a triple figure
	fig, axs = plt.subplots(3,1, figsize=(8,8), sharex=True)
	ax1, ax2, ax3 = axs

	ene=ax1.plot(iterations,energies, label='Mean kinetic energy')
	ax1.set_title("Evolution of conserved magnitudes")
	ax1.set_ylabel('Mean kinetic energy (J/kg)')
	ax1.set_xlim(iterations[0],iterations[-1])
	ax1.yaxis.set_major_formatter(y_formatter)
	ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	ens=ax2.plot(iterations,enstrophies, label='Enstrophy')
	ax2.set_ylabel(r'Enstrophy (1/s$^2$)')
	ax2.set_xlim(iterations[0],iterations[-1])
	ax2.yaxis.set_major_formatter(y_formatter)
	ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	zet=ax3.plot(iterations,vorticity_means, label='Mean vorticity')
	ax3.set_ylabel('Mean vorticity (1/s)')
	ax3.set_xlabel('Nº iterations')
	ax3.set_xlim(iterations[0],iterations[-1])
	ax3.yaxis.set_major_formatter(y_formatter)
	ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useOffset=True)

	fig.tight_layout()

	plt.savefig(im_dir + cons_file[:-3] + ".png", dpi=150)
	

	# 2) Evolution of the vorticity and stream function fields

	# Again, we read and extract all the information contained in the Dataset
	evo = xr.open_dataset(data_dir + evo_file, engine='netcdf4')

	streamfunctions = evo['streamfunction']
	vorticities = evo['vorticity']
	lon = evo['lon']
	lat = evo['lat']
	lons, lats = np.meshgrid(lon, lat)
	time_step = (evo['time'].values[1].astype('datetime64[s]') - 
			evo['time'].values[0].astype('datetime64[s]')).astype(int)
	times = [i*time_step / 3600 for i in range(len(evo['time'].values))]

	# First we plot the vorticity field evolution

	# We pick the levels of the colormap that best fit our data

	# To do so, we use the percentiles to know the different scales of the data
	all_data = vorticities.values.flatten()

	# We compute the percentiles
	p1 = np.percentile(all_data, 1)
	p99 = np.percentile(all_data, 99)

	# And we create a symmetric level scale based on the percentiles to give importance to
	# the range of values where most of the data is
	max_abs = max(abs(p1), abs(p99))
	levels = np.concatenate([
			np.linspace(-max_abs, -max_abs/2, 5),
			np.linspace(-max_abs/2, max_abs/2, 11),
			np.linspace(max_abs/2, max_abs, 5)
			])

	levels = np.unique(levels)
	norm = BoundaryNorm(levels, 256)

	# Then we plot each of the saved snapshots and generate a GIF
	images = []

	for i in range(len(times)):

		fig, ax = plt.subplots(figsize=(8,4))
		mesh = ax.contourf(lons, lats, vorticities[i], cmap='coolwarm', 
					 		norm=norm, levels=levels, extend='both')
		cbar = fig.colorbar(mesh, ax=ax, extend='both', label='Vorticity (1/s)')
		cbar.set_ticks(levels)
		ax.set_title(f'Vorticity field at t = {times[i]}h')
		ax.set_xlabel(r'$\lambda$ (º)')
		ax.set_ylabel(r'$\phi$ (º)')
		fig.tight_layout()

		fig_name = f"vorticity_field_{output_name}_t{times[i]}h.png"
		fig.savefig(im_dir + "temp_frames/" + fig_name, dpi=150)
		plt.close(fig)

		images.append(imageio.v2.imread(im_dir + "temp_frames/" + fig_name))
	
	gif_name = f"vorticity_field_{output_name}_evolution.gif"
	imageio.mimsave(im_dir + gif_name, images, duration=250, loop=0)

	# Finally, we plot the stream function field evolution
	images = []

	for i in range(len(times)):

		fig, ax = plt.subplots(figsize=(8,4))
		mesh = ax.contour(lons,lats,streamfunctions[i],cmap='coolwarm')
		cbar = fig.colorbar(mesh, ax=ax, label=r'Stream function ($\mathrm{m^2/s}$)')
		ax.set_title(f'Stream function field at t = {times[i]}h')
		ax.set_xlabel(r'$\lambda$ (º)')
		ax.set_ylabel(r'$\phi$ (º)')
		fig.tight_layout()

		fig_name = f"stream_function_field_{output_name}_t{times[i]}h.png"
		fig.savefig(im_dir + "temp_frames/" + fig_name, dpi=150)
		plt.close(fig)

		images.append(imageio.v2.imread(im_dir + "temp_frames/" + fig_name))
	
	gif_name = f"stream_function_field_{output_name}_evolution.gif"
	imageio.mimsave(im_dir + gif_name, images, duration=250, loop=0)