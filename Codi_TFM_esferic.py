import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pyshtools as pysh
import time
import sys
import os
import matplotlib.pyplot as plt


def grid2spec(field, grid='DH'):

	if grid == 'DH':
		field_spec = pysh.expand.SHExpandDH(field,sampling=2)

	elif grid == 'GLQ':
		lmax = field.shape[0]//2 - 1
		nlat_quad = 3 * (lmax + 1) // 2
		lmax_glq = nlat_quad - 1

		glq_nodes, glq_weights = pysh.expand.SHGLQ(lmax_glq)
		field_spec = pysh.expand.SHExpandGLQ(field, w=glq_weights, zero=glq_nodes, lmax_calc=lmax)

	return field_spec


def spec2grid(field_spec, grid='DH'):

	if grid == 'DH':
		field = pysh.expand.MakeGridDH(field_spec,sampling=2)

	elif grid == 'GLQ':
		lmax = field_spec.shape[0]//2 - 1
		nlat_quad = 3 * (lmax + 1) // 2
		lmax_glq = nlat_quad - 1

		glq_nodes, glq_weights = pysh.expand.SHGLQ(lmax_glq)
		field = pysh.expand.MakeGridGLQ(field_spec, zero=glq_nodes, lmax=lmax_glq)

	return field


def lambda_derivative(field_spec):

	lmax = field_spec.shape[1] - 1

	m = np.arange(lmax + 1)
	
	dlambda_C = m[None, :] * field_spec[1]
	dlambda_S = - m[None, :] * field_spec[0]

	return np.stack((dlambda_C, dlambda_S), axis=0) / R


def theta_derivative(field_spec):

	lmax = field_spec.shape[1] - 1
	dtheta = np.zeros_like(field_spec)

	m = np.arange(lmax + 1)

	for l in range(lmax + 1):
		m_valid = m[:l+1]

		eps_l = np.sqrt((l**2 - m_valid**2) / (4*l**2 - 1))
		eps_lp1 = np.sqrt(((l+1)**2 - m_valid**2) / (4*(l+1)**2 - 1))

		if l == 0:
			dtheta[:, l, :l+1] = (l+2) * eps_lp1 * field_spec[:, l+1, :l+1]
		elif l == lmax:
			dtheta[:, l, :l+1] = - (l-1) * eps_l * field_spec[:, l-1, :l+1]
		else:
			dtheta[:, l, :l+1] = ((l+2) * eps_lp1 * field_spec[:, l+1, :l+1]
								- (l-1) * eps_l * field_spec[:, l-1, :l+1])

	return dtheta / R


def compute_vort(u_spec, v_spec):
	
	u_theta_spec = theta_derivative(u_spec)
	v_lambda_spec = lambda_derivative(v_spec)

	vort_spec = v_lambda_spec - u_theta_spec

	return vort_spec


def compute_adv(u, v, vort):

	u_zeta = u * (vort + f)
	v_zeta = v * (vort + f)

	u_zeta_spec = grid2spec(u_zeta)
	v_zeta_spec = grid2spec(v_zeta)

	u_zeta_lambda_spec = lambda_derivative(u_zeta_spec)
	v_zeta_theta_spec = theta_derivative(v_zeta_spec)

	adv_spec = - (u_zeta_lambda_spec + v_zeta_theta_spec)

	return adv_spec


def compute_vel(stream_spec):

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

	from config import *
	global f, derfact
	
	# We generate the Driscoll-Healy spherical grid (i.e. equally spaced Nx2N)
	nlat_dh = len(lat) - 1
	lat_dh = np.linspace(90 - 0*90/nlat_dh, -90 + 0*90/nlat_dh, nlat_dh)
	lon_dh = np.linspace(0, 360, 2*nlat_dh, endpoint=False)
	lons_dh, lats_dh = np.meshgrid(lon_dh, lat_dh)

	# We precompute some useful parameters
	f = 2 * Omega * np.sin(np.pi * lats_dh / 180)	# Coriolis parameter
	derfact = 1 / np.cos(np.pi * lats_dh / 180)		# Spherical scale factor
	derfact[0] = derfact[1]			# Avoid division by zero
	derfact[-1] = derfact[-2]		# Avoid division by zero

	# We build the laplacian operators in the spectral space (with truncation: nlat = 2(lmax+1))
	lmax = nlat_dh//2 - 1
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
	zeta0_grid = interp_z((lats_dh, lons_dh))

	# We convert it to spectral space
	zeta0_spec = grid2spec(zeta0_grid)

	# We obtain the stream function field from the relative vorticity
	psi0_spec = inv_lap * zeta0_spec
	psi0 = spec2grid(psi0_spec)

	# And we extract the horizontal velocity fields from the stream function
	u0_spec, v0_spec = compute_vel(psi0_spec)
	u0 = spec2grid(u0_spec) * derfact
	v0 = spec2grid(v0_spec) * derfact




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
	zetanew = spec2grid(zetanew_spec)

	# We can also extract the new stream function field
	psi_spec = inv_lap * zetanew_spec
	
	# And extract the new velocity fields
	u_spec, v_spec = compute_vel(psi_spec)
	u = spec2grid(u_spec) * derfact
	v = spec2grid(v_spec) * derfact




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
		zetanew = spec2grid(zetanew_spec)

		# Now we can extract the new stream function field
		psi_spec = inv_lap * zetanew_spec

		# And compute the new velocity fields
		u_spec, v_spec = compute_vel(psi_spec)
		u = spec2grid(u_spec) * derfact
		v = spec2grid(v_spec) * derfact

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

		if np.isinf(zetanew).any():
			print("Infinity detected")
			break
		elif np.isnan(zetanew).any():
			print("NaN detected")
			break





		# Every time we get to the save interval, we save a snapshot of the psi and zeta fields
		if t >= next_save_time:
			times.append(t)
			psi = spec2grid(psi_spec)
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
	evo = xr.Dataset(
		{
			'streamfunction': (['time', 'lat', 'lon'], np.stack(streamfunctions)),
			'vorticity': (['time', 'lat', 'lon'], np.stack(vorticities))
		},
		coords={
			'time': times,
			'lat': lat_dh,
			'lon': lon_dh
		}
	)

	evo.attrs['description'] = 'Evolution of the vorticity and stream function fields during the simulation'
	evo['streamfunction'].attrs = {
		'description': '2D stream function field',
		'units': 'm^2/s',
		'long_name': 'Stream function'
	}
	evo['vorticity'].attrs = {
		'description': '2D vorticity field',
		'units': '1/s',
		'long_name': 'Vorticity',
		'positive': 'Cyclonic'
	}
	evo['lon'].attrs = {
		'description': 'Longitude coordinate',
		'units': 'm',
		'long_name': 'Longitude'
	}
	evo['lat'].attrs = {
		'description': 'Latitude coordinate',
		'units': 'm',
		'long_name': 'Latitude'
	}
	evo['time'].attrs = {
		'description': 'Time coordinate',
		'units': 's',
		'long_name': 'Time'
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
	times = [int(time/3600) for time in evo['time'].values]

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