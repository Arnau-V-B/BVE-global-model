import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pyshtools as pysh
import time
import sys
import os


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


def compute_vort(u, v, R=6371*10**3):
	u_spec = grid2spec(u)
	v_spec = grid2spec(v)
	
	grad_u = pysh.expand.MakeGradientDH(u_spec,sampling=2,radius=R)
	grad_v = pysh.expand.MakeGradientDH(v_spec,sampling=2,radius=R)

	vort = grad_v[1] - grad_u[0]

	return vort


def compute_adv(u, v, vort, R=6371*10**3):
	u_zeta = u * (vort + f)
	v_zeta = v * (vort + f)

	u_zeta_spec = grid2spec(u_zeta)
	v_zeta_spec = grid2spec(v_zeta)

	grad_u_zeta = pysh.expand.MakeGradientDH(u_zeta_spec,sampling=2,radius=R)
	grad_v_zeta = pysh.expand.MakeGradientDH(v_zeta_spec,sampling=2,radius=R)

	adv = grad_u_zeta[1] + grad_v_zeta[0]

	return - adv


def compute_vel(stream_spec, R=6371*10**3):
	grad_stream = pysh.expand.MakeGradientDH(stream_spec,sampling=2,radius=R)

	u = - grad_stream[0]
	v = grad_stream[1]

	return u, v


if __name__ == '__main__':
	# We set a time counter to keep track of the total execution time of the code
	start_time = time.time()

	# We generate an output folder to save the results of the simulation
	output_dir = "output_spherical/"
	os.makedirs(output_dir, exist_ok=True)

	# We define all the main parameters that will be used in the simulation
	print("Obtaining model parameters ...\n")

	from config import *
	
	# We generate the Driscoll-Healy spherical grid (i.e. equally spaced Nx2N)
	nlat_dh = len(lat) - 1
	lat_dh = np.linspace(-90, 90, nlat_dh)
	lon_dh = np.linspace(0, 360, 2*nlat_dh, endpoint=False)
	lons_dh, lats_dh = np.meshgrid(lon_dh, lat_dh)

	# We define the Coriolis parameter field and the hyperdifussion coefficient
	f = 2 * Omega * np.sin(np.pi * lats_dh / 180)
	eta = 10 * (3 * nlat_dh / np.pi)**4

	# We build the laplacian operators in the spectral space (with truncation: nlat = 2(lmax+1))
	lmax = nlat_dh//2 - 1
	l = np.arange(lmax + 1).reshape(1, -1, 1)		# i.e. [1, lmax+1, 1]
	lap = - l * (l + 1) / R**2
	lap2 = lap**2
	inv_lap = np.zeros_like(lap)
	inv_lap[l>0] = 1 / lap[l>0]

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

	# We configure the interpolators from regular grid to DH grid
	interp_u = RegularGridInterpolator((lat, lon), u0, bounds_error=False, fill_value=None)
	interp_v = RegularGridInterpolator((lat, lon), v0, bounds_error=False, fill_value=None)

	# We interpolate the velocity components to the DH grid
	u0_grid = interp_u((lats_dh, lons_dh))
	v0_grid = interp_v((lats_dh, lons_dh))

	# We compute the relative vorticity from the horizontal velovity fields
	zeta0 = compute_vort(u0_grid, v0_grid)

	# We obtain the stream function field from the relative vorticity
	zeta0_spec = grid2spec(zeta0)
	psi0_spec = inv_lap * zeta0_spec
	psi0 = spec2grid(psi0_spec)

	# We compute all the conserved values
	energy = np.mean(0.5 * (u0_grid**2 + v0_grid**2))
	enstrophy = np.sum((zeta0 + f)**2 / 2.0)
	zetamean = np.mean(zeta0)
	# And save them in the lists
	energies.append(energy)
	enstrophies.append(enstrophy)
	mean_vorticities.append(zetamean)

	# We also save the initial fields
	times.append(ti)
	streamfunctions.append(psi0.copy())
	vorticities.append(zeta0.copy())

	# Finally we set the next save time
	next_save_time = save_time


	# We start the time integration
	print("Starting time integration ...")

	# We compute the advection term and bring it to spectral space
	adv0 = compute_adv(u0_grid, v0_grid, zeta0)
	adv0_spec = grid2spec(adv0)

	# We compute the hyperdiffusion term
	hyp0_spec = eta * lap2 * zeta0_spec

	# Now, we can get the RHS of the BVE in the spectral space
	rhs0_spec = adv0_spec - hyp0_spec

	# And perform a forward Euler step in time for the first integration
	zeta_spec = zeta0_spec
	zetaold_spec = zeta0_spec
	zetanew_spec = zeta_spec + dt * rhs0_spec
	zetanew = spec2grid(zetanew_spec)

	# We can also extract the new stream function field
	psi_spec = inv_lap * zetanew_spec
	
	# And extract the new velocity fields
	u, v = compute_vel(psi_spec)

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
		
		# We compute the advection and hyperdiffusion terms
		adv = compute_adv(u, v, zeta)
		adv_spec = grid2spec(adv)
		hyp_spec = eta * lap2 * zeta_spec

		# And get the RHS of the BVE
		rhs_spec = adv_spec - hyp_spec

		# Now, a Leapfrog scheme is used to perform the time integration
		zetanew_spec = zetaold_spec + 2 * dt * rhs_spec

		# After the time step, we apply a Robert-Asselin-Williams filter to reduce the
		# computational mode amplitude while still mantaining nearly the second order precision
		# of the leapfrog scheme
		# We compute the correcting term (a centered difference)
		delta = zetanew_spec - 2.0*zeta_spec + zetaold_spec
		# And then we apply this correction to the current and new vorticity fields with a RAW filter
		# damping it with nu and displacing zeta forwards and zetanew backwards with alpha
		zeta_spec += nu*alpha/2.0 * delta
		zetanew_spec += - nu*(1-alpha)/2.0 * delta
		zetanew = spec2grid(zetanew_spec)

		# Now we can extract the new stream function field
		psi_spec = inv_lap * zetanew_spec

		# And compute the new velocity fields
		u, v = compute_vel(psi_spec)

		# Finally, we comptute the conserved values
		energy = np.mean(0.5 * (u**2 + v**2))
		enstrophy = np.sum((zetanew + f)**2 / 2.0)
		zetamean = np.mean(zetanew)
		# And save them in the lists
		energies.append(energy)
		enstrophies.append(enstrophy)
		mean_vorticities.append(zetamean)

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

	# We start by creating the directory where the data will be saved
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
			'streamfunction': (['time', 'y', 'x'], np.stack(streamfunctions)),
			'vorticity': (['time', 'y', 'x'], np.stack(vorticities))
		},
		coords={
			'time': times,
			'lat': (['y', 'x'], lats_dh),
			'lon': (['y', 'x'], lons_dh)
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
	xs = evo['lon']
	ys = evo['lat']
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
		mesh = ax.contourf(xs, ys, vorticities[i], cmap='coolwarm', 
					 		norm=norm, levels=levels, extend='both')
		cbar = fig.colorbar(mesh, ax=ax, extend='both', label='Vorticity (1/s)')
		cbar.set_ticks(levels)
		ax.set_title(f'Vorticity field at t = {times[i]}h')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
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
		mesh = ax.contour(xs,ys,streamfunctions[i],cmap='coolwarm')
		cbar = fig.colorbar(mesh, ax=ax, label=r'Stream function ($\mathrm{m^2/s}$)')
		ax.set_title(f'Stream function field at t = {times[i]}h')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		fig.tight_layout()

		fig_name = f"stream_function_field_{output_name}_t{times[i]}h.png"
		fig.savefig(im_dir + "temp_frames/" + fig_name, dpi=150)
		plt.close(fig)

		images.append(imageio.v2.imread(im_dir + "temp_frames/" + fig_name))
	
	gif_name = f"stream_function_field_{output_name}_evolution.gif"
	imageio.mimsave(im_dir + gif_name, images, duration=250, loop=0)