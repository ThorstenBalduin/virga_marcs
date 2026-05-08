#from bokeh.io import output_notebook
#from bokeh.plotting import show, figure
#from bokeh.palettes import Colorblind
#output_notebook()
import numpy as np
import pandas as pd
import astropy.units as u
from molmass import Formula

#here is pyeddy
import virga.justdoit as jdi
import virga.justplotit as jpi

#for doms mie file
import h5py
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

k_B = 1.380649e-16               #ergs/K
GRAV = 2.95                      # Gravity of the planet

########################## LOADING IN FROM MARCS ###############################
input_data_marcs='./data/marcs2virga.dat'
mieff_dir =  './virga/mieffs/'

condensates_start = ' # Begin Condensates'
condensates_end = ' # End Condensates'

with open(input_data_marcs, 'r') as f:
    lines = f.readlines()
start = next(i for i, l in enumerate(lines) if condensates_start in l)
end = next(i for i, l in enumerate(lines) if condensates_end in l)
block = lines[start+1:end]

condensates = {}
for line in block:
    parts = line.strip().lstrip('#').strip().split(':')
    molecules  = parts[0].strip()       
    ext_mmr = float(parts[1].strip())
    condensates[molecules] = ext_mmr

molecules = list(condensates.keys())
ext_mmr = np.array(list(condensates.values()))

# Need to add the read in the marcs2virga.dat
mean_molecular_weight = 2.2      # atmospheric mean molecular weight
metallicity = 10                 #atmospheric metallicity relative to Solar

T = np.genfromtxt(input_data_marcs, skip_header=end+2)[:,0]             #K
P = np.genfromtxt(input_data_marcs, skip_header=end+2)[:,1]*1e-6        #bar
kz = np.genfromtxt(input_data_marcs, skip_header=end+2)[:,2]            #

############################# SETTING UP VIRGA #################################
profile = pd.DataFrame({'pressure': P, 'temperature': T, 'kz': kz})

#set the run
a = jdi.Atmosphere(molecules,
                fsed=0.1,mh=metallicity,
                mmw = mean_molecular_weight)

#set the planet gravity
a.gravity(gravity=GRAV, gravity_unit=u.Unit('m/(s**2)'))

#Get pt profile for testing
#a.ptk(df = jdi.hot_jupiter())
a.ptk(df = profile)

#get full dictionary output
all_out = jdi.compute(a, as_dict=True,
                    directory=mieff_dir, ext_mmr=ext_mmr)

sig = list(all_out['scalar_inputs'].values())[2]  # Geometric standard deviation, is a scalar

pp = np.zeros((len(P[0:52]), len(molecules)))
n_c = np.zeros((len(P[0:52]), len(molecules)))
rg = np.zeros((len(P[0:52]), len(molecules)))
#print(r)
for i, igas in enumerate(molecules):
    molname = igas
    mass_igas = Formula(molname).mass                   #amu
    cgs_mass_H = Formula("H").mass*u.u.to(u.g)          #g

    mmr_c = all_out['condensate_mmr'][:,i]              #
    mmr_tot = all_out['cond_plus_gas_mmr'][:,i]         # 
    rho_p = all_out['condensate_density'][i]            # g/cm^3
    rg[:,i] = all_out['mean_particle_r'][:,i]*1e-4       #convert from microns to cm

    for j in range(len(mmr_c)):                         # r, mmr_c and P, T arent same lengths
        pp[j][i] = (mean_molecular_weight/mass_igas) * P[j]*1e6 * (mmr_tot[j] - mmr_c[j])         #dyn/cm^2
        if rg[j][i] > 0:
            n_c[j][i] = (3*mmr_c[j]*mean_molecular_weight* cgs_mass_H *(P[j]*1e6))/(4*np.pi*rg[j][i]**3*np.exp(4.5*np.log(sig)**2)*rho_p*k_B*T[j])   #number density

final_dat = np.vstack([rg, pp, n_c])

output_path = './data/virga2marcs.dat'
with open(output_path, 'w') as file:
    file.write("P (bar) r (cm) pp (dyn/cm^2) n_c (1/cm^3)\n")
    for j in range(len(molecules)):
        file.write(f"\nGas: {molecules[j]}\n")
        for i in range(len(P)):
            if i == len(P)-1:
                file.write(f"{P[i]:.6e} {0} {0} {0}\n")
            else:
                file.write(f"{P[i]:.6e} {rg[i][j]:.6e} {pp[i][j]:.6e} {n_c[i][j]:.6e}\n")

def lognormal(N,rg,sig,r_array):
    # Ackerman and Marley 2001 lognormal distribution
    # Equation 9, can't find a later reference in the literature
    # units are dn/dr so cm^-3cm^-1
    # N and rg are ashape (nz, ncond), sig is a scalar, and r_array is shape (nbins,)
    # want final output to be shape (nz, cond, nbins) so that we can integrate over the size distribution for each layer 
    # and then the condensate species separately
    
    print(np.shape(N), np.shape(rg), np.shape(r_array))
    prefactor = N[:,np.newaxis] / (r_array[np.newaxis, np.newaxis, :] * np.log(sig) * np.sqrt(2 * np.pi))
    exponent = - (np.log(r_array[np.newaxis, np.newaxis, :] / rg[:,np.newaxis]) / (np.sqrt(2) * np.log(sig)))**2
    print(np.shape(prefactor), np.shape(exponent))
    
    distribution = prefactor * np.exp(exponent)  # shape (nz, nbins)
    print(np.shape(distribution))
    
    return distribution

def lognormal_abs_sca_sum_matrix(r_array, distribution, Qabs, Qsca):
    # Integrate over radius bins first (per condensate), then sum condensates.
    # distribution: (nz, ncond, nbins)
    # Qabs, Qsca expected: (ncond, nbins, nwavelengths)
    # r_array: (nbins,)
    
    print("We're in the matrix now")

    nz, ncond, nbins = distribution.shape
    nwavelengths = Qabs.shape[-1]
    
    # Build per-bin widths dr (same length as r_array)
    dr = np.empty_like(r_array)
    dr[1:-1] = 0.5 * (r_array[2:] - r_array[:-2])
    dr[0] = r_array[1] - r_array[0]
    dr[-1] = r_array[-1] - r_array[-2]

    # Geometric factor (nz, ncond, nbins)
    factor = distribution * np.pi * r_array**2 * dr
    
    factor_2D = factor.reshape(nz, ncond*nbins)
    Qabs_2D = Qabs.reshape(ncond*nbins, nwavelengths)
    Qsca_2D = Qsca.reshape(ncond*nbins, nwavelengths)
    
    kappa_abs = factor_2D @ Qabs_2D  # shape (nz, nwavelengths)
    kappa_sca = factor_2D @ Qsca_2D  # shape (nz, nwavelengths)
    
    print('done')
            
    return kappa_abs, kappa_sca # shape (nz, nwavelengths) 

def read_in_mie_h5(file_path):
    """Load precomputed Mie data from HDF5 file."""
    with h5py.File(file_path, "r") as f:
        mie_radii = f["sizes_um"][:]
        wavelengths_um = f["wavelengths_um"][:]
        qabs = f["qabs"][:]
        qsca = f["qsca"][:]

    # Ensure orientation is (nbins, nwavelength)
    if qabs.shape[0] != mie_radii.size and qabs.shape[1] == mie_radii.size:
        qabs = qabs.T
    if qsca.shape[0] != mie_radii.size and qsca.shape[1] == mie_radii.size:
        qsca = qsca.T

    return mie_radii, wavelengths_um, qabs, qsca

for i, igas in enumerate(molecules):
    print(f"Processing {igas}...")
    h5_path = f"./Mie_data/Mie_{igas}.h5"
    mie_radii, wavelength, Qabs, Qsca = read_in_mie_h5(h5_path) #TODO need to actually loop over the materials
    # use mie_radii so that everything is on the same grid
    mie_radii_cm = mie_radii * 1e-4  # Convert microns to cm for the distribution function
    
    if i == 0:
        print("Qabs shape:", Qabs.shape)
        print("Qsca shape:", Qsca.shape)
        Qabs_all = np.zeros((len(molecules), Qabs.shape[0], Qabs.shape[1]))
        Qsca_all = np.zeros((len(molecules), Qsca.shape[0], Qsca.shape[1]))
        
    Qabs_all[i, :, :] = Qabs
    Qsca_all[i, :, :] = Qsca
    #print(mie_radii_cm)

distribution = lognormal(n_c, rg, sig, mie_radii_cm)
k_abs, k_sca = lognormal_abs_sca_sum_matrix(mie_radii_cm, distribution, Qabs_all, Qsca_all)

# Now we need to convert from cm^-1 to cm^2/g to compare with the gas opacities, 
# divide by the gas density in g/cm^3, which is rho_gas = P/(k_B*T) * mean_molecular_weight * m_H
# where P is in dyn/cm^2, T is in K, mean_molecular_weight is dimensionless, and m_H is in g. 
rho_gas = (P[:-1]*1e6) / (k_B * T[:-1]) * mean_molecular_weight * cgs_mass_H  # g/cm^3
kappa_abs_cm2_per_g = k_abs / rho_gas[:, np.newaxis]
kappa_sca_cm2_per_g = k_sca / rho_gas[:, np.newaxis]

# write these to a text file with a one line header
# columns are wavenumber (1/cm), pressure (bar), kappa_abs (cm^2/g), kappa_sca (cm^2/g)
output_path = './data/virga_mie_opacities.dat'
with open(output_path, 'w') as file:
    file.write("Wavenumber (1/cm) Pressure (dyn/cm^2) kappa_abs (cm^2/g) kappa_sca (cm^2/g)\n")
    for iw, wave in enumerate(wavelength[::-1]):
        waveno = 1e4 / wave  # convert micron to wavenumber in cm^-1
        for ip, p in enumerate(P[:-1]):
            file.write(f"{waveno:.6e} {p*1e6:.6e} {kappa_abs_cm2_per_g[ip, -iw]:.6e} {kappa_sca_cm2_per_g[ip, -iw]:.6e}\n")
            

# Re-read saved opacity table
opacity_file = "./data/virga_mie_opacities.dat"
df_op = pd.read_csv(
    opacity_file,
    sep=r"\s+",
    skiprows=1,
    names=["wavenumber_cm1", "pressure_dyn_cm2", "kappa_abs_cm2_g", "kappa_sca_cm2_g"],
)

# Convert to wavelength in microns for plotting
df_op["wavelength_um"] = 1e4 / df_op["wavenumber_cm1"]

pressures = np.sort(df_op["pressure_dyn_cm2"].unique())

# Initial pressure slice
p0 = pressures[0]
d0 = df_op[df_op["pressure_dyn_cm2"] == p0].sort_values("wavelength_um")

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.22)

line_abs, = ax.plot(d0["wavelength_um"], d0["kappa_abs_cm2_g"], label="kappa_abs")
line_sca, = ax.plot(d0["wavelength_um"], d0["kappa_sca_cm2_g"], label="kappa_sca")
line_ext, = ax.plot(d0["wavelength_um"], d0["kappa_abs_cm2_g"] + d0["kappa_sca_cm2_g"], label="kappa_ext", linestyle="--")

ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Opacity (cm$^2$/g)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title(f"Pressure = {p0:.3e} dyn/cm² ({p0*1e-6:.3e} bar)")
ax.legend()
ax.grid(alpha=0.3)

slider_ax = plt.axes([0.18, 0.08, 0.68, 0.04])
p_slider = Slider(
    ax=slider_ax,
    label="Pressure index",
    valmin=0,
    valmax=len(pressures) - 1,
    valinit=0,
    valstep=1,
)

def _update(val):
    idx = int(p_slider.val)
    p = pressures[idx]
    d = df_op[df_op["pressure_dyn_cm2"] == p].sort_values("wavelength_um")

    line_abs.set_xdata(d["wavelength_um"].to_numpy())
    line_abs.set_ydata(d["kappa_abs_cm2_g"].to_numpy())
    line_sca.set_xdata(d["wavelength_um"].to_numpy())
    line_sca.set_ydata(d["kappa_sca_cm2_g"].to_numpy())
    line_ext.set_xdata(d["wavelength_um"].to_numpy())
    line_ext.set_ydata((d["kappa_abs_cm2_g"] + d["kappa_sca_cm2_g"]).to_numpy())

    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"Pressure = {p:.3e} dyn/cm² ({p*1e-6:.3e} bar)")
    fig.canvas.draw_idle()

p_slider.on_changed(_update)
plt.show()