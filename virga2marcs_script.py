from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.palettes import Colorblind
output_notebook()
import numpy as np
import pandas as pd
import astropy.units as u
from molmass import Formula

#here is pyeddy
import virga.justdoit as jdi
import virga.justplotit as jpi

#for doms mie file
import h5py

k_B = 1.380649e-16               #ergs/K
GRAV = 2.95                      # Gravity of the planet

########################## LOADING IN FROM MARCS ###############################
input_data_marcs='/groups/astro/mhundrup/virga_marcs/data/marcs2virga.dat'
mieff_dir =  '/groups/astro/mhundrup/virga_marcs/virga/mieffs/'

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


pp = np.zeros((len(P[0:52]), len(molecules)))
n_c = np.zeros((len(P[0:52]), len(molecules)))
r = np.zeros((len(P[0:52]), len(molecules)))
#print(r)
for i, igas in enumerate(molecules):
    molname = igas
    mass_igas = Formula(molname).mass                   #amu
    cgs_mass_H = Formula("H").mass*u.u.to(u.g)          #g

    mmr_c = all_out['condensate_mmr'][:,i]              #
    mmr_tot = all_out['cond_plus_gas_mmr'][:,i]         # 
    rho_p = all_out['condensate_density'][i]            # g/cm^3
    r[:,i] = all_out['mean_particle_r'][:,i]*1e-4       #convert from microns to cm

    for j in range(len(mmr_c)):                         # r, mmr_c and P, T arent same lengths
        pp[j][i] = (mean_molecular_weight/mass_igas) * P[j]*1e6 * (mmr_tot[j] - mmr_c[j])         #dyn/cm^2
        if r[j][i] > 0:
            n_c[j][i] = (3*mmr_c[j]*mean_molecular_weight* cgs_mass_H *(P[j]*1e6))/(4*np.pi*r[j][i]**3*rho_p*k_B*T[j])   #number density

final_dat = np.vstack([r, pp, n_c])

output_path = '/groups/astro/mhundrup/virga_marcs/data/virga2marcs.dat'
with open(output_path, 'w') as file:
    file.write("P (bar) r (cm) pp (dyn/cm^2) n_c (1/cm^3)\n")
    for j in range(len(molecules)):
        file.write(f"\nGas: {molecules[j]}\n")
        for i in range(len(P)):
            if i == len(P)-1:
                file.write(f"{P[i]:.6e} {0} {0} {0}\n")
            else:
                file.write(f"{P[i]:.6e} {r[i][j]:.6e} {pp[i][j]:.6e} {n_c[i][j]:.6e}\n")

def lognormal(N,rg,sig,r_array):
    # Ackerman and Marley 2001 lognormal distribution
    # Equation 9, can't find a later reference in the literature
    # units are dn/dr so cm^-3cm^-1
    
    prefactor = N / (r_array * np.log(sig) * np.sqrt(2 * np.pi))
    exponent = - (np.log(r_array / rg) / (np.sqrt(2) * np.log(sig)))**2
    return prefactor * np.exp(exponent)

def lognormal_abs_sca_sum(r_array, distribution, Qabs, Qsca):
    # Integrate the absorption and scattering over the size distribution
    # radii are in cm, Qabs and Qsca are dimensionless efficiencies, distribution is in cm^-3cm^-1
    # r_array and distribution are shape nbins, Qabs and Qsca are shape (nbins, nwavelengths)
    
    integrand_abs = distribution[:, np.newaxis] * Qabs * np.pi * r_array[:, np.newaxis]**2  # cm^-3cm^-1 * dimensionless * cm^2 = cm^-1
    integrand_sca = distribution[:, np.newaxis] * Qsca * np.pi * r_array[:, np.newaxis]**2
    return np.trapezoid(integrand_abs, r_array, axis=0), np.trapezoid(integrand_sca, r_array, axis=0)  # Integrate over the radius array to get total absorption and scattering coefficients in cm^-1

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

h5_path = "../Mie_data/Mie_H2O.h5"
mie_radii, wavelength, Qabs, Qsca = read_in_mie_h5(h5_path)
# use mie_radii so that everything is on the same grid
mie_radii_cm = mie_radii * 1e-4  # Convert microns to cm for the distribution function
#print(mie_radii_cm)

N = n_c  # Total number of particles cm^-3
rg = r  # Geometric mean radius in cm
sig = list(all_out['scalar_inputs'].values())[2]  # Geometric standard deviation

distribution = lognormal(N, rg, sig, mie_radii_cm)
k_abs, k_sca = lognormal_abs_sca_sum(mie_radii_cm, distribution, Qabs, Qsca)