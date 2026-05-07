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

k_B = 1.380649e-16               #ergs/K

input_data_marcs='/groups/astro/mhundrup/virga_marcs/models/400K/marcs_output.dat'
mieff_dir =  '/groups/astro/mhundrup/virga_marcs/virga/mieffs/'
GRAV = 2.95

TP = np.genfromtxt(input_data_marcs, skip_header = 93, max_rows= 53)
chf = np.genfromtxt(input_data_marcs, skip_header = 39, max_rows= 53)[:,5]
layer = TP[:,0]
T = TP[:,4]-50                   #K
P = TP[:,6]*1e-6                 #bar
kz = 1e7                         #

K2_18b = pd.DataFrame({'pressure': P, 'temperature': T, 'kz': kz})

metallicity = 10                #atmospheric metallicity relative to Solar
mean_molecular_weight = 2.2     # atmospheric mean molecular weight

molecules = ['H2O']

#set the run
a = jdi.Atmosphere(molecules,
                fsed=0.1,mh=metallicity,
                mmw = mean_molecular_weight)

#set the planet gravity
a.gravity(gravity=GRAV, gravity_unit=u.Unit('m/(s**2)'))

#Get pt profile for testing
#a.ptk(df = jdi.hot_jupiter())
a.ptk(df = K2_18b)

#get full dictionary output
all_out = jdi.compute(a, as_dict=True,
                    directory=mieff_dir)


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
    r[:,i] = all_out['mean_particle_r'][:,i]*1e-4            #convert from microns to cm

    for j in range(len(mmr_c)):                         # r, mmr_c and P, T arent same lengths
        pp[j][i] = (mean_molecular_weight/mass_igas) * P[j]*1e6 * (mmr_tot[j] - mmr_c[j])         #dyn/cm^2
        if r[j][i] > 0:
            n_c[j][i] = (3*mmr_c[j]*mean_molecular_weight* cgs_mass_H *(P[j]*1e6))/(4*np.pi*r[j][i]**3*rho_p*k_B*T[j])   #number density

final_dat = np.vstack([r, pp, n_c])

output_path = '/groups/astro/mhundrup/virga_marcs/virga2marcs.dat'
with open(output_path, 'w') as file:
    file.write("P (bar) r (cm) pp (dyn/cm^2) n_c (1/cm^3)\n")
    for j in range(len(molecules)):
        file.write(f"\nGas: {molecules[j]}\n")
        for i in range(len(P)):
            if i == len(P)-1:
                file.write(f"{P[i]:.6e} {0} {0} {0}\n")
            else:
                file.write(f"{P[i]:.6e} {r[i][j]:.6e} {pp[i][j]:.6e} {n_c[i][j]:.6e}\n")