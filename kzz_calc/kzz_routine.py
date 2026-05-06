import numpy as np

# physical constants
k_B = 1.381e-16 #erg/K
PROTON_MASS =  1.673e-24 # g

# inputs needed from MARCS run (defined during setup or solved for in this itteration)
P_levs = np.logspace(-6, 2, 100) # pressure levels in bar
T_levs = 1500 * (P_levs/1e-3)**0.1 # temperature profile in K, as a function of pressure
mu = 2.3 # mean molecular weight in amu
g = 1e3 # gravity in cm/s^2 # can compute with Rp and Mp
Teq = 1500 # equilibrium temperature in K
Tint = 500 # internal temperature in K

# Compute Kzz
# Moses et al 2021 Eq 1, converted to cgs units
Teff = (Teq**4 + Tint**4)**(.25)
mbar_in_barye = 1e3 
T_at_1mbar = T_levs[np.argmin(np.abs(P_levs-mbar_in_barye))] # K
H_1mbar = (k_B * T_at_1mbar) / (mu * PROTON_MASS * g) # scale height at 1 mbar, in cm
scaling_factor = (H_1mbar/620e5)*(Teff/1450)**4 
kzz_levs = 5e8 * scaling_factor / np.sqrt((P_levs/1e6)) # Moses et al 2021 Eq 1

# Add a ceiling to Kzz so it doesn't exceed 10^11 cm^2/s,
kzz_levs = np.minimum(kzz_levs, 1e11)