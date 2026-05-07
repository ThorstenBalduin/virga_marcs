import numpy as np
import h5py
import matplotlib.pyplot as plt

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

def surface_area_average_radius(r_array, distribution):
    # Calculate the surface-area-weighted average radius
    # r_array is in cm, distribution is in cm^-3cm^-1
    L3_L2 = np.trapezoid(distribution * np.pi * r_array**3, r_array)/np.trapezoid(distribution * np.pi * r_array**2, r_array)  # L3/L2 where Lk = integral of n(r)*pi*r^k dr
    L2_L0 = np.sqrt(np.trapezoid(distribution * np.pi * r_array**2, r_array)/np.trapezoid(distribution, r_array)/np.pi)  # L2/L0 where L0 = integral of n(r) dr, this is the surface-area-weighted average radius
    return L3_L2, L2_L0  

if __name__ == "__main__":
    h5_path = "../Mie_data/Mie_H2O.h5"
    mie_radii, wavelength, Qabs, Qsca = read_in_mie_h5(h5_path)
    # use mie_radii so that everything is on the same grid
    mie_radii_cm = mie_radii * 1e-4  # Convert microns to cm for the distribution function
    #print(mie_radii_cm)
    
    N = 1e6  # Total number of particles cm^-3
    rg = 1e-4  # Geometric mean radius in cm
    sig = 2  # Geometric standard deviation
    
    distribution = lognormal(N, rg, sig, mie_radii_cm)
    moment_avg_radius, surface_avg_radius = surface_area_average_radius(mie_radii_cm, distribution)
    print(f"Surface-area-weighted average radius: {surface_avg_radius*1e4:.6g} microns, Moment-average radius (L3/L2, as reff in VIRGA): {moment_avg_radius*1e4:.6g} microns")
    
    integrated_number = np.trapezoid(distribution, mie_radii_cm)  # Integrate over the radius array to get total number density
    print('Total number of particles (integrated [cm^-3]):', integrated_number)

    plt.plot(mie_radii, distribution)
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Radius (microns)')
    plt.ylabel('Number Density')
    plt.title('Lognormal Size Distribution')
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

    k_abs, k_sca = lognormal_abs_sca_sum(mie_radii_cm, distribution, Qabs, Qsca)
    k_ext = k_abs + k_sca
    
    plt.figure()
    plt.plot(wavelength, k_abs, label="Absorption")
    plt.plot(wavelength, k_sca, label="Scattering")
    plt.plot(wavelength, k_ext, label="Extinction", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Coefficient (cm$^{-1}$)")
    plt.title("Lognormal-integrated Mie Coefficients")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    