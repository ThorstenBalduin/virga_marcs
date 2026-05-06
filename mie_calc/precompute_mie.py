import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import concurrent.futures

#Try to set up jit for miepython to get tha speed up
os.environ["MIEPYTHON_USE_JIT"] = "1"  # Set to "0" to disable JIT
import miepython as mie
print('Using Jit for Mie python',os.environ.get("MIEPYTHON_USE_JIT"))

# Set POSEIDON input data paths relative to this file's location
POSEIDON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CApyBaRA2', 'src', 
                                             'CApyBaRA2', 'submodules', 'POSEIDON', 'inputs'))
os.environ['POSEIDON_input_data'] = POSEIDON_ROOT
os.environ['PYSYN_CDBS'] = os.path.join(POSEIDON_ROOT, 'stellar_grids')

def read_refractive_index_files(folder_path):
    full_data = {}
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

    # Files have inconsistent column names,
    # so defining them here
    names=['Wavelength (um)', 'n', 'k']

    for file_path in file_paths:
        file = os.path.relpath(file_path, folder_path)
        file = file.split('.')[0]
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Filter out lines that start with '#'
            data_lines = [line for line in lines if not line.strip().startswith('#')]
            if len(data_lines) < 1:
                continue  # Skip files with insufficient data

            # Assume the data has 3 columns separated by '\t'
            data = np.genfromtxt(data_lines)
            # Transpose the data to have columns as [wavelength, n, k]
            data = data.T
            # Data might not be sorted by wavelength
            data = data[:, data[0, :].argsort()]

            data_dict = dict(zip(names, data))
            full_data[file] = data_dict
            
    return full_data

def interpolate_refractive_index(wavelengths, complex_refractive_indexs, wl):
    """
    Interpolates the complex refractive index from the given wavelengths grid to the target wl grid.

    Parameters:
        wavelengths (array): Original wavelength grid (in microns).
        complex_refractive_indexs (array): Complex refractive index values corresponding to the original grid.
        wl (array): Target wavelength grid (in microns).

    Returns:
        array: Interpolated complex refractive index values on the target grid.
    """
    real_interp = interp1d(wavelengths, complex_refractive_indexs.real, kind='linear', bounds_error=False, fill_value="extrapolate")
    imag_interp = interp1d(wavelengths, complex_refractive_indexs.imag, kind='linear', bounds_error=False, fill_value="extrapolate")
    return real_interp(wl) - 1j * imag_interp(wl)

def bin_size_mie_calc(size,complex_refractive_indexs, wavelengths_grid,mie_routine):
    # Size is in microns!!!

    # Define x_array and geometric cross-section
    x_array = 2 * np.pi * size / wavelengths_grid # both in microns
    geometric_cross_section = np.pi * (size * 1e-6) ** 2  # Cross section area in m^2

    # Calculate the extinction cross-section using Mie theory
    qext, qsca, qback, g = mie.efficiencies_mx(complex_refractive_indexs, x_array)  # Ignoring qback as it's unused
    # Convert to cross section (Cext or sigma ext depending on the convention)
    extinction_cross_section = qext * geometric_cross_section

    # Compute w (SSA)
    w = qsca / qext

    return extinction_cross_section, w, g

def run_mie(refractive_index_data,sizes,wavelengths_grid):
    # Extract the wavelength and refractive index values
    wavelengths = refractive_index_data['Wavelength (um)']  # in microns
    n_values = refractive_index_data['n']  # Real part of refractive index
    k_values = refractive_index_data['k']  # Imaginary part of refractive index

    # Convert n and k to complex refractive index
    complex_refractive_indexs = n_values - 1j * k_values  # miepython and POSEIDON use -ve k values convention

    # Interpolate the complex refractive index to the wl grid
    print('Interpolating refractive index...')
    complex_refractive_indexs = interpolate_refractive_index(wavelengths, complex_refractive_indexs, wavelengths_grid)
    print('Interpolation complete.')

    # Process each bin size sequentially
    kappa_ext_array = np.zeros((len(sizes), len(wavelengths_grid)))
    w_array = np.zeros((len(sizes), len(wavelengths_grid)))
    g_array = np.zeros((len(sizes), len(wavelengths_grid)))

    for b, size in enumerate(sizes):

        extinction_cross_section, w, g = bin_size_mie_calc(size, complex_refractive_indexs, wavelengths_grid, mie_routine)

        kappa_ext_array[b, :] = extinction_cross_section
        w_array[b, :] = w
        g_array[b, :] = g

        # Print progress for size bins
        print(f"\rProcessing size bins: {b + 1}/{len(sizes)} complete", end='')

    return {
        'wavelengths_grid': wavelengths_grid,
        'size_bins': sizes, # shape (nbin,nwave)
        'kappa_ext': kappa_ext_array, #TODO: in future name this sigma_ext
        'w': w_array,
        'g': g_array
    }


if __name__ == '__main__':
    ####################################################
    # Read in the refractive index
    
    # load all the files in this folder
    folder_path = os.path.join('Refractive_indices')
    print(f"Reading refractive index files from {folder_path}...")
    ref_ind_data_dict = read_refractive_index_files(folder_path)
    #print(f"Refractive index data keys: {list(ref_ind_data_dict.keys())}")
    
    # pick a material and load its data
    material = 'H2O'
    material_ref_ind_data = ref_ind_data_dict.get(material)
    
    if material_ref_ind_data is None:
        print(f"Material '{material}' not found. Available: {list(ref_ind_data_dict.keys())}")
        quit()
    else:
        wl_data = material_ref_ind_data['Wavelength (um)']
        n_data = material_ref_ind_data['n']
        k_data = material_ref_ind_data['k']
        complex_ref_ind = n_data - 1j * k_data
        print(f"{material} wavelength range (um): min={np.min(wl_data):.6g}, max={np.max(wl_data):.6g}")
        print(f"{material} n range: min={np.min(n_data):.6g}, max={np.max(n_data):.6g}")
        print(f"{material} k range: min={np.min(k_data):.6g}, max={np.max(k_data):.6g}")
    
    ####################################################
    # Set up the wavelength grid for the Mie calculations
        
    #TODO correctly with MARCS grid
    wl_min = 0.125  # in microns
    wl_max = 25.0  # in microns
    R = 1000  # Resolving power
    
    wavelengths_grid = np.logspace(np.log10(wl_min), np.log10(wl_max), int((wl_max - wl_min) * R))  # in microns
    
    print(f"Wavelength grid shape: {wavelengths_grid.shape}")
    print(f"Wavelength grid: {wavelengths_grid} microns")
    
    #################################################
    # Interpolate the refractive index to the wavelength grid
    complex_ref_ind_interp = interpolate_refractive_index(wl_data, complex_ref_ind, wavelengths_grid)

    # Interpolated n and k on the target wavelength grid
    n_interp = np.real(complex_ref_ind_interp)
    k_interp = np.imag(complex_ref_ind_interp)

    fig, (ax_n, ax_k) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Top panel: n
    ax_n.scatter(wl_data, n_data, s=10, alpha=0.7, label="original n")
    ax_n.plot(wavelengths_grid, n_interp, lw=1.5, label="interpolated n")
    ax_n.set_ylabel("n")
    ax_n.legend()

    # Bottom panel: k
    ax_k.scatter(wl_data, k_data, s=10, alpha=0.7, label="original k")
    ax_k.plot(wavelengths_grid, k_interp, lw=1.5, label="interpolated k")
    ax_k.set_xscale("log")
    ax_k.set_xlabel("Wavelength (micron)")
    ax_k.set_ylabel("k")
    ax_k.legend()
    ax_k.set_yscale("log")

    plt.tight_layout()
    plt.show()
    
    quit()
    
    #################################################
    # Define the size bins for the Mie calculations
    size_bins = np.logspace(-3, 1, 1000)  # in microns
    
    #################################################
    # Do the Mie calculations for each size bin and save the results in an HDF5 file
'''

    # Note to me later: groups can share a material but still need separate Mie runs (different bin sizes)
    def mie_calc_for_material(args):
            wavelengths_grid, complex_refractive_indexs = args
            
            
            
            return group_name, mie_result

        # Prepare arguments for each group
        args_list = [
            (group_name, group_name_list, h5_path, ref_ind_name_dict, ref_ind_data_dict, wavelengths_grid, mie_routine)
            for group_name in group_name_list
        ]

        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(mie_calc_for_material, args_list))

        for group_name, mie_result in results:
            Mie_data[group_name] = mie_result

    # Save to a single TXT file
    output_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'Mie_data',
        f"mie_data_{mie_routine}_wl_{wl_min}_{wl_max}_{R}.txt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving Mie data to {output_path}")

    with open(output_path, "w") as f:
        f.write("# Mie data output\n")
        f.write("# wavelength_unit: micron\n")
        f.write("# bin_size_unit: micron\n")
        f.write("# kappa_ext_unit: m^2\n")
        f.write("# w_unit: -\n")
        f.write("# g_unit: -\n")
        f.write("#\n")
        f.write("# group_name ref_index_name\n")
        for group_name in group_name_list:
            f.write(f"# {group_name} {ref_ind_name_dict[group_name]}\n")

        for group_name in group_name_list:
            data = Mie_data[group_name]
            wl = data["wavelengths_grid"]
            size_bins = data["size_bins"]

            f.write("\n")
            f.write(f"# --- GROUP: {group_name} ---\n")
            f.write("# columns: size_bin_micron wavelength_micron kappa_ext w g\n")

            for i, size in enumerate(size_bins):
                out = np.column_stack([
                    np.full(wl.shape, size, dtype=float),
                    wl,
                    data["kappa_ext"][i, :],
                    data["w"][i, :],
                    data["g"][i, :]
                ])
                np.savetxt(f, out, fmt="%.8e")
                f.write("\n")

    print(f"Saved data to {output_path}")'''