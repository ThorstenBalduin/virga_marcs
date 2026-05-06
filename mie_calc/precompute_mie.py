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

def interpolate_refractive_index(wavelengths_data, real_part, imag_part, wavelengths_interp):
    """
    Interpolate refractive index as separate real/imaginary float arrays.

    Parameters:
        wavelengths (array): Original wavelength grid (microns).
        real_part (array): Real part of the complex refractive index corresponding to the original wavelengths.
        imag_part (array): Imaginary part of the complex refractive index corresponding to the original
        wl (array): Target wavelength grid (microns).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            real_part_interp, imag_part_interp
            (both real-valued float arrays)
    """

    real_interp_fn = interp1d(
        wavelengths_data, real_part, kind='linear',
        bounds_error=False, fill_value="extrapolate"
    )
    imag_interp_fn = interp1d(
        wavelengths_data, imag_part, kind='linear',
        bounds_error=False, fill_value="extrapolate"
    )

    real_part_interp = real_interp_fn(wavelengths_interp)
    imag_part_interp = imag_interp_fn(wavelengths_interp)

    return real_part_interp, imag_part_interp

def mie_calc(args):
    
    size, wavelength, n_grid, k_grid = args
    
    # Define x_array and geometric cross-section
    x_array = 2 * np.pi * size / wavelength # both in microns
    
    # Define the complex refractive index array
    complex_refractive_indexs = n_grid - 1j * k_grid

    # Calculate the extinction cross-section using Mie theory
    qext, qsca, qback, g = mie.efficiencies_mx(complex_refractive_indexs, x_array)  # Ignoring qback as it's unused
    
    qabs = qext - qsca  # Absorption efficiency, wanted for MARCS
    
    return qabs, qsca # floats for given wavelength and size


if __name__ == '__main__':
    ####################################################
    # Control values
    folder_path = 'Refractive_indices'
    material = 'H2O'
    
    #TODO correctly with MARCS grid
    wl_path = '../data/marcs_wavelengths_um.dat' #if this is set, the params below are ignored
    wl_min = 0.125  # in microns
    wl_max = 25.0  # in microns
    R = 1000  # Resolving power
    
    quick_plot = True
    
    # Define the size bins for the Mie calculations
    sizes = np.logspace(-3, 1, 1000)  # in microns
    
    ####################################################
    # Read in the refractive index
    
    # load all the files in this folder
    print(f"Reading refractive index files from {folder_path}...")
    ref_ind_data_dict = read_refractive_index_files(folder_path)
    #print(f"Refractive index data keys: {list(ref_ind_data_dict.keys())}")
    
    # pick a material and load its data
    material_ref_ind_data = ref_ind_data_dict.get(material)
    
    if material_ref_ind_data is None:
        print(f"Material '{material}' not found. Available: {list(ref_ind_data_dict.keys())}")
        quit()
    else:
        wavelengths_data = material_ref_ind_data['Wavelength (um)']
        n_data = material_ref_ind_data['n']
        k_data = material_ref_ind_data['k']
        complex_ref_ind = n_data - 1j * k_data
        print(f"{material} wavelength range (um): min={np.min(wavelengths_data):.6g}, max={np.max(wavelengths_data):.6g}")
        print(f"{material} n range: min={np.min(n_data):.6g}, max={np.max(n_data):.6g}")
        print(f"{material} k range: min={np.min(k_data):.6g}, max={np.max(k_data):.6g}")
    
    ####################################################
    # Set up the wavelength grid for the Mie calculations
    if wl_path is not None:
        print(f"Loading wavelength grid from {wl_path}...")
        wavelengths_grid = np.loadtxt(wl_path)  # in microns
    else:
        print(f"Creating wavelength grid from {wl_min} to {wl_max} microns with R={R}...")
        wavelengths_grid = np.logspace(np.log10(wl_min), np.log10(wl_max), int((wl_max - wl_min) * R))  # in microns
    
    print(f"Wavelength grid shape: {wavelengths_grid.shape}")
    print(f"Wavelength grid: {wavelengths_grid} microns")
    
    #################################################
    # Interpolate the refractive index to the wavelength grid
    # Extract the wavelength and refractive index values

    # Interpolate the complex refractive index to the wl grid
    print('Interpolating refractive index...')
    n_grid, k_grid = interpolate_refractive_index(wavelengths_data, n_data, k_data, wavelengths_grid)
    print('Interpolation complete.')
    
    if quick_plot:
        fig, (ax_n, ax_k) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Top panel: n
        ax_n.scatter(wavelengths_data, n_data, s=10, alpha=0.7, label="original n")
        ax_n.plot(wavelengths_grid, n_grid, lw=1.5, label="interpolated n")
        ax_n.set_ylabel("n")
        ax_n.legend()

        # Bottom panel: k
        ax_k.scatter(wavelengths_data, k_data, s=10, alpha=0.7, label="original k")
        ax_k.plot(wavelengths_grid, k_grid, lw=1.5, label="interpolated k")
        ax_k.set_xscale("log")
        ax_k.set_xlabel("Wavelength (micron)")
        ax_k.set_ylabel("k")
        ax_k.legend()
        ax_k.set_yscale("log")

        plt.tight_layout()
        plt.show()
    
    # Create argument list for parallel processing
    args_list = [(size, wavelength, n_grid, k_grid) for size in sizes for wavelength in wavelengths_grid]

    # Run Mie calculations in parallel
    print("Running Mie calculations in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(mie_calc, args_list))

    # Save results to file
    output_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'Mie_data',
        f"Mie_{material}.txt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving Mie data to {output_path}")
    with open(output_path, "w") as f:
        f.write("r [um]\tlambda [um]\tqabs\tqsca\n")
        
        for i, size in enumerate(sizes):
            qabs, qsca = results[i]
            for j, wl in enumerate(wavelengths_grid):
                f.write(f"{size:.8e}\t{wl:.8e}\t{qabs[j]:.8e}\t{qsca[j]:.8e}\n")

    print("Mie calculations complete.")