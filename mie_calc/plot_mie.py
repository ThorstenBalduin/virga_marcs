import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def load_mie_h5(file_path):
	"""Load precomputed Mie data from HDF5 file."""
	with h5py.File(file_path, "r") as f:
		sizes_um = f["sizes_um"][:]
		wavelengths_um = f["wavelengths_um"][:]
		qabs = f["qabs"][:]
		qsca = f["qsca"][:]

	return sizes_um, wavelengths_um, qabs, qsca


def make_plot(sizes_um, wavelengths_um, qabs, qsca, initial_index=0):
	"""Create interactive plot with slider over particle size index."""
	if len(sizes_um) == 0:
		raise ValueError("No size bins found in input file.")

	if initial_index < 0 or initial_index >= len(sizes_um):
		raise ValueError(
			f"initial_index={initial_index} is out of bounds for {len(sizes_um)} sizes."
		)

	qext = qabs + qsca
	tiny = np.finfo(float).tiny

	fig, ax = plt.subplots(figsize=(10, 6))
	plt.subplots_adjust(bottom=0.22)

	line_qabs, = ax.plot(
		wavelengths_um,
		qabs[initial_index],
		label="qabs",
		linewidth=2,
	)
	line_qsca, = ax.plot(
		wavelengths_um,
		qsca[initial_index],
		label="qsca",
		linewidth=2,
	)
	line_qext, = ax.plot(
		wavelengths_um,
		qext[initial_index],
		label="qext = qabs + qsca",
		linewidth=2,
		linestyle="--",
	)

	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel("Wavelength (micron)")
	ax.set_ylabel("Efficiency")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend()

	title = ax.set_title(
		f"Mie Efficiencies, size = {sizes_um[initial_index]:.4g} micron "
		f"(index {initial_index})"
	)

	slider_ax = fig.add_axes([0.12, 0.08, 0.76, 0.04])
	size_slider = Slider(
		ax=slider_ax,
		label="Size index",
		valmin=0,
		valmax=len(sizes_um) - 1,
		valinit=initial_index,
		valstep=1,
	)

	def update(_):
		idx = int(size_slider.val)
		y_qabs = np.maximum(qabs[idx], tiny)
		y_qsca = np.maximum(qsca[idx], tiny)
		y_qext = np.maximum(qext[idx], tiny)

		line_qabs.set_ydata(y_qabs)
		line_qsca.set_ydata(y_qsca)
		line_qext.set_ydata(y_qext)

		y_max = np.max([y_qabs.max(), y_qsca.max(), y_qext.max()])
		#y_min = max(1e-3 * y_max, tiny)
		y_min = min(y_qabs.min(), y_qsca.min(), y_qext.min())
		if y_max <= y_min:
			y_max = y_min * 10.0
		ax.set_ylim(y_min, y_max)

		title.set_text(
			f"Mie Efficiencies, size = {sizes_um[idx]:.4g} micron (index {idx})"
		)
		fig.canvas.draw_idle()

	size_slider.on_changed(update)
	update(None)
	plt.show()


def parse_args():
	parser = argparse.ArgumentParser(
		description="Plot qabs, qsca, and qext from precomputed Mie HDF5 data."
	)
	default_h5 = os.path.join(
		os.path.dirname(__file__),
		"..",
		"Mie_data",
		"Mie_H2O.h5",
	)
	parser.add_argument(
		"--h5",
		default=default_h5,
		help="Path to HDF5 file created by precompute_mie.py",
	)
	parser.add_argument(
		"--index",
		type=int,
		default=0,
		help="Initial size index for plotting",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	h5_path = os.path.abspath(args.h5)

	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

	sizes_um, wavelengths_um, qabs, qsca = load_mie_h5(h5_path)
	make_plot(sizes_um, wavelengths_um, qabs, qsca, initial_index=args.index)


if __name__ == "__main__":
	main()
