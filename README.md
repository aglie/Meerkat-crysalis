# Reciprocal Space Reconstruction from CrysAlis Pro

This script reconstructs a 3D reciprocal space volume from CrysAlis Pro frm2hkl exports. The resulting volume is voxelized and compatible with programs like Yell.

## Installation

```bash
git clone https://github.com/asimonov/Meerkat-crysalis.git
cd Meerkat-crysalis
pip install -e .
```

This installs the `meerkat-crysalis` command into your Python environment.
The only non-standard dependency is [NumPy](https://numpy.org/).

## Step 1: Exporting Data from CrysAlis Pro

To use this script, you must first export your experiment pixels into instrument $hkl$ space using the frm2hkl feature.

- **Version Requirement:** Ensure you are using CAP 44 (at least version 44.124a).
- **Open Export Dialog:** Click on the Import/Export power tool button and select hkl information format.
- **Select Format:** Choose the `*.fhkl` or `*.fhkl_abs` format.
  - `*.fhkl` contains the status byte, $h, k, l$ coordinates, and intensity.
  - `*.fhkl_abs` includes an additional pixel-based absorption correction.
- **Run Corrections:** Ensure the export generates the `*.fhkl_lp` files. CrysAlis produces one LP file per run containing pixel-based corrections (Solid angle, polarization, etc.).

## Step 2: Running the Reconstruction

The script processes the exported frames and applies the run-specific corrections found in the LP files to create a normalized 3D grid.

**Command:**

```bash
meerkat-crysalis /path/to/export/folder --out reconstructed_volume.npy --size 601 601 601 --min -10.0 -10.0 -10.0
```

**Arguments:**

- `folder`: Path to the directory containing your `.fhkl` and `.fhkl_lp` files.
- `--out`: Name of the output NumPy file (default: `reconstructed_volume.npy`).
- `--size`: Grid dimensions for H, K, and L.
- `--min`: Minimum HKL indices for the grid boundaries. Maximum indices are calculated automatically to make a symmetric map.

## Result

Resulting reconstruction can be used for further processing in software like Meerkat2-average. After averaging and Bragg peak removal, it can be used for the ∆PDF refinement in program Yell. 

## Important Notes

- **Data Size:** Be aware that $hkl$ exports are a "massive data explosion," typically 25–30 times larger than the original compressed frames.

## TODO

- [ ] Apply absorption correction from the fhkl_abs files.
- [ ] Implement micro stepping for coarse experiments.

## Authors

- Arkadiy Simonov
- Ella Schmidt
