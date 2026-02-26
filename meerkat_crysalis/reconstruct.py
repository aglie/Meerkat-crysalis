#!/usr/bin/env python3
import numpy as np
import argparse
import os
from glob import glob
from sys import stdout



def extract_unit_cell(log_file):
    """
    Parses unit cell parameters from CrysAlis FRM2XYZ log file.
    Returns: np.array([a, b, c, alpha, beta, gamma])
    """
    with open(log_file, 'r') as f:
        content = f.read()

    # Find the block starting with "unit cell:"
    # This regex looks for the header and grabs the next two lines of numbers
    pattern = r"unit cell:\s*\n\s*([\d\.\(\)]+\s+[\d\.\(\)]+\s+[\d\.\(\)]+)\s*\n\s*([\d\.\(\)]+\s+[\d\.\(\)]+\s+[\d\.\(\)]+)"
    match = re.search(pattern, content)

    if not match:
        raise ValueError("Could not find 'unit cell:' block in log file.")

    # Combine the two lines into one string of 6 values
    raw_values = match.group(1) + " " + match.group(2)
    
    # Remove error bars: e.g., '26.1803(6)' -> '26.1803'
    clean_values = re.sub(r'\(\d+\)', '', raw_values)
    
    # Convert to float array
    cell = np.array([float(x) for x in clean_values.split()])
    
    return cell

def get_esperanto_dims(fname):
    """Extracts (width, height) from the 9216-byte Esperanto header."""
    HEADER_SIZE = 36 * 256
    with open(fname, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
    header_text = header_bytes.decode("ascii", errors="ignore")
    for line in header_text.splitlines():
        if line.startswith("IMAGE"):
            parts = line.split()
            return int(parts[1]), int(parts[2])
    raise ValueError(f"Could not find IMAGE dimensions in {fname}")

def read_dfhkl(fname):
    """Reads .dfhkl file. Returns status, H, K, L, and Intensity."""
    w, h = get_esperanto_dims(fname)
    raw = np.fromfile(fname, dtype=np.uint8, offset=36*256)
    f_data = raw.view(np.float32).reshape(h, w, 5)
    i_data = raw.view(np.int32).reshape(h, w, 5)
    return i_data[..., 0], f_data[..., 1], f_data[..., 2], f_data[..., 3], i_data[..., 4]

def read_corrections(fname):
    """Reads .fhkl_lp correction files."""
    w, h = get_esperanto_dims(fname)
    raw = np.fromfile(fname, dtype=np.float32, offset=36*256)
    data = raw.reshape(h, w, -1)
    # Return Polarization (p) and Solid Angle (sa)
    return data[..., 1], data[..., 4]

def main():
    parser = argparse.ArgumentParser(
        prog="meerkat-crysalis",
        description="Reconstruct CrysAlis hklf frames by run.",
    )
    parser.add_argument("folder", help="Path to the folder containing .fhkl and .fhkl_lp files")
    parser.add_argument("--out", default="reconstructed_volume.npy", help="Output filename")
    parser.add_argument("--size", type=int, nargs=3, default=[601, 601, 601], help="Grid size: H K L")
    parser.add_argument("--min", type=float, nargs=3, default=[-10.0, -10.0, -10.0], help="Min indices: H K L")

    args = parser.parse_args()

    # Setup Grid
    final_size = np.array(args.size)
    min_ind = np.array(args.min)
    max_ind = -min_ind
    step_sizes = (max_ind - min_ind) / (final_size - 1)
    inv_step_sizes = 1.0 / step_sizes


    # Grouping Logic
    lp_files = sorted(glob(os.path.join(args.folder, '*.fhkl_lp')))
    if not lp_files:
        print(f"Error: No .fhkl_lp files found in {args.folder}")
        return

    volume_data = np.zeros(final_size, dtype=np.float32)
    volume_counts = np.zeros(final_size, dtype=np.uint32)

    for lp_path in lp_files:
        # Determine the run prefix (e.g., '100K_a_1')
        run_prefix = lp_path.replace('.fhkl_lp', '')
        # Match frames belonging to this run (e.g., '100K_a_1_*.fhkl')
        run_frames = sorted(glob(run_prefix + '_*.fhkl'))

        if not run_frames:
            print(f"Warning: No frames found for run {run_prefix}")
            continue

        print(f"\nProcessing run: {os.path.basename(run_prefix)} ({len(run_frames)} frames)")

        # Load corrections specific to this run
        p, sa = read_corrections(lp_path)
        corr_map = (p * (sa + 0.75)**1.5)

        for i, fname in enumerate(run_frames):
            status, h_arr, k_arr, l_arr, i_arr = read_dfhkl(fname)

            mask = (status == 0) & (i_arr > 0)

            # Map coordinates to grid indices
            idx_h = ((h_arr[mask] - min_ind[0]) * inv_step_sizes[0] + 0.5).astype(np.int32)
            idx_k = ((k_arr[mask] - min_ind[1]) * inv_step_sizes[1] + 0.5).astype(np.int32)
            idx_l = ((l_arr[mask] - min_ind[2]) * inv_step_sizes[2] + 0.5).astype(np.int32)
            iv_ints = i_arr[mask].astype(np.float32) * corr_map[mask]

            # Bounds Checking
            in_bounds = (idx_h >= 0) & (idx_h < final_size[0]) & \
                        (idx_k >= 0) & (idx_k < final_size[1]) & \
                        (idx_l >= 0) & (idx_l < final_size[2])

            # Aggregate
            np.add.at(volume_data, (idx_h[in_bounds], idx_k[in_bounds], idx_l[in_bounds]), iv_ints[in_bounds])
            np.add.at(volume_counts, (idx_h[in_bounds], idx_k[in_bounds], idx_l[in_bounds]), 1)

            if i % 10 == 0:
                stdout.write(f"\r  Frame {i}/{len(run_frames)}")
                stdout.flush()

    print("\nNormalizing volume...")
    #TODO: fix this. Different runs should be noramlized separately.
    with np.errstate(divide='ignore', invalid='ignore'):
        reconstructed_volume = volume_data / volume_counts
        # we do not clean up NaNs, as they mark unreconstructed voxels.

    log_file = glob(os.path.join(args.folder, 'frm2xyzlog.txt'))[0]
    cell = extract_unit_cell(log_file)

    # Save to Yell format
    with h5py.File('reconstruction.h5', 'w') as result:
        result['data']=reconstructed_volume
        result['format']=b"Yell 1.0"
        result['unit_cell'] = cell
        result["step_sizes"] = step_sizes
        result["lower_limits"] = min_ind
        result['is_direct'] = False
    print(f"Success! Volume saved to {args.out}")

if __name__ == "__main__":
    main()
