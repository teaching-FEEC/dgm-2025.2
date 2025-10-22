import os
import scipy.io as sio
import numpy as np

def reshape_mat_files_to_channels_last(directory: str):
    """
    Reshape .mat files in a directory from (C, H, W) to (H, W, C),
    assuming the number of channels is 16.
    """
    for filename in os.listdir(directory):
        if not filename.endswith(".mat"):
            continue

        path = os.path.join(directory, filename)
        print(f"Processing: {filename}")

        # Load the .mat file
        mat_data = sio.loadmat(path)
        keys = [k for k in mat_data.keys() if not k.startswith("__")]

        if not keys:
            print(f"  ⚠️ No valid data variables found in {filename}. Skipping.")
            continue

        for key in keys:
            array = mat_data[key]
            # Expect (C, H, W) and want (H, W, C)
            if isinstance(array, np.ndarray) and array.ndim == 3 and array.shape[0] == 16:
                mat_data[key] = np.transpose(array, (1, 2, 0))  # (C, H, W) → (H, W, C)
                print(f"  Reshaped {key}: {array.shape} → {mat_data[key].shape}")
            else:
                print(f"  Skipping {key}: shape {array.shape}")

        # Overwrite the file
        sio.savemat(path, mat_data)
        print(f"  ✅ Saved {filename}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reshape .mat files from (C, H, W) to (H, W, C)"
    )
    parser.add_argument("directory", help="Path to the directory containing .mat files")
    args = parser.parse_args()

    reshape_mat_files_to_channels_last(args.directory)