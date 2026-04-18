import numpy as np
import h5py

PTYPE_CONFIG = {
    "pion":     [10, 20, 40, 60, 80],
    "electron": [10, 20, 40, 60],
}

for ptype, energies in PTYPE_CONFIG.items():
    X = np.load(f"voxels_{ptype}.npy")  # (num_showers, num_voxels)
    y = np.load(f"labels_{ptype}.npy")  # (num_showers,)

    # convert GeV to MeV
    incident_energies = (y * 1000).astype(np.float32).reshape(-1, 1)
    showers = X.astype(np.float32) *1000

    with h5py.File(f"dataset_{ptype}.hdf5", "w") as f:
        f.create_dataset("incident_energies", data=incident_energies)
        f.create_dataset("showers",           data=showers)

    print(f"[INFO] Saved dataset_{ptype}.hdf5 — showers: {showers.shape}, energies: {incident_energies.shape}")
