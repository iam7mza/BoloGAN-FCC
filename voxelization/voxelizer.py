import numpy as np
import pandas as pd


CHI_MAP = {
    "SCEP": 0.41,
    "DR":   0.31,
}


def voxelize(EVENTdf, layerID):
    R_range, r_edges, n_alpha_bins = Binning[layerID]

    r_edges     = np.array(r_edges)
    alpha_edges = np.linspace(-np.pi, np.pi, n_alpha_bins + 1)
    n_r_bins    = len(r_edges) - 1
    n_voxels    = n_r_bins * n_alpha_bins

    # hit selection 
    mask = (EVENTdf['R'] >= R_range[0]) & (EVENTdf['R'] <= R_range[1])
    hits = EVENTdf[mask]

    if hits.empty:
        return np.zeros(n_voxels)

    # per-hit chi assignment
    branches   = hits['branch'].to_numpy()
    is_DR   = hits['branch'].str.contains("DR").to_numpy()
    is_SCEP = hits['branch'].str.contains("SCEP").to_numpy()
    chi_values = np.where(is_DR, 0.31, np.where(is_SCEP, 0.41, 0.0))

    type_mask = (hits['type'] == 'Scintillation').to_numpy() 

    # coordinate transform 
    phi        = np.arctan2(hits['PosY'].to_numpy(), hits['PosX'].to_numpy())
    z          = hits['PosZ'].to_numpy()
    hits_r     = np.sqrt(z**2 + phi**2)
    hits_alpha = np.arctan2(phi, z)
    energy     = hits['Energy'].to_numpy()

    # dual-readout calibration 
    # NOTE: can produce negative voxel values; expected and acceptable globally
    energy_scin = np.where(type_mask,  energy, 0.0)
    energy_cher = np.where(~type_mask, energy, 0.0)
    energy_cal  = (energy_scin - chi_values * energy_cher) / (1 - chi_values)

    # bin assignment 
    r_idx     = np.searchsorted(r_edges[1:-1],     hits_r)      
    alpha_idx = np.searchsorted(alpha_edges[1:-1], hits_alpha)
    flat_idx  = r_idx * n_alpha_bins + alpha_idx

    voxels = np.zeros(n_voxels)
    np.add.at(voxels, flat_idx, energy_cal)


    # Debugging
    # print(f"Energy: {energy_cal.sum()}")
    # print(f"voxel Energy: {voxels.sum()}")

    return voxels


PATH = "/storage-hpc/bologan/alhaddad/DataCSV/"
PTYPE = "pion"
BRANCHES_CALIBRATION = {
        'SCEPCal_MainScounts': (1960.92, 'Scintillation'),
        'SCEPCal_MainCcounts': (97.531, 'Cherenkov'),
        'DRBTCher':            (68.069, 'Cherenkov'),     
        'DRBTScin':            (206.284, 'Scintillation'),    
        'DRETCherRight':       (68.069, 'Cherenkov'),     
        'DRETCherLeft':        (68.069, 'Cherenkov'),     
        'DRETScinLeft':        (206.284, 'Scintillation'),      
        'DRETScinRight':       (206.284, 'Scintillation'), 
    }


def loadData(Energy):
    df =pd.DataFrame()
    for branch, calib in BRANCHES_CALIBRATION.items():
        branch_df = pd.read_csv(f"{PATH}/{PTYPE}_{Energy}GeV_{branch}.csv")
        branch_df = branch_df[branch_df['Energy'] > 0] # filter out zero-energy hits
        branch_df['branch'] = branch
        branch_df['type'] = calib[1]
        branch_df['Energy'] = branch_df['Energy'] / calib[0] # apply calibration
        branch_df['R'] = np.sqrt(branch_df['PosX']**2 + branch_df['PosY']**2)
        df = pd.concat([df, branch_df], ignore_index=True)

    return df


# TODO: proper binning
Binning = {
    # layerID: ((Rmin, Rmax), r_edges, n_alpha_bins)
    0: ((460,  2376), (0, 5, 10, np.inf), 5),
    1: ((2804, 4587), (0, 5, 10, np.inf), 5),
    2: ((279,  459.999), (0, np.inf), 1),
}

if __name__ == "__main__":
    
    Energies = [10, 20, 40, 60, 80]

    all_voxels = []
    all_labels = []

    for Energy in Energies:
        print(f"\n{'='*50}")
        print(f"[INFO] Loading data for {PTYPE} {Energy} GeV...")
        df = loadData(Energy)

        for Event in range(5000):
            if Event % 500 == 0:
                print(f"[INFO] Processing Event {Event}...")

            Eventdf = df[df['Event'] == Event]

            event_voxels = []
            for layer, bin_params in Binning.items():
                voxels = voxelize(Eventdf, layer)
                event_voxels.append(voxels)

            event_voxels = np.concatenate(event_voxels)

            all_voxels.append(event_voxels)
            all_labels.append(Energy) 


    X = np.stack(all_voxels)
    y = np.array(all_labels)

    print("Voxelization complete.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Sample X:", X[0][:10])
    print("Sample y:", y[:10])

np.save(f"voxels_{PTYPE}.npy", X)
np.save(f"labels_{PTYPE}.npy", y)