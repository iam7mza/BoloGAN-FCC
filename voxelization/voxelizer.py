import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Geometry information: arXiv:2502.21223 (p5)
# R: 0 - 278 mm vertex detector. no calo hits
# R: 278 - ~2100 mm Drift chamber. only endcap hits
# R: ~2.1 - ~2.4 m EM crystal calorimeter
# R: ~2.4 - ~2.8 m coil. only endcap hits
# R: ~2.8 - 4.6 HCAL
# r_max = 4601 mm
# Equal binning in R; r and alpha depening on hit density
# TODO:must get exact numbers
# TODO: lower number of bins


_PION_BINNING = {
    0:  ((278,  458),  (0, 4601),                                       4),
    1:  ((458,  638),  (0, 4601),                                       4),
    2:  ((638,  818),  (0, 4601),                                       4),
    3:  ((818,  998),  (0, 4601),                                       4),
    4:  ((998,  1178), (0, 4601),                                       4),
    5:  ((1178, 1358), (0, 4601),                                       4),
    6:  ((1358, 1539), (0, 4601),                                       4),
    7:  ((1539, 1719), (0, 4601),                                       4),
    8:  ((1719, 1899), (0, 4601),                                       4),
    9:  ((1899, 2079), (0, 4601),                                       4),
    10: ((2079, 2259), (0, 5, 10, 30, 50, 100, 500, 1000, 2000, 4601), 10),
    11: ((2259, 2439), (0, 5, 10, 30, 50, 100, 500, 1000, 2000, 4601), 10),
    12: ((2439, 2619), (0, 4601),                                       4),
    13: ((2619, 2799), (0, 4601),                                       4),
    14: ((2799, 2979), (0, 5, 10, 30, 50, 100, 500, 1000, 2000, 4601), 10),
    15: ((2979, 3159), (0, 5, 10, 30, 50, 100, 500, 1000, 2000, 4601), 10),
    16: ((3159, 3339), (0, 10, 50, 100, 500, 2000, 4601),              8),
    17: ((3339, 3520), (0, 10, 50, 100, 500, 2000, 4601),              8),
    18: ((3520, 3700), (0, 10, 50, 100, 500, 2000, 4601),              6),
    19: ((3700, 3880), (0, 10, 50, 100, 500, 2000, 4601),              6),
    20: ((3880, 4060), (0, 100, 500, 1500, 4601),                      6),
    21: ((4060, 4240), (0, 100, 500, 1500, 4601),                      6),
    22: ((4240, 4420), (0, 100, 500, 1500, 4601),                      6),
    23: ((4420, 4600), (0, 100, 500, 1500, 4601),                      6),
}

# Placeholder
_ELECTRON_BINNING = _PION_BINNING

BINNING_PER_PID = {
    211: _PION_BINNING,      
    11:  _ELECTRON_BINNING
}


CHI_MAP = {
    "SCEP": 0.41,
    "DR":   0.31,
}


def voxelize(EVENTdf, layerID, binning):
    R_range, r_edges, n_alpha_bins = binning[layerID]

    r_edges     = np.array(r_edges)
    alpha_edges = np.linspace(-np.pi, np.pi, n_alpha_bins + 1)
    n_r_bins    = len(r_edges) - 1
    n_voxels    = n_r_bins * n_alpha_bins

    # hit selection 
    mask = (EVENTdf['R'] >= R_range[0]) & (EVENTdf['R'] < R_range[1])
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


# XML generation 

def _binning_signature(binning: dict) -> list:
    """Return a canonical, hashable representation of a binning dict."""
    return [(lid, params[1], params[2]) for lid, params in sorted(binning.items())]


def generate_binning_xml(
    binning_per_pid: dict,
    pid_meta: dict,
    output_path: str,
) -> None:
    """
    Write a binning XML file.
    """
    
    signatures = {pid: _binning_signature(b) for pid, b in binning_per_pid.items()}
    unique_sigs = {}
    for pid, sig in signatures.items():
        key = str(sig)
        unique_sigs.setdefault(key, []).append(pid)

    root = ET.Element("Bins")

    for pid, binning in sorted(binning_per_pid.items()):
        meta = pid_meta.get(pid, {})

        all_rmins = [params[0][0] for params in binning.values()]
        all_rmaxs = [params[0][1] for params in binning.values()]
        eta_min = meta.get("etaMin", min(all_rmins))
        eta_max = meta.get("etaMax", max(all_rmaxs))

        bin_el = ET.SubElement(root, "Bin",
                               pid=str(pid),
                               etaMin=str(eta_min),
                               etaMax=str(eta_max))

        for layer_id, (r_range, r_edges, n_alpha) in sorted(binning.items()):
            r_edges_str = ",".join(str(v) for v in r_edges)
            ET.SubElement(bin_el, "Layer",
                          id=str(layer_id),
                          r_edges=r_edges_str,
                          n_bin_alpha=str(n_alpha))

    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines = pretty.splitlines()
    pretty_no_header = "\n".join(lines[1:]) 

    with open(output_path, "w") as f:
        f.write(pretty_no_header)


PATH = "/storage-hpc/bologan/alhaddad/DataCSV/"
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


def loadData(ptype: str, energy: int) -> pd.DataFrame:
    df = pd.DataFrame()
    for branch, calib in BRANCHES_CALIBRATION.items():
        branch_df = pd.read_csv(f"{PATH}/{ptype}_{energy}GeV_{branch}.csv")
        branch_df = branch_df[branch_df['Energy'] > 0]
        branch_df['branch'] = branch
        branch_df['type']   = calib[1]
        branch_df['Energy'] = branch_df['Energy'] / calib[0]
        branch_df['R']      = np.sqrt(branch_df['PosX']**2 + branch_df['PosY']**2)
        df = pd.concat([df, branch_df], ignore_index=True)
    return df


PTYPE_CONFIG = {
    #  name      pid   energies
    "pion":     (211, [10, 20, 40, 60, 80]),
    "electron": (11,  [10, 20, 40, 60]),   
}

if __name__ == "__main__":

    PID_META = {
        211: {"etaMin": 0, "etaMax": 80},
        11:  {"etaMin": 0, "etaMax": 60},
        22:  {"etaMin": 0, "etaMax": 80},
    }

    for ptype, (pid, energies) in PTYPE_CONFIG.items():

        Binning = BINNING_PER_PID[pid]

        all_voxels, all_labels = [], []

        for energy in energies:
            print(f"\n{'='*50}")
            print(f"[INFO] Loading data for {ptype} {energy} GeV...")
            df = loadData(ptype, energy)

            for event in range(5000):
                if event % 500 == 0:
                    print(f"[INFO] Processing Event {event}...")

                event_df     = df[df['Event'] == event]
                event_voxels = np.concatenate([
                    voxelize(event_df, layer, Binning) for layer in sorted(Binning)
                ])
                all_voxels.append(event_voxels)
                all_labels.append(energy)

        X = np.stack(all_voxels)
        y = np.array(all_labels)

        print(f"\nVoxelization complete for {ptype}.")
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        
        np.save(f"voxels_{ptype}.npy", X)
        np.save(f"labels_{ptype}.npy", y)
        print(f"[INFO] Saved voxels_{ptype}.npy and labels_{ptype}.npy")

        print(f"\n[INFO] Generating binning XML for {ptype}...")
        generate_binning_xml(
            binning_per_pid={pid: Binning},
            pid_meta=PID_META,
            output_path=f"binning_{ptype}.xml",
        )
        print(f"[INFO] Saved binning_{ptype}.xml")
    print(f"\n[INFO] DONE!")
    print(f"[STATS]: Number of particles processed: {len(PTYPE_CONFIG)};")
    print(f"[STATS]: Number of cigarettes smoked: {np.random.randint(1000, 99999999)}")
    print(f"[STATS]: Number of times I get joyful when the cycle ends; only for it to start \033[31magain\033[0m: 7")