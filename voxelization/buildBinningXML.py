from xml.etree.ElementTree import Element, SubElement, indent, tostring
import pandas as pd
import numpy as np

SUBDETECTORS = ['SCEPCal_MainCcounts', 'SCEPCal_MainScounts', 'DRBTCher',
                'DRBTScin', 'DRETCherLeft', 'DRETScinLeft', 'DRETCherRight', 'DRETScinRight']

PARTICLES = {'pion': 211, 'electron': 11}
ENERGIES = {
    'pion':     [10, 20, 40, 60, 80],
    'electron': [10, 20, 40, 60],
}

PATH = "/storage-hpc/bologan/alhaddad/DataCSV/"

# Configuration dictionary: subdetector -> (n_layers, n_r_bins, n_alpha_bins)
SUBDETECTOR_CONFIG = {
    'SCEPCal_MainCcounts': (3, 10, 10),
    'SCEPCal_MainScounts': (3, 10, 10),
    'DRBTCher':            (3, 10, 10),
    'DRBTScin':            (3, 10, 10),
    'DRETCherLeft':        (2,  1,  1),
    'DRETScinLeft':        (2,  1,  1),
    'DRETCherRight':       (2,  1,  1),
    'DRETScinRight':       (2,  1,  1),
}


def loadBranchDataFrame(branch_name, pType, path=PATH):
    all_data = []
    for energy in ENERGIES[pType]: 
        file_path = f"{path}/{pType}_{energy}GeV_{branch_name}.csv"
        data = pd.read_csv(file_path)
        data = data[data['Energy'] > 0]
        data['R']   = np.sqrt(data['PosX']**2 + data['PosY']**2)
        data['phi'] = np.arctan2(data['PosY'], data['PosX'])
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

def createSubDetectorElement(subdetector_name, dataFrame, n_layers, n_r_bins, n_alpha_bins, layer_idx_start):
    R_min = np.round(dataFrame['R'].min() - 1)
    R_max = np.round(dataFrame['R'].max() + 1)
    subdetector_elem = Element('SubDetector', id=subdetector_name, R_Range=f"{R_min},{R_max}")

    layer_edges = np.round(np.linspace(R_min - 1, R_max + 1, n_layers + 1))
    r = np.sqrt(dataFrame['phi']**2 + dataFrame['PosZ']**2)

    for i in range(n_layers):
        mask  = (dataFrame['R'] >= layer_edges[i]) & (dataFrame['R'] < layer_edges[i + 1])
        r_min = np.maximum(r[mask].min() - 1, 0)
        r_max = r[mask].max() + 1
        r_edges = np.round(np.linspace(r_min, r_max, n_r_bins + 1, endpoint=True))
        SubElement(
            subdetector_elem, 'Layer',
            id=str(layer_idx_start + i),
            R_Range=f"{layer_edges[i]},{layer_edges[i + 1]}",
            r_edges=','.join(map(str, r_edges)),
            n_bin_alpha=str(n_alpha_bins)
        )

    return subdetector_elem, layer_idx_start + n_layers 


XML_FILE = "binning.xml"
root = Element('Bin')

for pType, pid in PARTICLES.items():
    print(f"\n[INFO] Processing particle: {pType} (pid={pid})")
    bin_elem   = SubElement(root, 'Bin', pid=str(pid))
    layer_idx  = 0

    for subdetector_name, (n_layers, n_r_bins, n_alpha_bins) in SUBDETECTOR_CONFIG.items():
        print(f"  [INFO] Loading {subdetector_name}...")
        df = loadBranchDataFrame(subdetector_name, pType)
        subdetector_elem, layer_idx = createSubDetectorElement(
            subdetector_name, df, n_layers, n_r_bins, n_alpha_bins, layer_idx
        )
        bin_elem.append(subdetector_elem)

indent(root)
xml_str = tostring(root, encoding='unicode')
with open(XML_FILE, 'w') as f:
    f.write(xml_str)

print(f"\n[INFO] Done. Written to {XML_FILE}")