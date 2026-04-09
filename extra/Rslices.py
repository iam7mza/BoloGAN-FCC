import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

PATH = "/storage-hpc/bologan/alhaddad/DataCSV/"

LAYER_MAP = {
    'SCEPCal_MainCcounts' : 0,
    'SCEPCal_MainScounts' : 1,
    'DRBTCher'            : 2,
    'DRBTScin'            : 3,
    'DRETCherLeft'        : 4,
    'DRETScinLeft'        : 5,
    'DRETCherRight'       : 6,
    'DRETScinRight'       : 7,
}

# CALEBRATION FACTORS p.e./GeV
# BARREL
CALIBRATION_BARREL_S = 206.284
CALIBRATION_BARREL_C = 68.069
CALIBRATION_BARREL_Chi = 0.31

# CRYSTAL
CALIBRATION_CRYSTAL_S = 1960.92
CALIBRATION_CRYSTAL_C = 97.531
CALIBRATION_CRYSTAL_Chi = 0.41

def calibrateEnergy(df, branch):
    df.rename(columns={'Energy': 'nPE'}, inplace=True)
    if branch in ['DRBTScin', 'DRETScinLeft', 'DRETScinRight']:
        df['Energy'] = df['nPE'] / CALIBRATION_BARREL_S
    elif branch in ['DRBTCher', 'DRETCherLeft', 'DRETCherRight']:
        df['Energy'] = df['nPE'] / CALIBRATION_BARREL_C
    elif branch in ['SCEPCal_MainScounts']:
        df['Energy'] = df['nPE'] / CALIBRATION_CRYSTAL_S
    elif branch in ['SCEPCal_MainCcounts']:
        df['Energy'] = df['nPE'] / CALIBRATION_CRYSTAL_C
    else:
        df['Energy'] = df['nPE']  # No calibration for unknown branches
    


def load_all_branches(energy_gev, particle):
    """Load and calibrate all branches for one energy point."""
    dfs = []
    for branch, layer_id in LAYER_MAP.items():
        try:
            df = pd.read_csv(f"{PATH}{particle}_{energy_gev}GeV_{branch}.csv")
            calibrateEnergy(df, branch)
            df['branch']   = branch
            df['layer_id'] = layer_id
            dfs.append(df)
        except FileNotFoundError:
            print(f"    WARNING: {particle}_{energy_gev}GeV_{branch}.csv not found")
    return pd.concat(dfs, ignore_index=True)


def plotStuff(pType, energy):
    df = load_all_branches(energy, pType)
    df = df[df['nPE'] > 0]
    df['R'] = np.sqrt(df['PosX']**2 + df['PosY']**2)
    df['phi'] = np.arctan2(df['PosY'], df['PosX'])

    for branch in df['branch'].unique():
        branchmask = df['branch'] == branch
        R_min = df[branchmask]['R'].min()
        R_max = df[branchmask]['R'].max()
        RSlices = np.linspace(R_min, R_max+1, 9)
        fig = plt.figure(figsize=(20, 10))
        for i in range(len(RSlices)-1):
            mask = (df['R'] >= RSlices[i]) & (df['R'] < RSlices[i+1]) & branchmask
            
            ax = plt.subplot(2, 4, i+1)
            
            if mask.sum() == 0:  
                ax.text(0.5, 0.5, 'No hits', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{branch} R: {RSlices[i]:.1f} - {RSlices[i+1]:.1f} mm')
                continue
            
            h = ax.hist2d(df[mask]['PosZ'], df[mask]['phi'],
                        weights=df[mask]['Energy'], bins=20,
                        norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
            plt.colorbar(h[3], ax=ax, label='Energy (GeV)') 
            ax.set_xlabel('PosZ (mm)')
            ax.set_ylabel('Phi (radians)')
            ax.set_title(f'{branch} R: {RSlices[i]:.1f} - {RSlices[i+1]:.1f} mm')
        plt.tight_layout()
        plt.suptitle(f'{branch} Energy Distribution in R Slices ({pType} {energy} GeV)', y=1.02)
        plt.savefig(f'rSlicePlots/{branch}_R_slices_{pType}_{energy}GeV.pdf')
        plt.close(fig) 


if __name__ == "__main__":
    PARTICLE_LIST = ["electron", "pion"]
    ENERGY_LIST = [10, 20, 40, 60, 80]

    for pType in PARTICLE_LIST:
        for energy in ENERGY_LIST:
            if energy == 80 and pType == "electron":
                continue  
            print(f"Processing {pType} at {energy} GeV...")
            plotStuff(pType, energy)

    




