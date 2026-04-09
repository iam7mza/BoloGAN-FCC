import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



PATH = "/storage-hpc/bologan/alhaddad/DataCSV/"


BRANCHES = ["DRBTCher", "DRBTScin", "DRETCherLeft",
            "DRETCherRight", "DRETScinLeft", "DRETScinRight",
            "SCEPCal_MainCcounts", "SCEPCal_MainScounts"]

def loadBranches(P_TYPE, P_ENERGY):
    df = pd.DataFrame({"Event": range(5000)})
    for branch in BRANCHES:
        FILE = f"{PATH}{P_TYPE}_{P_ENERGY}GeV_{branch}.csv"
        df_branch = pd.read_csv(FILE)
        df_branch = df_branch.groupby('Event')['Energy'].sum()
        df_branch = df_branch.reindex(range(5000), fill_value=0)
        df[branch] = df_branch.values

    return df


# P.E/GeV calibration 
# const double CALIBRATION_CONSTANT_S = 206.284; // S p.e./GeV
# const double CALIBRATION_CONSTANT_C = 68.069; // C p.e./GeV
# const double CALIBRATION_CONSTANT_Chi = 0.31; // hadronic scale calibration constant

# // --- CALIBRATION CONSTANTS CRYSTALS (p.e./GeV) ---
# const double CALIBRATION_CRYSTAL_S = 1960.92; // S p.e./GeV
# const double CALIBRATION_CRYSTAL_C = 97.531; // C p.e./GeV
# const double CALIBRATION_CRYSTAL_Chi = 0.41; // hadronic scale calibration constant for crystals

CALIBRATION_BARREL_S = 206.284
CALIBRATION_BARREL_C = 68.069
CALIBRATION_BARREL_Chi = 0.31

CALIBRATION_CRYSTAL_S = 1960.92
CALIBRATION_CRYSTAL_C = 97.531
CALIBRATION_CRYSTAL_Chi = 0.41

def calibrateEnergy(df):
    df['DRBT_S_GeV'] = (df['DRBTScin'] + df['DRETScinLeft'] + df['DRETScinRight']) / CALIBRATION_BARREL_S
    df['DRBT_C_GeV'] = (df['DRBTCher'] + df['DRETCherLeft'] + df['DRETCherRight']) / CALIBRATION_BARREL_C
    df['Crystal_S_GeV'] = df['SCEPCal_MainScounts'] / CALIBRATION_CRYSTAL_S
    df['Crystal_C_GeV'] = df['SCEPCal_MainCcounts'] / CALIBRATION_CRYSTAL_C
    df['E_Barrel']   = (df['DRBT_S_GeV']   - CALIBRATION_BARREL_Chi   * df['DRBT_C_GeV'])   / (1 - CALIBRATION_BARREL_Chi)
    df['E_Crystal'] = (df['Crystal_S_GeV'] - CALIBRATION_CRYSTAL_Chi * df['Crystal_C_GeV']) / (1 - CALIBRATION_CRYSTAL_Chi)
    df['E_Total']   = df['E_Barrel'] + df['E_Crystal']

if __name__ == "__main__":
    PARTICLE_LIST = ["pion", "electron"]
    ENERGY_LIST = [10, 20, 40, 60, 80]

    fig, ax = plt.subplots(figsize=(10, 8))  
    for particle in PARTICLE_LIST:
        for energy in ENERGY_LIST:
            if energy == 80 and particle == "electron":
                continue

            print(f"[INFO] Processing {particle} - {energy} GeV..")
            df = loadBranches(particle, energy)
            calibrateEnergy(df)

            fig_indiv, axes_indiv = plt.subplots(2, 3, figsize=(12, 6))
            SUBPLOTLIST = ['DRBT_S_GeV', 'DRBT_C_GeV', 'Crystal_S_GeV', 'Crystal_C_GeV', 'E_Total']
            for i, col in enumerate(SUBPLOTLIST):
                plt.subplot(2, 3, i+1)
                plt.hist(df[col], bins=100, alpha=0.7, color='blue', histtype='step')
                plt.title(f"{col} - {particle} {energy} GeV")
                plt.xlabel("Energy (GeV)")
                plt.ylabel("Counts")
                plt.legend([f"Mean: {df[col].mean():.3f} GeV\nStd: {df[col].std():.3f} GeV"])

            plt.subplot(2, 3, 6)
            plt.scatter(
                (df['DRBT_S_GeV'] + df['Crystal_S_GeV']) / df['E_Total'],
                (df['DRBT_C_GeV'] + df['Crystal_C_GeV']) / df['E_Total'],
                alpha=0.5, label='DRBT + Crystal', s=3
            )
            plt.xlabel('S/E')
            plt.ylabel('C/E')
            plt.title('(S, C) / E  Correlation')
            plt.plot([0, 1], [0, 1], 'r--', label='S=C')
            plt.plot([0, 1], [1, 1], '--')
            plt.plot([1, 1], [0, 1], '--')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Plots/{particle}_{energy}GeV_EnergyDistribution.png")
            plt.close(fig_indiv)  


            ax.scatter(
                (df['DRBT_S_GeV'] + df['Crystal_S_GeV']) / df['E_Total'],
                (df['DRBT_C_GeV'] + df['Crystal_C_GeV']) / df['E_Total'],
                alpha=0.5, s=10,
                color="cyan" if particle == "electron" else "orange",
                edgecolors='black'
            )

    ax.set_xlabel('S/E')
    ax.set_ylabel('C/E')
    ax.set_title('(S, C) / E  Correlation for All Particles and Energies')
    ax.plot([0, 1], [0, 1], 'r--', label='S=C')
    ax.plot([0, 1], [1, 1], '--')
    ax.plot([1, 1], [0, 1], '--')
    plt.scatter([], [], color='cyan', label='Electron', alpha=0.5, s=10)
    plt.scatter([], [], color='orange', label='Pion', alpha=0.5, s=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig("Plots/AllParticles_AllEnergies_SCEnergyCorrelation.png")