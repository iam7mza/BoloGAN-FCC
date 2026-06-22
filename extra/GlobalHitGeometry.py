from sys import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path.append("/afs/cern.ch/user/h/halhadda/BoloGAN-FCC/voxelization/")
from voxelizer import loadData, PTYPE_CONFIG


if __name__ == "__main__":

    for ptype, (pid, energies) in PTYPE_CONFIG.items():
        particle_data = pd.DataFrame()
        for energy in energies:
            print(f"\n{'='*50}")
            print(f"[INFO] Loading data for {ptype} {energy} GeV...")
            df = loadData(ptype, energy)
            # normalizing Energy
            df['Energy'] = df['Energy'] / energy
            df['phi'] = np.arctan2(df['PosY'], df['PosX'])
            particle_data = pd.concat([particle_data, df], ignore_index=True)
        
        print(f"\n[INFO] Global geometry for {ptype} across all energies:")
        print(f"R min: {particle_data['R'].min():.3f} mm")
        print(f"R max: {particle_data['R'].max():.3f} mm")
        print(f"z min: {particle_data['PosZ'].min():.3f} mm")
        print(f"z max: {particle_data['PosZ'].max():.3f} mm")

        # R_bins = np.linspace(particle_data['R'].min(), particle_data['R'].max()+1, 25, endpoint=True)
        R_bins = [particle_data['R'].min(), 2259, 2439, 2799, particle_data['R'].max()+1]  # Based on detector geometry
        fig = plt.figure(figsize=(15, 15))
        fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        n_bins = len(R_bins) - 1
        for i in range(n_bins):
            # 2D histogram for E deposited in Z vs phi for each R slice
            ax = fig.add_subplot(2, 2, i+1) # note the 2x2 grid for 4 R slices. If you have more slices, adjust accordingly.
            mask = (particle_data['R'] >= R_bins[i]) & (particle_data['R'] < R_bins[i+1])
            h = ax.hist2d(
                particle_data[mask]['PosZ'],
                particle_data[mask]['phi'],
                weights=particle_data[mask]['Energy'],
                bins=[20, 20],
                norm=plt.matplotlib.colors.LogNorm(),
                cmap=plt.cm.viridis,
            )
            ax.figure.colorbar(h[3], ax=ax, label='norm_E')
            ax.set_xlabel('PosZ (mm)')
            ax.set_ylabel('Phi (radians)')
            ax.set_title(f'R: {R_bins[i]:.1f}–{R_bins[i+1]:.1f} mm')

            # 1D histogram for E deposited in R 
            axes[i].hist(
                particle_data[mask]['R'],
                weights=particle_data[mask]['Energy'],
                bins=50,
                color='steelblue',
                edgecolor='none',
            )
            axes[i].set_xlabel('R (mm)')
            axes[i].set_ylabel('Deposited Energy (norm)')
            axes[i].set_title(f'R: {R_bins[i]:.1f}–{R_bins[i+1]:.1f} mm')
            axes[i].set_yscale('log')


        # save fig (2D)
        fig.suptitle(f'{ptype}, energy distribution across all energies\n', y=1.02)
        fig.tight_layout()
        fig.savefig(f'GlobalGeometry/{ptype}geometry_info_allEnergies.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {ptype}geometry_info_allEnergies.pdf")

        # save fig2 (1D)
        fig2.suptitle(f'{ptype}, radial energy deposition across all energies\n', y=1.02)
        fig2.tight_layout()
        fig2.savefig(f'GlobalGeometry/{ptype}_radial_energy_allEnergies.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {ptype}_radial_energy_allEnergies.pdf")