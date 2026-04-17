from sys import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path.append("/home/a/alhaddad/BoloGAN-FCC/voxelization/")
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

        R_bins = np.linspace(particle_data['R'].min(), particle_data['R'].max()+1, 25, endpoint=True)
        fig = plt.figure(figsize=(15, 15))
        for i in range(24):
            ax = fig.add_subplot(5, 5, i+1)
            mask = (particle_data['R'] >= R_bins[i]) & (particle_data['R'] < R_bins[i+1])
            h = ax.hist2d(
                particle_data[mask]['PosZ'],
                particle_data[mask]['phi'],
                weights=particle_data[mask]['Energy'],
                bins=[20, 20],
                norm=plt.matplotlib.colors.LogNorm(),
                cmap=plt.cm.viridis,
            )
            plt.colorbar(h[3], ax=ax, label='norm_E')
            ax.set_xlabel('PosZ (mm)')
            ax.set_ylabel('Phi (radians)')
            ax.set_title(f'R: {R_bins[i]:.1f}–{R_bins[i+1]:.1f} mm')

        plt.suptitle(
            f'{ptype}, energy distribution across all energies\n',
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(f'GlobalGeometry/{ptype}geometry_info_allEnergies.pdf',
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {ptype}geometry_info_allEnergies.pdf")
            