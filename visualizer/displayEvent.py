import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.patches as mpatches
import pandas as pd


def draw_box(ax, x_range, y_range, z_range, color='deepskyblue', lw=1.5, alpha=0.7):
    """Draw the 12 edges of an axis-aligned bounding box."""
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range

    edges = [
        [(xmin, ymin, zmin), (xmax, ymin, zmin)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin)],
        [(xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymax, zmin), (xmin, ymin, zmin)],
        [(xmin, ymin, zmax), (xmax, ymin, zmax)],
        [(xmax, ymin, zmax), (xmax, ymax, zmax)],
        [(xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymax, zmax), (xmin, ymin, zmax)],
        [(xmin, ymin, zmin), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymin, zmax)],
        [(xmax, ymax, zmin), (xmax, ymax, zmax)],
        [(xmin, ymax, zmin), (xmin, ymax, zmax)],
    ]
    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha)

def _style_ax(ax):
    """Black background, no grid, no panes."""
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')


def drawShower(ax, showerVector, EnergyVector, color='white', vmin=None, vmax=None, size_scale=400, alpha=0.8):
    if vmin is None: vmin = EnergyVector.min()
    if vmax is None: vmax = EnergyVector.max()
    E_norm = (EnergyVector - vmin) / (vmax - vmin + 1e-12)
    sizes  = 5 + E_norm * size_scale
    ax.scatter(showerVector[:, 0], showerVector[:, 1], showerVector[:, 2],
               c=color, s=sizes, alpha=alpha, edgecolors='white', zorder=5)

if __name__ == "__main__":

    # Detector geometry (The real detector is cylinderical, but I aint gonna do allat)
    DRBTCher_BOUNDS = {
        'x_range': (-3459.98, 4567.76),
        'y_range': (-4001.93, 4240.6),
        'z_range': (-2957.57, 3114.86),
    }
    DRETCherLeft_BOUNDS = {
        'x_range': (-2730.2, 2761.94),
        'y_range': (-2663.14, 2730.5),
        'z_range': (-4074.94, -2800.38),
    }
    DRETCherRight_BOUNDS = {
        'x_range': (-2723.6, 2763.38),
        'y_range': (-2690.14, 2751.99),
        'z_range': (2800.46, 3969.27),
    }
    DRBTScin_BOUNDS = {
        'x_range': (-3920.43, 4546.71),
        'y_range': (-4303.39, 4336.3),
        'z_range': (-3308.44, 3652.22),
    }
    DRETScinLeft_BOUNDS = {
        'x_range': (-2750.78, 2971.34),
        'y_range': (-2960.51, 3173.94),
        'z_range': (-3847.83, -2800.19),
    }
    DRETScinRight_BOUNDS = {
        'x_range': (-2766.79, 2971.24),
        'y_range': (-2810.08, 2831.46),
        'z_range': (2800.19, 4088.43),
    }
    SCEPCal_BOUNDS = {
        'x_range': (-2375.0, 2375.0),
        'y_range': (-2375.0, 2375.0),
        'z_range': (-2571.92, 2571.92),
    }

    # NOTE: SCEPCal S and C count; I DO NOT KNOW HOW TO DEAL WITH THEM!! left them out for now.
    BRANCH_STYLE = {
        'SCEPCal_MainEdep': {'color': 'yellow',      'label': 'SCEPCal Edep'},
        'DRBTCher':         {'color': 'deepskyblue',  'label': 'DRBTCher'},
        'DRBTScin':         {'color': 'cyan',         'label': 'DRBTScin'},
        'DRETCherLeft':     {'color': 'lime',         'label': 'DRETCher Left'},
        'DRETCherRight':    {'color': 'lime',         'label': 'DRETCher Right'},
        'DRETScinLeft':     {'color': 'magenta',      'label': 'DRETScin Left'},
        'DRETScinRight':    {'color': 'magenta',      'label': 'DRETScin Right'},
    }

    fig = plt.figure(facecolor='black', figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    _style_ax(ax)

    draw_box(ax, **DRBTCher_BOUNDS,    color='deepskyblue', lw=1.5, alpha=0.7)
    draw_box(ax, **DRETCherLeft_BOUNDS, color='lime',       lw=1.5, alpha=0.7)
    draw_box(ax, **DRETCherRight_BOUNDS,color='lime',       lw=1.5, alpha=0.7)
    draw_box(ax, **DRBTScin_BOUNDS,    color='cyan',        lw=1.5, alpha=0.7)
    draw_box(ax, **DRETScinLeft_BOUNDS, color='magenta',    lw=1.5, alpha=0.7)
    draw_box(ax, **DRETScinRight_BOUNDS,color='magenta',    lw=1.5, alpha=0.7)
    draw_box(ax, **SCEPCal_BOUNDS,     color='yellow',      lw=1.5, alpha=0.7)

    # Beam line
    z_min, z_max = -4074.94, 4088.43
    ax.plot([0, 0], [0, 0], [z_min, z_max], color='white', lw=1.0, alpha=0.8, linestyle='--')

    # loading event
    PATH = "/home/hamza/Workspace/Data/csv/"
    EVENT = np.random.randint(0, 5000)
    print(f"[INFO] Loading event {EVENT}...")


    all_data = {}
    global_vmin, global_vmax = np.inf, -np.inf

    for branch in BRANCH_STYLE:
        print(f"[INFO] Processing branch '{branch}'...")
        chunks = []
        for chunk in pd.read_csv(f"{PATH}pion_10GeV_{branch}.csv", chunksize=10000):
            filtered = chunk[chunk['Event'] == EVENT]
            if len(filtered) > 0:
                chunks.append(filtered)
        if not chunks:
            continue
        df = pd.concat(chunks)
        all_data[branch] = df
        global_vmin = min(global_vmin, df['Energy'].min())
        global_vmax = max(global_vmax, df['Energy'].max())

    print(f"[INFO] Global energy range: {global_vmin:.4f} - {global_vmax:.4f} MeV")

    # Drawing shower
    for branch, style in BRANCH_STYLE.items():
        if branch not in all_data:
            continue
        df = all_data[branch]
        showerVector = df[['PosX', 'PosY', 'PosZ']].values
        EnergyVector = df['Energy'].values
        drawShower(ax, showerVector, EnergyVector,
                color=style['color'],
                vmin=global_vmin, vmax=global_vmax)
        print(f"[INFO] {branch}: {len(df)} hits")
    
    # Legend
    legend_entries = [
        mpatches.Patch(color='deepskyblue', label='DRBTCher (wireframe)'),
        mpatches.Patch(color='lime',        label='DRETCher (wireframe)'),
        mpatches.Patch(color='cyan',        label='DRBTScin (wireframe)'),
        mpatches.Patch(color='magenta',     label='DRETScin (wireframe)'),
        mpatches.Patch(color='yellow',      label='SCEPCal (wireframe)'),
        Line3D([0], [0], [0], color='white', linestyle='--', label='Beam line'),
        # hits
        plt.scatter([], [], c='yellow',     s=50,  label='SCEPCal hits',    alpha=0.8),
        plt.scatter([], [], c='deepskyblue',s=50,  label='DRBTCher hits',   alpha=0.8),
        plt.scatter([], [], c='cyan',       s=50,  label='DRBTScin hits',   alpha=0.8),
        plt.scatter([], [], c='lime',       s=50,  label='DRETCher hits',   alpha=0.8),
        plt.scatter([], [], c='magenta',    s=50,  label='DRETScin hits',   alpha=0.8),
        # size guide
        plt.scatter([], [], c='white', s=5,   label=f'Low  (~{global_vmin:.3f} MeV)', alpha=0.8),
        plt.scatter([], [], c='white', s=200, label=f'High (~{global_vmax:.3f} MeV)', alpha=0.8),
    ]
    leg = ax.legend(handles=legend_entries, loc='upper left',
                    facecolor='#111111', edgecolor='white', labelcolor='white',
                    fontsize=9, framealpha=0.7)


    # INITIAL VIEW
    ax.view_init(elev=30, azim=160, roll=-50)

    ax.set_position([0, 0, 1, 1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

    # TODO: 
    # Compute total/Calibrated energy deposit
    # Include MC truth info
    # Use better colors; especially DRBT: I cant tell the difference between Cherenkov and Scintillator hits. 
    # Add event info the title (energy, particle type, etc.)
    # include argparse: p-type, energy, event. 