import numpy as np
import plotly.graph_objects as go
import pandas as pd


def draw_box(fig, x_range, y_range, z_range, color='deepskyblue', lw=1.5, alpha=0.7):
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
    # Plotly draws disconnected segments by inserting None between pairs
    xs, ys, zs = [], [], []
    for edge in edges:
        (x0, y0, z0), (x1, y1, z1) = edge
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=color, width=lw),
        opacity=alpha,
        showlegend=False,
        hoverinfo='skip',
    ))


def _style_fig(fig, event, ptype, energy, global_vmin, global_vmax):
    """Black background, no axes, no grid."""
    fig.update_layout(
        paper_bgcolor='black',
        scene=dict(
            bgcolor='black',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        legend=dict(
            bgcolor='rgba(17,17,17,0.8)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=10),
        ),
        title=dict(
            text=f"{ptype.capitalize()}  {energy}  |  event {event}<br>"
                 f"<sup>Hit energy  {global_vmin*1000:.3f} – {global_vmax*1000:.3f} MeV  (marker size ∝ energy)</sup>",
            font=dict(color='white', size=13),
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def drawShower(fig, showerVector, EnergyVector, color='white', vmin=None, vmax=None, size_scale=20, alpha=0.8, name='hits', log_scale=False):
    if vmin is None: vmin = 0
    if vmax is None: vmax = EnergyVector.max()
    if log_scale:
        E_log    = np.log10(EnergyVector + 1)
        vmin_log = np.log10(vmin + 1)
        vmax_log = np.log10(vmax + 1)
        E_norm   = (E_log - vmin_log) / (vmax_log - vmin_log + 1e-12)
    else:
        E_norm = (EnergyVector - vmin) / (vmax - vmin + 1e-12)
    sizes  = 4 + E_norm * size_scale
    fig.add_trace(go.Scatter3d(
        x=showerVector[:, 0],
        y=showerVector[:, 1],
        z=showerVector[:, 2],
        mode='markers',
        marker=dict(color=color, size=sizes, opacity=alpha,
                    line=dict(width=0)),
        name=name,
        hovertemplate='E: %{customdata:.4f} GeV<extra>' + name + '</extra>',
        customdata=EnergyVector,
    ))


PDG_STYLE = {
    -211:       ('red',         'π⁻ (primary)', 1.5),
    211:        ('orangered',   'π⁺',           1.5),
    11:         ('royalblue',   'e⁻',           1.0),
    -11:        ('deepskyblue', 'e⁺',           1.0),
    22:         ('yellow',      'γ',            0.8),
    2212:       ('limegreen',   'p',            1.2),
    2112:       ('gray',        'n',            1.0),
    1000060120: ('violet',      '¹²C',          1.5),
    1000020040: ('hotpink',     '⁴He',          1.5),
}
DEFAULT_STYLE = ('white', 'other', 0.8)


def drawMCTracks(fig, mc_df, min_track_length=10.0):
    seen_pdgs = set()
    for _, row in mc_df.iterrows():
        vx, vy, vz = row['vertexX'], row['vertexY'], row['vertexZ']
        ex, ey, ez = row['endpointX'], row['endpointY'], row['endpointZ']

        length = np.sqrt((ex-vx)**2 + (ey-vy)**2 + (ez-vz)**2)
        if length < min_track_length:
            continue

        pdg = int(row['PDG'])
        color, label, lw = PDG_STYLE.get(pdg, DEFAULT_STYLE)
        is_primary = row['generatorStatus'] == 1

        show_in_legend = pdg not in seen_pdgs
        fig.add_trace(go.Scatter3d(
            x=[vx, ex], y=[vy, ey], z=[vz, ez],
            mode='lines',
            line=dict(color=color, width=lw * (2.5 if is_primary else 1.5)),
            opacity=1.0 if is_primary else 0.55,
            name=label,
            legendgroup=str(pdg),
            showlegend=show_in_legend,
            hoverinfo='skip',
        ))

        if not is_primary:
            fig.add_trace(go.Scatter3d(
                x=[vx], y=[vy], z=[vz],
                mode='markers',
                marker=dict(color=color, size=3, opacity=0.6),
                showlegend=False,
                legendgroup=str(pdg),
                hoverinfo='skip',
            ))

        seen_pdgs.add(pdg)

    # Origin marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(color='white', size=6, symbol='diamond'),
        name='IP',
        hoverinfo='skip',
    ))
    return seen_pdgs


BRANCH_STYLE = {
        'SCEPCal_MainScounts': {'color': 'yellow',       'label': 'SCEPCal Scounts'},
        'SCEPCal_MainCcounts': {'color': 'purple',       'label': 'SCEPCal Ccounts'},
        'DRBTCher':         {'color': 'deepskyblue',  'label': 'DRBTCher'},
        'DRBTScin':         {'color': 'cyan',         'label': 'DRBTScin'},
        'DRETCherLeft':     {'color': 'lime',         'label': 'DRETCher Left'},
        'DRETCherRight':    {'color': 'lime',         'label': 'DRETCher Right'},
        'DRETScinLeft':     {'color': 'magenta',      'label': 'DRETScin Left'},
        'DRETScinRight':    {'color': 'magenta',      'label': 'DRETScin Right'},
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

# def calibrateEnergy(df):
#     df['DRBT_S_GeV'] = (df['DRBTScin'] + df['DRETScinLeft'] + df['DRETScinRight']) / CALIBRATION_BARREL_S
#     df['DRBT_C_GeV'] = (df['DRBTCher'] + df['DRETCherLeft'] + df['DRETCherRight']) / CALIBRATION_BARREL_C
#     df['Crystal_S_GeV'] = df['SCEPCal_MainScounts'] / CALIBRATION_CRYSTAL_S
#     df['Crystal_C_GeV'] = df['SCEPCal_MainCcounts'] / CALIBRATION_CRYSTAL_C
#     df['E_Barrel']   = (df['DRBT_S_GeV']   - CALIBRATION_BARREL_Chi   * df['DRBT_C_GeV'])   / (1 - CALIBRATION_BARREL_Chi)
#     df['E_Crystal'] = (df['Crystal_S_GeV'] - CALIBRATION_CRYSTAL_Chi * df['Crystal_C_GeV']) / (1 - CALIBRATION_CRYSTAL_Chi)
#     df['E_Total']   = df['E_Barrel'] + df['E_Crystal']

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


    fig = go.Figure()

    draw_box(fig, **DRBTCher_BOUNDS,      color='deepskyblue', lw=1.5, alpha=0.7)
    draw_box(fig, **DRETCherLeft_BOUNDS,  color='lime',        lw=1.5, alpha=0.7)
    draw_box(fig, **DRETCherRight_BOUNDS, color='lime',        lw=1.5, alpha=0.7)
    draw_box(fig, **DRBTScin_BOUNDS,      color='cyan',        lw=1.5, alpha=0.7)
    draw_box(fig, **DRETScinLeft_BOUNDS,  color='magenta',     lw=1.5, alpha=0.7)
    draw_box(fig, **DRETScinRight_BOUNDS, color='magenta',     lw=1.5, alpha=0.7)
    draw_box(fig, **SCEPCal_BOUNDS,       color='yellow',      lw=1.5, alpha=0.7)

    # Beam line
    z_min, z_max = -4074.94, 4088.43
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[z_min, z_max],
        mode='lines',
        line=dict(color='white', width=1.5, dash='dash'),
        name='Beam line',
        hoverinfo='skip',
    ))

    # Loading event
    PATH = "/home/hamza/Workspace/Data/csv/"
    PTYPE = "pion"
    ENERGY = "10GeV"
    # Good events to show: 3741
    # EVENT = np.random.randint(0, 5000)
    EVENT = 3741
    SIZE_SCALE = 50 
    print(f"[INFO] Loading event {EVENT}...")


    all_data = {}
    global_vmin, global_vmax = np.inf, -np.inf

    for branch in BRANCH_STYLE:
        print(f"[INFO] Processing branch '{branch}'...")
        chunks = []
        for chunk in pd.read_csv(f"{PATH}{PTYPE}_{ENERGY}_{branch}.csv", chunksize=10000):
            filtered = chunk[chunk['Event'] == EVENT]
            if len(filtered) > 0:
                chunks.append(filtered)
        if not chunks:
            continue
        df = pd.concat(chunks)
        calibrateEnergy(df, branch)
        df = df[df['nPE'] > 0]
        all_data[branch] = df
        global_vmin = min(global_vmin, df['Energy'].min())
        global_vmax = max(global_vmax, df['Energy'].max())

    print(f"[INFO] Global energy range: {global_vmin*1000:.4f} - {global_vmax*1000:.4f} MeV")

    # Drawing shower
    for branch, style in BRANCH_STYLE.items():
        if branch not in all_data:
            continue
        df = all_data[branch]
        showerVector = df[['PosX', 'PosY', 'PosZ']].values
        EnergyVector = df['Energy'].values
        drawShower(fig, showerVector, EnergyVector,
                   color=style['color'],
                   vmin=global_vmin, vmax=global_vmax,
                   size_scale=SIZE_SCALE,
                   name=style['label'],
                   log_scale=True)
        print(f"[INFO] {branch}: {len(df)} hits")


    # MC truth particles
    mc_df = pd.read_csv(f"{PATH}{PTYPE}_{ENERGY}_MCParticles.csv")
    mc_ev = mc_df[mc_df['Event'] == EVENT].copy()
    print(f"[INFO] MC particles: {len(mc_ev)}")

    seen_pdgs = drawMCTracks(fig, mc_ev, min_track_length=10.0)

    # Style + title
    _style_fig(fig, EVENT, PTYPE, ENERGY, global_vmin, global_vmax)

    # Export
    OUTPUT = f"event_{PTYPE}_{ENERGY}_{EVENT}.html"
    fig.write_html(OUTPUT, include_plotlyjs='cdn')
    print(f"[INFO] Saved → {OUTPUT}")

    fig.show()   # also opens in browser directly


    # NOTE: SCEPCal S and C count; I DO NOT KNOW HOW TO DEAL WITH THEM!! left them out for now.
    # TODO:
    # Compute total/Calibrated energy deposit
    # Include MC truth info
    # Use better colors; especially DRBT: I cant tell the difference between Cherenkov and Scintillator hits.
    # Add event info the title (energy, particle type, etc.)
    # include argparse: p-type, energy, event.
    # Add electron support (currently only pions)