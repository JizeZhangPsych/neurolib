#%%
from pathlib import Path
import os, sys
import numpy as np
from sys import argv
from osl_dynamics.analysis import static, power, connectivity
from osl_dynamics.data import Data
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.getcwd()))
from utils.io import load_pkl, save_pkl
from utils.static import compute_aec
from utils.visualize import StaticVisualizer, _colormap_transparent
from utils.pathfinder import IrenePathfinder, StaresinaPathfinder
from utils import ensure_dir

from mne.io import read_raw

#%%
structurals = "standard"
TMP_DIR = Path("/ohba/pi/mwoolrich/jzhang/staresina_proc/stsprep/")
ensure_dir(TMP_DIR)
sfreq = 250
# pf = IrenePathfinder()
pf = StaresinaPathfinder()

src_key = 'src'

print("File list: ", pf.files.keys())
# print(pf['3111'][src_key])

exclude_ids = ['8111', '8112', '8121', '15111', '15112', '17111', '17112', '26111', '26112', '31111', '31112', '31121', '31211', '31212', '31211']

#%%

input_data = []
for file_id, pth_dict in pf.files.items():
    if file_id in exclude_ids:
        print(f"Excluding {file_id}")
        continue
    try:
        raw = read_raw(pth_dict[src_key], preload=True)
    except KeyError:
        print(f"WARNING: {file_id} has no src file, skipping (likely error in src recon)")
        
    input_data.append(raw.get_data().T) # (samples, channels)

input_n_samples = [d.shape[0] for d in input_data]


#%%

save_path = os.path.join(TMP_DIR, "test_static_features.pkl")

if structurals == "standard":
    save_path = save_path.replace(".pkl", "_no_struct.pkl")

if os.path.exists(save_path):
    # Load static network features
    print("(Step 2-1) Loading static network features ...")
    static_network_features = load_pkl(save_path)
    freqs = static_network_features["freqs"]
    psds = static_network_features["psds"]
    weights = static_network_features["weights"]
    power_maps = static_network_features["power_maps"]
    conn_maps = static_network_features["conn_maps"]
else:
    # Compute subject-specific static power spectra
    print("(Step 2-1) Computing static PSDs ...")
    sfreq = 250 # sampling frequency
    freqs, psds, weights = static.welch_spectra(
        data=input_data,
        sampling_frequency=sfreq,
        window_length=int(sfreq * 2),
        step_size=int(sfreq),
        frequency_range=[1.5, 45],
        return_weights=True,
        standardize=True,
    )
    # dim: (n_subjects, n_parcels, n_freqs)

    # Compute subject-specific static power maps
    print("(Step 2-2) Computing static power maps ...")
    power_maps = dict()
    freq_ranges = [[1.5, 20], [1.5, 4], [4, 8], [8, 13], [13, 20]]
    freq_bands = ["wide", "delta", "theta", "alpha", "beta"]
    for n, freq_range in enumerate(freq_ranges):
        power_maps[freq_bands[n]] = power.variance_from_spectra(
            freqs, psds, frequency_range=freq_range
        )
        # dim: (n_subjects, n_parcels)

    # Compute subject-specific static AEC maps
    print("(Step 2-3) Computing static AEC maps ...")
    conn_maps = compute_aec(
        input_data, sfreq, freq_range=[1.5, 20], tmp_dir=TMP_DIR/"tmp"
    )
    
    # dim: (n_subjects, n_parcles, n_parcels)

    # Save computed features
    print("(Step 2-4) Saving computed features ...")
    output = {
        "freqs": freqs,
        "psds": psds,
        "weights": weights,
        "power_maps": power_maps,
        "conn_maps": conn_maps,
    }
    save_pkl(output, save_path)
    
    
#%%
# --------- [3] ---------- #
#      Visualization       #
# ------------------------ #
print("\n*** STEP 3: VISUALIZATION ***")

for bandname, freq_range in zip(
    ["wide", "delta", "theta", "alpha", "beta"],
    [[1.5, 20], [1.5, 4], [4, 8], [8, 13], [13, 20]],
):
    print(f"Frequency band: {bandname}, range: {freq_range} Hz")
    
    TGT_DIR = TMP_DIR / bandname
    if not os.path.exists(TGT_DIR):
        os.makedirs(TGT_DIR)

    conn_maps = compute_aec(
        input_data, sfreq, freq_range=freq_range, tmp_dir=TGT_DIR/"tmp"
    )

    # Set up visualization tools
    SV = StaticVisualizer()
    cmap_hot_tp = _colormap_transparent("gist_heat")

    # Plot static wide-band power map (averaged over all subjects)
    gpower_all = np.mean(power_maps[bandname], axis=0) # dim: (n_parcels,)
    gpower_range = np.abs(max(gpower_all) - min(gpower_all))

    SV.plot_power_map(
        power_map=gpower_all,
        filename=os.path.join(TGT_DIR, "power_map.png"),
        plot_kwargs={
            "vmin": 0,
            "vmax": max(gpower_all) + 0.1 * gpower_range,
            "symmetric_cbar": False,
            "cmap": cmap_hot_tp,
        },
    )

    # Plot static wide-band AEC map (averaged over all subjects)
    gconn_all = np.mean(conn_maps, axis=0) # dim: (n_parcels, n_parcels)
    gconn_all = connectivity.threshold(
        gconn_all,
        percentile=95
    ) # select top 5%

    SV.plot_aec_conn_map(
        connectivity_map=gconn_all,
        filename=os.path.join(TGT_DIR, "conn_map.png"),
        colormap="Reds",
        plot_kwargs={"edge_vmin": 0, "edge_vmax": np.max(gconn_all)},
    )

# Plot static power spectral densities (averaged over all subjects)
gpsd_all = np.mean(psds, axis=(1, 0)) # dim: (n_freqs,)
gpsd_sem = np.std(np.mean(psds, axis=1), axis=0) / np.sqrt(len(psds)) # dim: (n_freqs,)

SV.plot_psd(
    freqs=freqs,
    psd=gpsd_all,
    error=gpsd_sem,
    filename=os.path.join(TMP_DIR, "psd.png"),
)

print("Computation completed.")

    # %%
