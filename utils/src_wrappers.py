import os, re
import numpy as np
import mne
import matplotlib.pyplot as plt
from osl_ephys import source_recon
from .eeg import parse_subj, HeteroStudy as Study
from osl_ephys.utils.logger import log_or_print
from scipy.signal import welch
from nilearn.plotting import plot_markers
from osl_ephys.source_recon.parcellation import parcel_centers
from .util import ensure_dir

def polhemus_translation(outdir, subject, correct_sign=True):
    
    rhino_pth_dict = source_recon.rhino.get_coreg_filenames(outdir, subject)
    subj_dict = parse_subj(subject)

    polhemus_files = Study([
        os.path.join(os.path.dirname(outdir) + '/{subj}/polhemus/{subj}_{ses}_{run}_{foo}.pom'),
        os.path.join(os.path.dirname(outdir) + '/{subj}/{ses}/polhemus/{subj}_{ses}_{run}_{foo}.pom')
    ])
    polhemus = polhemus_files.get(subj=subj_dict["subj"], ses=subj_dict["ses"], block=subj_dict["block"], run=subj_dict["run"])
    
    assert len(polhemus) == 1
    polhemus = polhemus[0]
    # Extract LOCATION_LIST data
    with open(polhemus, 'r') as f:
        polhemus_content = f.read()
    polhemus_content = re.sub(r"#.*?\n", "\n", polhemus_content)  # Remove comments

    location_list_match = re.search(r"LOCATION_LIST START_LIST([\s\S]*?)LOCATION_LIST END_LIST", polhemus_content)
    assert location_list_match is not None
    locations_data = location_list_match.group(1).strip().splitlines()
    locations = [line.split() for line in locations_data]

    remark_list_match = re.search(r"REMARK_LIST START_LIST([\s\S]*?)REMARK_LIST END_LIST", polhemus_content)
    assert remark_list_match is not None
    remarks = remark_list_match.group(1).strip().splitlines()

    # Iterate through remarks and locations
    headshape_coords = []
    spec_coords = {}
    for idx, remark in enumerate(remarks):
        if remark == 'Left ear':
            spec_coords['lpa'] = np.array(locations[idx], dtype=np.float64)
        elif remark == 'Right ear':
            spec_coords['rpa'] = np.array(locations[idx], dtype=np.float64)
        elif remark == 'Nasion':
            spec_coords['nasion'] = np.array(locations[idx], dtype=np.float64)
        elif remark == 'Cz' and correct_sign:
            spec_coords['Cz'] = np.array(locations[idx], dtype=np.float64)
            headshape_coords.append(np.array(locations[idx], dtype=np.float64))
        else:
            headshape_coords.append(np.array(locations[idx], dtype=np.float64))
            
    # Correct the sign of the coordinates if needed
    if correct_sign:
        sign_x = (spec_coords['lpa'][0] < spec_coords['rpa'][0]) * 2 - 1    # TODO: radiological?
        sign_y = (spec_coords['nasion'][1] > np.mean([float(coord[1]) for coord in headshape_coords])) * 2 - 1
        sign_z = (spec_coords['Cz'][2] > np.mean([float(coord[2]) for coord in headshape_coords])) * 2 - 1
        
        headshape_coords = [[sign_x*x,sign_y*y,sign_z*z] for x,y,z in headshape_coords] 
        spec_coords['lpa'] = [sign_x*spec_coords['lpa'][0], sign_y*spec_coords['lpa'][1], sign_z*spec_coords['lpa'][2]]
        spec_coords['rpa'] = [sign_x*spec_coords['rpa'][0], sign_y*spec_coords['rpa'][1], sign_z*spec_coords['rpa'][2]]
        spec_coords['nasion'] = [sign_x*spec_coords['nasion'][0], sign_y*spec_coords['nasion'][1], sign_z*spec_coords['nasion'][2]]
        
    with open(rhino_pth_dict['polhemus_lpa_file'], 'w') as f:
        f.write(f"{spec_coords['lpa'][0]}\n{spec_coords['lpa'][1]}\n{spec_coords['lpa'][2]}\n")
    with open(rhino_pth_dict['polhemus_rpa_file'], 'w') as f:
        f.write(f"{spec_coords['rpa'][0]}\n{spec_coords['rpa'][1]}\n{spec_coords['rpa'][2]}\n")
    with open(rhino_pth_dict['polhemus_nasion_file'], 'w') as f:
        f.write(f"{spec_coords['nasion'][0]}\n{spec_coords['nasion'][1]}\n{spec_coords['nasion'][2]}\n")      

    # Write headshape coordinates to file
    headshape_coords = np.array(headshape_coords).T  # Transpose to get x, y, z in separate rows
    with open(rhino_pth_dict['polhemus_headshape_file'], 'w') as f:
        for row in headshape_coords:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")
            
def plot_parc(outdir, subject, parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz", nperseg=4096, nfft=None, filename=None, freq_range=[1,35], figsize=(15,3)):
    ensure_dir(f"{outdir}/{subject}/parc_psd")
    if filename is None:
        filename = f"{outdir}/{subject}/parc_psd/psd_{nperseg}"
    else:
        filename = os.path.join(outdir, subject, "parc_psd", filename)
    parcel_data = mne.io.read_raw_fif(f'{outdir}/{subject}/parc/parc-raw.fif', preload=True)
    parc_ts = parcel_data.get_data()
    fs = parcel_data.info['sfreq']
    
    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average
        psd = []
        for i in range(parc_ts.shape[-1]):
            f, p = welch(parc_ts[..., i], fs=fs, nperseg=nperseg, nfft=nfft)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calcualte PSD of continuous data
        f, psd = welch(parc_ts, fs=fs, nperseg=nperseg, nfft=nfft)

    n_parcels = psd.shape[0]

    if freq_range is None:
        freq_range = [f[0], f[-1]]

    # Re-order to use colour to indicate anterior->posterior location
    parc_centers = parcel_centers(parcellation_file)
    order = np.argsort(parc_centers[:, 1])
    parc_centers = parc_centers[order]
    psd = psd[order]

    # Plot PSD
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    cmap = plt.cm.viridis_r
    for i in reversed(range(n_parcels)):
        ax.plot(f, psd[i], c=cmap(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (a.u.)", fontsize=14)
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    # Plot parcel topomap
    inside_ax = ax.inset_axes([0.45, 0.55, 0.5, 0.55])
    plot_markers(np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax)

    # Save
    log_or_print(f"saving {filename}.png")
    plt.savefig(filename+".png")
    plt.close()
    
    for i in reversed(range(n_parcels)):
        plt.figure(figsize=figsize)
        
        plt.plot(f, psd[i])
        plt.xlim(freq_range[0], freq_range[1])
        plot_markers(np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax)
        # Save
        plt.savefig(filename + f"_{i}.png")
        plt.close()
    