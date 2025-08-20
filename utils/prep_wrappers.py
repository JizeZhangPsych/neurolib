import os, re, glob, copy, pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode, kurtosis
from scipy.signal import find_peaks
from scipy.io import loadmat
import torch
from functools import partial
from sklearn.decomposition import PCA
from scipy.linalg import lstsq as scipy_lstsq
from torch.linalg import lstsq

import mne
from mne.preprocessing import find_ecg_events
from osl_ephys.report.preproc_report import plot_channel_dists # usage: plot_channel_dists(raw, savebase)
from osl_ephys.utils.logger import log_or_print
from osl_ephys.preprocessing.osl_wrappers import gesd
from .util import ensure_dir, proc_userargs
from .metric import EEGTracer, psd_band_ratio, psd_band_stat
from .eeg import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, filename2subj, HeteroStudy as Study, Pathfinder, find_spurious_channels, pcs_plot, pick_indices, SingletonEEG, correct_trigger
from .qrs import kteager_detect, qrs_correction, QRSDetector
from ecgdetectors import panPeakDetect
from mne.preprocessing import ICA

spurious_subject_list = ['13121', '8111', '8112', '8121', '17111', '17112', '31111', '31112', '31121']


def voltage_correction(dataset, userargs):
    """Corrects scaling if EOG/ECG/EMG channels are stored in µV but marked as volts."""
    
    ratio_threshold = userargs.get('ratio_threshold', 1000)  # Default threshold for scaling down EOG/ECG channels
    picks = userargs.get('picks', ['eog', 'ecg', 'emg'])  # Channels to check for scaling
    
    # Get EEG stats
    eeg_data = dataset['raw'].get_data(picks='eeg', units='V')
    eeg_rms = np.sqrt(np.mean(eeg_data**2, axis=1))
    ref_rms = np.median(eeg_rms)
    ref_rms = ref_rms

    for ch_type in picks:
        pick_idx = mne.pick_types(dataset['raw'].info, eeg=False,
                        eog=(ch_type=='eog'),
                        ecg=(ch_type=='ecg'),
                        emg=(ch_type=='emg'),)
        if len(pick_idx) == 0:
            continue
        
        data = dataset['raw'].get_data(picks=ch_type, units='V')
        
        ch_rms = np.mean(np.sqrt(np.mean(data**2, axis=1)))
        
        if (ch_rms / ref_rms) > ratio_threshold:
            print(f"Warning: {ch_type.upper()} channels have a high RMS value ({ch_rms*1e6:.2f} µV) compared to EEG RMS ({ref_rms*1e6:.2f} µV). Scaling down by 1e6.")
            dataset['raw']._data[pick_idx] /= 1e6  # Scale down from µV to V

    return dataset

def cleanup(dataset, userargs):
    keywords = userargs.get('keywords', ['_ep'])
    pop_keys = []
    for k in dataset.keys():
        for keyword in keywords:
            if keyword in k:
                pop_keys.append(k)
                break
    for k in pop_keys:
        dataset.pop(k)
    return dataset


def init_tracer(dataset, userargs):
    """Initializes the EEGTracer with specific metrics for the dataset."""
    
    tracer_kwargs = {        
        "psd_mean": partial(psd_band_stat, band=[1, 40], fn=np.mean),
        "psd_kurtosis": partial(psd_band_stat, band=[1, 40], fn=kurtosis),
        "psd_maxmed_ratio": partial(psd_band_ratio, band1=[1, 40], fn1=np.max, band2=[1, 40], fn2=np.median),
        "psd_alpha_mean": partial(psd_band_stat, band='alpha', fn=np.mean),
        "psd_alpha_kurtosis": partial(psd_band_stat, band='alpha', fn=kurtosis),
        "psd_alpha_maxmed_ratio": partial(psd_band_ratio, band1='alpha', fn1=np.max, band2='alpha', fn2=np.median),
        "psd_beta_mean": partial(psd_band_stat, band='beta', fn=np.mean),
        "psd_beta_kurtosis": partial(psd_band_stat, band='beta', fn=kurtosis),
        "psd_beta_maxmed_ratio": partial(psd_band_ratio, band1='beta', fn1=np.max, band2='beta', fn2=np.median),
    }
    
    tracer_kwargs.update(userargs)
    
    dataset['tracer'] = EEGTracer(**tracer_kwargs)
    
    return dataset

def summary(dataset, userargs):
    """Generates a summary of the dataset, including basic statistics and channel information.
    Currently only plot the tracer checkpoints."""
    
    subject = dataset['subject']
    if 'tracer' in dataset:
        dataset['tracer'].plot(save_pth=os.path.join(dataset['pf'].get_fdr_dict()['prep'], "ckpt", subject), show=False)
        
    # dataset.pop('tracer', None)  # Remove tracer from dataset after plotting
    return dataset

def initialize(dataset, userargs):
    ds_name = userargs.get('ds_name', 'staresina')
    dataset['pf'] = Pathfinder(**userargs)
    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']

    if 'Trigger' in dataset['raw'].ch_names:
        dataset['raw'].drop_channels(['Trigger'])
        
    subject = filename2subj(dataset['raw'].filenames[0], ds_name=ds_name)
    dataset['subject'] = subject

    if ds_name == 'irene':  # for irene, dev_head_t is note set, so we need to set it to identity
        dataset['raw'].info['dev_head_t'] = SingletonEEG("/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/edfs/sub-003_ses-01_run-01_block-01_task-resting_convert.cdt.edf").info['dev_head_t']
        
        if int(subject[:2]) >= 14:
            dataset['raw'].set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'EKG': 'ecg', 'EMG': 'emg'})
        else:
            dataset['raw'].set_channel_types({'VEO': 'eog', 'HEO': 'eog', 'EKG': 'ecg'})
        dataset['tr_event_key'] = ['100005', '1200002']
        dataset['slice_interval'] = 0.061
        dataset['tr_interval'] = 1
        dataset['raw'].drop_channels(['F11', 'F12', 'FT11', 'FT12', 'CB1', 'CB2', 'M1', 'M2', 'Cb1', 'Cb2'], on_missing='warn')
        
        rename_dict = {}
        for ch in dataset['raw'].ch_names:
            new_ch = ch
            if ch.startswith('FP'):
                new_ch = new_ch.replace('FP', 'Fp')
            if ch.endswith('Z'):
                new_ch = new_ch.replace('Z', 'z')
            rename_dict[ch] = new_ch
        dataset['raw'].rename_channels(rename_dict)
        
    elif ds_name == 'staresina':
        dataset['slice_interval'] = 0.07
        dataset['tr_interval'] = 1.14
        if subject == '2111':   # radiographer error, one more session recorded after 400s
            try:
                dataset['raw'] = dataset['raw'].crop(tmin=0, tmax=400)  
            except ValueError:
                print("Warning: Subject 2111 has no data after 400s, so no cropping is needed.")
        if subject == '4121':   # Accidental overwriting of resting stage EEG file after computer prompted an overwriting towards the end of the recording. So a very small file is recorded.
            raise Exception("Subject 4121 has its eeg file corrupted: Accidental overwriting of resting stage EEG file after computer prompted an overwriting towards the end of the recording. So a very small file is recorded.")
        if subject == '31212':   # incorrect event triggering after 311s
            try:
                dataset['raw'] = dataset['raw'].crop(tmin=0, tmax=311)  
            except ValueError:
                print("Warning: Subject 31212 has no data after 311s, so no cropping is needed.")
        if subject == '27212':   # forgot one TR event trigger at onset=18.4582s
            dataset['raw'].annotations.append(18.4582, 0, '1200002')
        if subject == '17121':   # forgot one TR event trigger at onset=59.8632s
            dataset['raw'].annotations.append(59.8632, 0, '1200002')
        dataset['raw'].drop_channels(['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2'], on_missing='warn')
        print("Warning: F11, F12, FT11, FT12, Cb1, Cb2 are dropped from the raw data, as no gel is used in these channels.")
        
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}. Please provide a valid dataset name.")

    return dataset


def calc_psd(dataset, userargs):
    """Calculates the power spectral density (PSD) of the raw data and stores it in the dataset."""
    fmin = userargs.get('fmin', 1)  # minimum frequency for PSD calculation
    fmax = userargs.get('fmax', 49)  # maximum frequency for PSD calculation
    resolution = userargs.get('resolution', 0.05)  # resolution multiplier for psd calc
    picks = userargs.get('picks', 'eeg')  # channels to compute PSD for, default is 'eeg'
    
    n_fft = int(np.round(dataset['raw'].info['sfreq'] / resolution))
    dataset['psd'] = dataset['raw'].compute_psd(n_fft=n_fft, picks=picks, fmin=fmin, fmax=fmax)
    return dataset
    
def mreeg_bad_channels(dataset, userargs):
    """
    Detects bad channels based on their power or frequency characteristics.
    Channels already marked as bad in the dataset will not be considered.
    """
    
    mode = userargs.pop('mode')
    picks = userargs.pop('picks', 'eeg')  # channels to compute power for, default is 'eeg'
    
    if mode == 'freq': # define bad channels as channels with strong power at a specified frequency
        freq = userargs.pop('freq')  # frequency to check for bad channels
        tight_win = userargs.pop('tight_win', 1.14)  # tight window in Hz around the frequency to check for bad channels
        broad_win = userargs.pop('broad_win', [-2.5*tight_win, 2.5*tight_win])  # broad window in Hz around the frequency to check for bad channels
        resolution = userargs.pop('resolution', 0.05)  # resolution for psd calc
        fmin = userargs.pop('fmin', 1)  # minimum frequency for PSD calculation
        fmax = userargs.pop('fmax', 35)  # maximum frequency for PSD calculation
        outlier_side = userargs.pop('outlier_side', 1)  # one-sided or two-sided outlier detection, default is one-sided, only remove large
        
        n_fft = int(np.round(dataset['raw'].info['sfreq'] / resolution))
        
        psd = dataset['raw'].compute_psd(n_fft=n_fft, picks=picks, fmin=fmin, fmax=fmax)
        psd_res = np.median(np.diff(psd._freqs))
        
        tight_hwin = np.round(tight_win / psd_res / 2).astype(np.int64)
        broad_lwin = np.round(broad_win[0] / psd_res).astype(np.int64)
        broad_hwin = np.round(broad_win[1] / psd_res).astype(np.int64)
        
        freq_idx = np.abs(psd._freqs - freq).argmin()
        ratio = psd._data[:, freq_idx-tight_hwin:freq_idx+tight_hwin+1].mean(axis=1) / psd._data[:, freq_idx+broad_lwin:freq_idx+broad_hwin+1].mean(axis=1)
        
        rm_ind, _ = gesd(ratio, outlier_side=outlier_side, **userargs)
        
        if np.any(rm_ind):
            bad_ch = list(np.array(psd.ch_names)[np.where(rm_ind)[0]])
            dataset['raw'].info["bads"].extend(bad_ch)
            log_or_print(f"Bad channels detected by frequency {freq} Hz: {bad_ch}")
        else:
            log_or_print(f"No bad channels detected by frequency {freq} Hz.")
        
    elif mode == 'power': # define bad channels as having significantly higher or lower power than others, usually due to bad contact, usually only consider 15 Hz+ part of the spectrum or 6 Hz- part of the spectrum if detecting higher power, and consider all part of the spectrum if detecting lower power.
        l_freq = userargs.pop('l_freq', None)  # lower frequency bound for power calculation
        h_freq = userargs.pop('h_freq', None)  # upper frequency bound for power calculation
        
        good_ch_names = np.array(dataset['raw'].ch_names)[mne.io.pick._picks_to_idx(dataset['raw'].info, picks)]
        
        filtered_data = mne.filter.filter_data(dataset['raw'].get_data(picks=good_ch_names), l_freq=l_freq, h_freq=h_freq, sfreq=dataset['raw'].info['sfreq'])
        power = np.mean(filtered_data**2, axis=1)
        
        rm_ind, _ = gesd(power, **userargs)
        if np.any(rm_ind):
            bad_ch = list(good_ch_names[np.where(rm_ind)[0]])
            dataset['raw'].info["bads"].extend(bad_ch)
            log_or_print(f"Bad channels detected by power in range {l_freq}-{h_freq} Hz: {bad_ch}")
        else:
            log_or_print(f"No bad channels detected by power in range {l_freq}-{h_freq} Hz.")
    else:
        raise ValueError(f"Unknown mode: {mode}. Please provide a valid mode ('freq' or 'power').")
    
    return dataset

def debug_init(dataset, userargs):
    userargs['ds_name'] = userargs.get('ds_name', 'staresina')
    dataset = initialize(dataset, userargs)
    dataset["real_raw"] = copy.deepcopy(dataset['raw'])
    return dataset

def ckpt_report(dataset, userargs):
    """a function for debugging the preprocessing steps.
        strictly requires Python >=3.7, for dict keys ordering

    Args:
        dataset (dict): the dict containing raw data and metadata
        userargs (dict): a dictionary containing the optional arguments

    Returns:
        dataset: the updated dataset with the extra metadata
    """
    default_args = {
        'ckpt_name': datetime.now().strftime("%H:%M:%S"),
        'resolution': 0.05,
        'max_freq': 50,
        'qrs_event': False,
        'key_to_print': None,
        'always_print': ['EKG'],    # must be name, 'eeg' is not allowed
        'std_channel_pick': 'eeg',
        'print_pcs': True,
        'print_noise': True,
        'ds_name': 'staresina',
        'dB': False,  # whether to plot psd in dB scale
        'focus_range': [100, 110],  # in seconds, for temp_plot
        'log_tracer': True
    }
    userargs = proc_userargs(userargs, default_args)
    
    fs = dataset['raw'].info['sfreq']
    picks = dataset[f"picks_{userargs['key_to_print']}"] if f"picks_{userargs['key_to_print']}" in dataset else 'eeg'
    subject = dataset['subject']
    save_fdr = os.path.join(dataset['pf'].get_fdr_dict()['prep'], "ckpt", subject, userargs['ckpt_name'])
    ensure_dir(save_fdr)
    
    if userargs['key_to_print'] is None:
        userargs['print_noise'] = userargs['print_pcs'] = False
    
    psd = psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], save_pth=os.path.join(save_fdr, f"psd.pdf"), picks=picks, dB=userargs['dB'])
    std = np.mean(np.std(dataset['raw'].get_data(picks=userargs['std_channel_pick'], reject_by_annotation='omit'), axis=1))
    plot_channel_dists(dataset['raw'], os.path.join(save_fdr, f"std={std:.4e}.pdf"))
    
    def print_ch(ch_name):
        extra_str = "temp"
        print_fdr = os.path.join(save_fdr, extra_str)
        ensure_dir(print_fdr)
        
        psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], picks=ch_name, save_pth=os.path.join(print_fdr, f"{ch_name}_psd.pdf"), dB=userargs['dB'])
        temp_plot(dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}.pdf"), name=ch_name)
        temp_plot(dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}.pdf"), name=ch_name)
        if 'ckpt_raw' in dataset:
            if dataset['ckpt_raw'].info['sfreq'] != dataset['raw'].info['sfreq']:
                dataset['ckpt_raw'].resample(dataset['raw'].info['sfreq'])
            if not userargs['qrs_event']:
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}_diff.pdf"),name=ch_name)
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}_diff.pdf"), name=ch_name)
            else:
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}_diff.pdf"),name=ch_name, events=dataset['bcg_ep'].events, event_id=999)
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}_diff.pdf"), name=ch_name, events=dataset['bcg_ep'].events, event_id=999)
    
    channel_to_print = psd.ch_names
    if len(channel_to_print) > 3:
        channel_to_print = np.random.choice(channel_to_print, 3, replace=False)
    
    channel_to_print = np.unique(np.concatenate([np.array(userargs['always_print']), channel_to_print]))
    for ch_name in channel_to_print:
        print_ch(ch_name)
    
    if userargs['print_pcs']:
        pc_fdr_name = os.path.join(dataset['pf'].get_fdr_dict()['prep'], "ckpt", subject, f"pc_{userargs['key_to_print']}")
        ensure_dir(pc_fdr_name)
        
        # channel_idx_to_print = []
        # for ch_name in channel_to_print:
            # if ch_name in psd.ch_names:
                # channel_idx_to_print.append(psd.ch_names.index(ch_name))
        pcs_plot(dataset[f"pc_{userargs['key_to_print']}"], pc_fdr_name, channel_to_print, psd.ch_names, info=psd.info)
    
    if userargs['print_noise']:
        noise_fdr_name = os.path.join(dataset['pf'].get_fdr_dict()['prep'], "ckpt", subject, f"noise_{userargs['key_to_print']}")
        ensure_dir(noise_fdr_name)
        for ch_name in channel_to_print:
            psd_plot(dataset[f"noise_{userargs['key_to_print']}"], resolution=userargs['resolution'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], picks=[ch_name], save_pth=os.path.join(noise_fdr_name, f"{ch_name}_psd.pdf"), dB=userargs['dB'])
            temp_plot(dataset[f"noise_{userargs['key_to_print']}"], ch_name, fs=fs, save_pth=os.path.join(noise_fdr_name, f"{ch_name}.pdf"), name=ch_name)
        
        psd_plot(dataset[f"noise_{userargs['key_to_print']}"], resolution=userargs['resolution'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], save_pth=os.path.join(noise_fdr_name, f"noise_psd.pdf"), picks=channel_to_print, dB=userargs['dB'])
    
    dataset['ckpt_raw'] = copy.deepcopy(dataset['raw'])
    if userargs['log_tracer']:
        if picks in ['eeg', 'all', 'data'] or 'eeg' in picks:
            dataset['tracer'].checkpoint(dataset['raw'].get_data(picks='eeg'), name=userargs['ckpt_name'])
            dataset['tracer'].checkpoint_psd(psd, name=userargs['ckpt_name'])
        else:
            log_or_print(f"Warning: No EEG channels found in picks '{picks}', skipping tracer checkpointing.")
    
    return dataset

def set_channel_montage(dataset, userargs):
    correct_sign = userargs.get('correct_sign', True)
    ds_name = userargs.get('ds_name', 'staresina')
    
    if ds_name == 'staresina':
        dpo_files = Study([
            os.path.join(dataset['pf'].get_fdr_dict()['base'], "sub-{subj}/eeg/sub-{subj}_ses-{ses}_run-{run}_{foo}rest{foo2}block-{block}.cdt.dpo"),
            os.path.join(dataset['pf'].get_fdr_dict()['base'], "sub-{subj}/eeg/sub-{subj}_ses-{ses}_run-{run}_block-{block}{foo}rest{foo2}.cdt.dpo"),
            os.path.join(dataset['pf'].get_fdr_dict()['base'], "sub-{subj}/ses-{ses}/eeg/sub-{subj}_ses-{ses}_run-{run}_block-{block}{foo1}rest{foo2}.cdt.dpo")
        ])
        subject = dataset['subject']
        subj_dict = parse_subj(subject, True)
        dpo = dpo_files.get(subj=subj_dict["subj"], ses=subj_dict["ses"], block=subj_dict["block"], run=subj_dict["run"])
        assert len(dpo) == 1
        dpo = dpo[0]
        with open(dpo, 'r') as f:
            dpo_content = f.read()
        dpo_content = re.sub(r"#.*?\n", "\n", dpo_content)  # Remove comments
        labels_list_match = re.search(r"LABELS START_LIST([\s\S]*?)LABELS END_LIST", dpo_content)
        assert labels_list_match is not None
        labels_data = labels_list_match.group(1).strip().splitlines()
        
        sensors_list_match = re.search(r"SENSORS START_LIST([\s\S]*?)SENSORS END_LIST", dpo_content)
        assert sensors_list_match is not None
        sensors_data = sensors_list_match.group(1).strip().splitlines()
        sensors = np.array([line.split() for line in sensors_data], dtype=np.float64)
        
        if correct_sign:
            sign_x = (sensors[labels_data.index('C6')][0] > sensors[labels_data.index('C5')][0]) * 2 - 1
            sign_y = (sensors[labels_data.index('Fz')][1] > sensors[labels_data.index('Pz')][1]) * 2 - 1
            sign_z = (sensors[labels_data.index('Cz')][2] > np.mean(sensors[:,2])) * 2 - 1
            sensors = [[sign_x*x,sign_y*y,sign_z*z] for x,y,z in sensors] 

        sensors = np.array(sensors)*1e-3
        ch_pos = {ch: loc for ch, loc in zip(labels_data, sensors)}
        
        ch_to_drop = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
        for ch in ch_to_drop:
            if ch in ch_pos:
                ch_pos.pop(ch)
        
        custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        dataset["raw"].set_montage(custom_montage)
    elif ds_name == 'irene':
        # from mne.channels._standard_montage_utils import _safe_np_loadtxt, _check_dupes_odict
        # options = dict(dtype=('U100', 'f4', 'f4', 'f4'))
        # fid_names = ('Nz', 'LPA', 'RPA')
        # ch_names, xs, ys, zs = _safe_np_loadtxt('/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/Standard-10-5-Cap385.sfp', **options)
        # # deal with "headshape"
        # mask = np.array([ch_name == 'headshape' for ch_name in ch_names], bool)
        # hsp = np.stack([xs[mask], ys[mask], zs[mask]], axis=-1)
        # mask = ~mask
        # pos = np.stack([xs[mask], ys[mask], zs[mask]], axis=-1)
        # ch_names = [ch_name for ch_name, m in zip(ch_names, mask) if m]
        # ch_pos = _check_dupes_odict(ch_names, pos)
        # del xs, ys, zs, ch_names
        # # no one grants that fid names are there.
        # nasion, lpa, rpa = [ch_pos.pop(n, None) for n in fid_names]

        # scale = 0.095 / np.median(np.linalg.norm(pos, axis=-1))
        # for value in ch_pos.values():
        #     value *= scale
        # nasion = nasion * scale if nasion is not None else None
        # lpa = lpa * scale if lpa is not None else None
        # rpa = rpa * scale if rpa is not None else None

        # montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='unknown', nasion=nasion, rpa=rpa, lpa=lpa, hsp=hsp)
        
        template = mne.io.read_raw_fif("/ohba/pi/knobre/irene/data_for_jize/clean/visit1/s02/s02_block1_curry_bcg_cleaner_ica_raw.fif")
        rename_dict = {}
        for ch in template.ch_names:
            new_ch = ch
            if ch.startswith('FP'):
                new_ch = new_ch.replace('FP', 'Fp')
            if ch.endswith('Z'):
                new_ch = new_ch.replace('Z', 'z')
            rename_dict[ch] = new_ch
        template.rename_channels(rename_dict)
        montage = template.get_montage()
        
        IRENE_RESCALE = np.array((0.8, 0.92, 1.1))
        LPA_CHANGE = np.array((5, -10, -40)) * 1e-3
        RPA_CHANGE = np.array((-5, -10, -40)) * 1e-3
        NAS_CHANGE = np.array((0, -5, -33)) * 1e-3
        
        for ch_dict in montage.dig:
            if ch_dict['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG:
                ch_dict['r'] *= IRENE_RESCALE
            
            elif ch_dict['kind'] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL:
                if ch_dict['ident'] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                    ch_dict['r'] += LPA_CHANGE
                elif ch_dict['ident'] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                    ch_dict['r'] += RPA_CHANGE
                elif ch_dict['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                    ch_dict['r'] += NAS_CHANGE
        
        dataset["raw"].set_montage(montage)

    return dataset

def crop_TR(dataset, userargs):
    """
    Crops the dataset to the TRs of the fMRI data.
    userargs{event_reference: bool} - If True, after cropping, the event would be overwritten to the event in dataset["raw"].
    """
    
    # event_reference = userargs.get('event_reference', False)
    # freq = userargs.get('freq', 5000)
    TR = userargs.get('TR', 1.14)
    tmin = userargs.get('tmin', -0.04*1.14)
    event_name = userargs.get('event_name', None)
    num_edge_TR = userargs.get('num_edge_TR', 0)

    freq = dataset['raw'].info['sfreq']
    if event_name is None:
        if 'tr_event_key' in dataset:
            event_name = dataset['tr_event_key']            
        else:
            event_name = '1200002'
            
        if isinstance(event_name, list):
            for event in event_name:
                if str(event) in mne.events_from_annotations(dataset['raw'])[1]:
                    event_name = event
                    break
            else:
                event_name = 'TR'
    # def crop_eeg_to_tr(eeg, change_onset=True):   
    def crop_eeg_to_tr(eeg, tmin, num_edge_TR=0):           
        try:
            trig = mne.events_from_annotations(eeg)[1][str(event_name)]
        except KeyError:
            trig = mne.events_from_annotations(eeg)[1]['TR']
        
        start_point = end_point = -1
        for timepoint, _, trig_value in mne.events_from_annotations(eeg)[0]:
            if trig_value == trig:
                if start_point == -1:
                    start_point = timepoint - eeg.first_samp
                end_point = timepoint+TR*freq - eeg.first_samp
        
        new_tmin = max(start_point/freq+tmin+num_edge_TR*TR, eeg.tmin)
        tmax = min(end_point/freq-num_edge_TR*TR, eeg.tmax)
        eeg = eeg.crop(tmin=new_tmin, tmax=tmax, include_tmax=False)  
        return eeg
    
    dataset["raw"] = crop_eeg_to_tr(dataset["raw"], tmin=tmin, num_edge_TR=num_edge_TR)
    return dataset 

def crop_by_epoch(dataset, userargs):
    """
    Crops the dataset to the epochs of the EEG data.
    """
    
    epoch_name = userargs.get('epoch_name', 'sim_ep')
    num_edge_epoch = userargs.get('num_edge_epoch', 0)
    
    epoch = dataset[epoch_name]
    events = copy.deepcopy(epoch.events)
    events = events[np.argsort(events[:, 0])]  # sort events by timepoint
    
    start_point = events[0,0] - dataset['raw'].first_samp
    end_point = events[-1,0] + epoch.tmax*dataset['raw'].info['sfreq'] - dataset['raw'].first_samp
    
    edge_time_crop = num_edge_epoch*(epoch.tmax-epoch.tmin)
    new_tmin = max(start_point/dataset['raw'].info['sfreq']+epoch.tmin, dataset['raw'].tmin) + edge_time_crop
    tmax = min(end_point/dataset['raw'].info['sfreq'], dataset['raw'].tmax) - edge_time_crop
    
    dataset["raw"] = dataset["raw"].crop(tmin=new_tmin, tmax=tmax, include_tmax=False)  
    return dataset

def set_channel_type_raw(dataset, userargs):
    remove_trigger = userargs.get('remove_trigger', True)
    
    dataset["raw"].set_channel_types({'VEOG': 'eog'})
    dataset["raw"].set_channel_types({'HEOG': 'eog'})
    dataset["raw"].set_channel_types({'EKG': 'ecg'})
    dataset["raw"].set_channel_types({'EMG': 'emg'})
    
    if 'Trigger' in dataset['raw'].ch_names:
        if remove_trigger:
            dataset['raw'].drop_channels(['Trigger'])
        else:
            dataset["raw"].set_channel_types({'Trigger': 'misc'})
    return dataset

def create_epoch(dataset, userargs):
    event = userargs.get('event', 'TR')
    tmin = userargs.get('tmin', -0.04*1.14)    # remember changing 1.14 to 0.07 if event = slice!
    tmax = userargs.get('tmax', 0.97*1.14)      # note that the 'tmax' is in a matlab style, i.e. tmax-tmin is not the length of the epoch, but +1 timepoint
    random = userargs.get('random', False)
    event_name = userargs.get('event_name', None)
    epoch_name_diy = userargs.get('epoch_name', None)   # if None, will be set to event + '_ep' or event + '_ep_rand' if random is True
    correct_trig = userargs.get('correct_trig', False)  # whether to correct the trigger event using pearson correlation. only works for 'TR' event.
    l_freq = userargs.get('l_freq', None)  # low frequency cutoff for the bandpass filter
    h_freq = userargs.get('h_freq', None)
    ssp = userargs.get('ssp', 0)
    
    
    if event == 'slice':
        if event_name is None:
            if 'tr_event_key' in dataset:
                event_name = dataset['tr_event_key']
            else:
                event_name = '1200002'
        if isinstance(event_name, list):
            for event in event_name:
                if str(event) in mne.events_from_annotations(dataset['raw'])[1]:
                    event_name = event
                    break
            else:
                event_name = 'TR'
        try:
            event_id = mne.events_from_annotations(dataset['raw'])[1][str(event_name)]
        except KeyError:
            event_id = mne.events_from_annotations(dataset['raw'])[1]['TR']

        if random:
            epoch_name = 'slice_ep_rand'
            
            tp_list = mne.events_from_annotations(dataset['raw'])[0]
            tp_list = tp_list[tp_list[:,2] == event_id][:,0]
            rand_tp_list = np.sort(np.random.choice(np.arange(np.min(tp_list), np.max(tp_list)), size=16*len(tp_list), replace=False))
            events = rand_tp_list.reshape(-1, 1)
        else:
            epoch_name = 'slice_ep'
            
            subject = dataset['subject']
            abnormal_mat_fp = os.path.join(dataset['pf'].fdr['slice'], f"{subject}.mat")
            if os.path.exists(abnormal_mat_fp):
                onset = loadmat(abnormal_mat_fp)['UniqueTiming']
            else:
                onset = np.arange(0, 0.07*16, 0.07)
            onset = (onset * dataset['raw'].info['sfreq']).astype(np.int64)
            
            slice_tp_list = []
            for timepoint, _, trig_value in mne.events_from_annotations(dataset['raw'])[0]:
                if trig_value == event_id:
                    slice_tp_list.append(timepoint + onset)
            events = np.concatenate(slice_tp_list).reshape(-1, 1)
        events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        event_id = 1
    elif event == 'TR':
        epoch_name = 'tr_ep' if not random else 'tr_ep_rand'
        if epoch_name in dataset:
            events = dataset[epoch_name].events
            events[: ,0] //= int(dataset[epoch_name].info['sfreq'] // dataset['raw'].info['sfreq'])
            events[:, 0] = events[:, 0].astype(np.int64)
            event_id = list(dataset[epoch_name].event_id.values())[0]
        else:
            if event_name is None:
                if 'tr_event_key' in dataset:
                    event_name = dataset['tr_event_key']
                else:
                    event_name = '1200002'
            if isinstance(event_name, list):
                for event in event_name:
                    if str(event) in mne.events_from_annotations(dataset['raw'])[1]:
                        event_name = event
                        break
                else:
                    event_name = 'TR'
            try:
                event_id = mne.events_from_annotations(dataset['raw'])[1][str(event_name)]
            except KeyError:
                event_id = mne.events_from_annotations(dataset['raw'])[1]['TR']


            events = mne.events_from_annotations(dataset['raw'])[0]
            if correct_trig:
                events = correct_trigger(dataset['raw'], events, event_id, tmin=tmin, tmax=tmax, template='mid', channel=0, hwin=3)
            if random:
                tr_tp_list = events[events[:,-1]==event_id][:,0]            
                rand_tp_list = np.sort(np.random.choice(np.arange(np.min(tr_tp_list), np.max(tr_tp_list)), size=len(tr_tp_list), replace=False))
                events = rand_tp_list.reshape(-1, 1)
                events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        
    elif event == 'He132':  # tmin and tmax are not used
        event_name_list = ['128', '132', '192', '196']
        if event_name is not None:
            event_id = mne.events_from_annotations(dataset['raw'])[1][str(event_name)]
        else:
            for event_name in event_name_list:
                if event_name in mne.events_from_annotations(dataset['raw'])[1]:
                    event_id = mne.events_from_annotations(dataset['raw'])[1][event_name]
                    break
            
        events = mne.events_from_annotations(dataset['raw'])[0]
        he_tp_list = events[events[:,-1]==event_id][:,0]
        time_diff = np.diff(he_tp_list)
        tmax = min(np.median(time_diff)*1.02, np.max(time_diff)) / dataset['raw'].info['sfreq']
        tmin = 0
        epoch_name = 'he_ep'
        
        if random:
            epoch_name = 'he_ep_rand'
            rand_tp_list = np.sort(np.random.choice(np.arange(np.min(he_tp_list), np.max(he_tp_list)), size=len(he_tp_list), replace=False))
            events = rand_tp_list.reshape(-1, 1)
            events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
    elif event == 'simulate':
        ### WARNING: random in this case represents the percentage of noise in the epoch timepoints, not the random sampling of the events.
        epoch_diff = tmax*dataset['raw'].info['sfreq']
        rand_range = int(epoch_diff*random)
        
        tp_list = np.arange(dataset['raw'].first_samp, dataset['raw'].last_samp, epoch_diff)
        if rand_range > 0:
            tp_list = tp_list + np.random.rand(-rand_range, rand_range, size=len(tp_list))
        events = tp_list.reshape(-1, 1).astype(np.int64)
        events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        epoch_name = 'sim_ep'
        event_id = 1
    else:
        raise ValueError(f"Event {event} not recognized.")

    # while True:
    #     if epoch_name in dataset:
    #         epoch_name = epoch_name + "_"
    #     else:
    #         break

    if epoch_name_diy is not None:
        epoch_name = epoch_name_diy
    if l_freq is not None or h_freq is not None:
        ep_raw = copy.deepcopy(dataset['raw'])
        ep_raw = ep_raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params={"order": 4, "ftype": "butter"})
    else:
        ep_raw = dataset['raw']
    dataset[epoch_name] = mne.Epochs(ep_raw, events=events, tmin=tmin, tmax=tmax, event_id=event_id, baseline=None, proj=False)
    
    if ssp > 0: 
        proj = mne.compute_proj_epochs(dataset[epoch_name], n_grad=0, n_mag=0, n_eeg=ssp, verbose=True)
        dataset['raw'].add_proj(proj)
        dataset['raw'].apply_proj()
    
    return dataset

def epoch_sw_pca(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    npc = userargs.get('npc', 3)
    window_length = userargs.get('window_length', 30)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    do_align = userargs.get('do_align', False)
    remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
    spurious_event = userargs.get('spurious_event', False)  # if True, epoch_key is used for removal, epoch_key + "_screener" is used for PC calculation
    pre_pad = userargs.get('pre_pad', 0.5)  # in percentage, the padding before the first epoch. 1-pre_pad is the padding after the last epoch.
    
    assert not do_align, "Alignment not implemented yet."
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)

    if spurious_event:
        raise NotImplementedError("Spurious event not implemented yet, please set spurious_event=False.")
        screener = dataset[f"{epoch_key}_screener"]
        epoch_std = orig_data[screener].std(dim=(1,2))  
        within_3_std = epoch_std < (epoch_std.mean()+3*epoch_std.std())
        screener = screener[within_3_std]
    
    pca_mean = torch.mean(orig_data, dim=2) * int(remove_mean)    # 29+#win, #ch
    det_orig_data = orig_data - pca_mean.unsqueeze(2)
    spurious_data = det_orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    if force_mean_pc0:
        pc0 = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
        detrended = spurious_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1)
    else: 
        U, S, _ = torch.pca_lowrank(spurious_data)
        all_pcs = U[..., :npc]*S[..., None, :npc]   # #win, #ch, len(ep), #pc
    
    
    pre_padding = int(pre_pad * (window_length-1))
    post_padding = window_length - pre_padding - 1
    if pre_padding > 0 and post_padding > 0:
        pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
        post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
        all_pcs = torch.cat([pre_padding, all_pcs, post_padding], dim=0)
    elif pre_padding > 0:
        pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
        all_pcs = torch.cat([pre_padding, all_pcs], dim=0)
    else:
        post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
        all_pcs = torch.cat([all_pcs, post_padding], dim=0)
    
    # padding = torch.repeat_interleave(all_pcs[0:1], window_length-1, dim=0)
    # all_pcs = torch.cat([padding, all_pcs], dim=0)    # 29+#win, #ch, len(ep), #pc
    
    if spurious_event:
        pass
    
    noise = lstsq(all_pcs, det_orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
    noise = (all_pcs @ noise)[...,0] + pca_mean.unsqueeze(-1)  # 29+#win, #ch, len(ep)  # [...,0] means squeezing the last dim, not squeeze() for the case #ch=1
    cleaned = np.array(orig_data - noise)
        
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset

def epoch_sw_pca_bievent(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'bcg_ep')
    npc = userargs.get('npc', 3)
    window_length = userargs.get('window_length', 10)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'even')
    
    orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))  # #ep, #ch, len(ep)
    
    assert window_length % 2 == 0
    pca_mean = torch.mean(orig_data, dim=2)
    spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    if force_mean_pc0:
        pc0 = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
        detrended = spurious_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1)
    else: 
        U, S, _ = torch.pca_lowrank(spurious_data)
        all_pcs = U[..., :npc]*S[..., None, :npc]   # #win, #ch, len(ep), #pc
    
    orig_data = torch.tensor(dataset[f"{epoch_key}"].get_data(picks=picks))  # #ep, #ch, len(ep)
    safe_idx = []
    spurious_event = dataset[epoch_key].events
    safe_event = dataset[f"{epoch_key}_safe"].events
    assert spurious_event.shape[0] == orig_data.shape[0], f"Spurious event shape {spurious_event.shape} does not match spurious data shape {orig_data.shape}."
    for idx, _ in enumerate(spurious_event):
        if spurious_event[idx, 0] in safe_event[:,0]:
            safe_idx.append(idx)
    safe_idx = torch.tensor(safe_idx)
    
    all_pcs_padded = torch.empty(orig_data.shape[0], *all_pcs.shape[1:], dtype=orig_data.dtype)  # #win, #ch, len(ep), #pc
    for idx in range(orig_data.shape[0]):
        dist = torch.abs(safe_idx-idx)
        window_idx = safe_idx[torch.argmin(dist)]
        window_idx = min(list(safe_idx).index(window_idx)-window_length//2, 0)
        window_idx = max(window_idx, all_pcs.shape[0]-1)
        all_pcs_padded[idx] = all_pcs[window_idx]
    
    noise = lstsq(all_pcs_padded, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
    noise = (all_pcs_padded @ noise)[...,0]  # 29+#win, #ch, len(ep)  # [...,0] means squeezing the last dim, not squeeze() for the case #ch=1
    cleaned = np.array(orig_data - noise)
        
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs_padded
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset

def epoch_aas_bievent_mask(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'bcg_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    # spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    
    unsafe_idx = []
    spurious_event = dataset[epoch_key].events
    safe_event = dataset[f"{epoch_key}_safe"].events
    assert spurious_event.shape[0] == orig_data.shape[0], f"Spurious event shape {spurious_event.shape} does not match spurious data shape {orig_data.shape}."
    for idx, _ in enumerate(spurious_event):
        if not (spurious_event[idx, 0] in safe_event[:,0]):
            unsafe_idx.append(idx)
    unsafe_idx = torch.tensor(unsafe_idx)

    all_pcs = torch.empty_like(orig_data)  # 29+#win, #ch, len(ep)
    for idx in range(orig_data.shape[0]):
        tmp_window_length = window_length
        while True:
            win_left = max(idx - tmp_window_length//2, 0)
            win_left = min(win_left, orig_data.shape[0] - tmp_window_length)
            win_idx = list(range(win_left, win_left + tmp_window_length))
            win_idx = [i for i in win_idx if i not in unsafe_idx]
            if len(win_idx) != 0:
                tmp_window_length = window_length
                break
            tmp_window_length *= 2
        all_pcs[idx] = torch.mean(orig_data[win_idx], dim=0)  # #ch, len(ep)

    all_pcs = all_pcs.unsqueeze(-1)  # 29+#win, #ch, len(ep), 1    

    # all_pcs = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
    
    # pre_padding = int(pre_pad * (window_length-1))
    # post_padding = window_length - pre_padding - 1

    # if pre_padding > 0 and post_padding > 0:
    #     pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
    #     post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
    #     all_pcs = torch.cat([pre_padding, all_pcs, post_padding], dim=0)
    # elif pre_padding > 0:
    #     pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
    #     all_pcs = torch.cat([pre_padding, all_pcs], dim=0)
    # else:
    #     post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
    #     all_pcs = torch.cat([all_pcs, post_padding], dim=0)

    if fit:
        noise = lstsq(all_pcs, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
        noise = (all_pcs @ noise)[...,0]
        cleaned = np.array(orig_data - noise)
    else:
        cleaned = np.array(orig_data - all_pcs.squeeze())
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset

def epoch_aas_bievent(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'bcg_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    # spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    
    safe_idx = []
    spurious_event = dataset[epoch_key].events
    safe_event = dataset[f"{epoch_key}_safe"].events
    assert spurious_event.shape[0] == orig_data.shape[0], f"Spurious event shape {spurious_event.shape} does not match spurious data shape {orig_data.shape}."
    for idx, _ in enumerate(spurious_event):
        if spurious_event[idx, 0] in safe_event[:,0]:
            safe_idx.append(idx)
    safe_idx = torch.tensor(safe_idx)

    all_pcs = torch.empty_like(orig_data)  # 29+#win, #ch, len(ep)
    for idx in range(orig_data.shape[0]):
        dist = torch.abs(safe_idx-idx)
        sorted_safe = safe_idx[torch.argsort(dist)]
        window_idx = sorted_safe[:window_length]
        all_pcs[idx] = torch.mean(orig_data[window_idx], dim=0)  # #ch, len(ep)

    all_pcs = all_pcs.unsqueeze(-1)  # 29+#win, #ch, len(ep), 1    

    # all_pcs = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
    
    # pre_padding = int(pre_pad * (window_length-1))
    # post_padding = window_length - pre_padding - 1

    # if pre_padding > 0 and post_padding > 0:
    #     pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
    #     post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
    #     all_pcs = torch.cat([pre_padding, all_pcs, post_padding], dim=0)
    # elif pre_padding > 0:
    #     pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
    #     all_pcs = torch.cat([pre_padding, all_pcs], dim=0)
    # else:
    #     post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
    #     all_pcs = torch.cat([all_pcs, post_padding], dim=0)

    if fit:
        noise = lstsq(all_pcs, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
        noise = (all_pcs @ noise)[...,0]
        cleaned = np.array(orig_data - noise)
    else:
        cleaned = np.array(orig_data - all_pcs.squeeze())
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset

def epoch_aas(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    pre_pad = userargs.get('pre_pad', 0.5)  # in percentage, the padding before the first epoch. 1-pre_pad is the padding after the last epoch.
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep

    all_pcs = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
    
    pre_padding = int(pre_pad * (window_length-1))
    post_padding = window_length - pre_padding - 1

    if pre_padding > 0 and post_padding > 0:
        pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
        post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
        all_pcs = torch.cat([pre_padding, all_pcs, post_padding], dim=0)
    elif pre_padding > 0:
        pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
        all_pcs = torch.cat([pre_padding, all_pcs], dim=0)
    else:
        post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
        all_pcs = torch.cat([all_pcs, post_padding], dim=0)

    if fit:
        noise = lstsq(all_pcs, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
        noise = (all_pcs @ noise)[...,0]
        cleaned = np.array(orig_data - noise)
    else:
        cleaned = np.array(orig_data - all_pcs.squeeze())
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset
    

def impulse_removal(dataset, userargs):
    picks = userargs.get('picks', 'all')
    thres = userargs.get('thres', 3.0)
    iteration = userargs.get('iteration', 1)  # number of iterations to perform impulse removal, 0 or negative means infinite iterations
    
    bad_exist = False
    data = dataset['raw'].get_data(picks=picks)  # #ch, #timepoints
    data_abs_diff = np.abs(data[:, 1:] - data[:, :-1])  # #ch, #timepoints-1
    avg, std = np.mean(data_abs_diff, axis=1), np.std(data_abs_diff, axis=1)  # #ch,
    thres = avg + thres * std  # #ch,
    
    mask = data_abs_diff > thres[:, None]  # #ch, #timepoints-1
    mask = np.concatenate([np.zeros((mask.shape[0], 1), dtype=bool), mask], axis=1)  # #ch, #timepoints
    
    interp_data = data.copy()
    for ch in range(data.shape[0]):
        bad_idx = np.where(mask[ch])[0]
        good_idx = np.where(~mask[ch])[0]
        if len(bad_idx) > 0:
            interp_data[ch, bad_idx] = np.interp(bad_idx, good_idx, data[ch, good_idx])
            bad_exist = True

    dataset['raw']._data[pick_indices(dataset['raw'], picks)] = interp_data  # Update the raw data with the interpolated data
    
    if not bad_exist or iteration == 1:
        return dataset
    else:
        userargs['iteration'] = iteration - 1
        return impulse_removal(dataset, userargs)    

def epoch_impulse_removal(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    overwrite = userargs.get('overwrite', 'new')
    picks = userargs.get('picks', 'all')
    thres = userargs.get('thres', 3.0)
    
    orig_data = dataset[epoch_key].get_data(picks=picks)  # #ep, #ch, len(ep)
    abs_diff_data = np.abs(orig_data[:, :, 1:] - orig_data[:, :, :-1])  # #ep, #ch, len(ep)-1
    avg, std = np.mean(abs_diff_data, axis=-1), np.std(abs_diff_data, axis=-1)
    thres = avg + thres * std  # #ep, #ch
    mask = abs_diff_data > thres[:, :, None]  # #ep, #ch, len(ep)-1
    interp_data = orig_data.copy()
    
    for ep in range(orig_data.shape[0]):
        for ch in range(orig_data.shape[1]):
            ep_data = orig_data[ep, ch, :]
            ep_mask = np.concatenate([mask[ep, ch, :], np.zeros((1,), dtype=bool)])  # #timepoints
            
            bad_idx  = np.flatnonzero(ep_mask)
            good_idx = np.flatnonzero(~ep_mask)
            
            if bad_idx.size > 0:
                interp_data[ep, ch, bad_idx] = np.interp(bad_idx, good_idx, ep_data[good_idx])
    
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], interp_data, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    return dataset
    

# def channel_pca(dataset, userargs):
#     npc = userargs.get('npc', 3)
#     picks = userargs.get('picks', 'eeg')
    
#     data = torch.tensor(dataset['raw'].get_data(picks=picks))  # #ch, #timepoints
#     pca_mean = torch.mean(data, dim=0)  # #timepoints
#     detrended = data - pca_mean.unsqueeze(0)  # #ch, #timepoints
#     U, S, _ = torch.pca_lowrank(detrended.T)   # [#timepoints, q]; [q,]
#     all_pcs = U[:, :npc]*S[None, :npc]
    
# def epoch_pca_scipy(dataset, userargs):
#     epoch_key = userargs.get('epoch_key', 'tr_ep')
#     npc = userargs.get('npc', 3)
#     force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
#     picks = userargs.get('picks', 'eeg')
#     overwrite = userargs.get('overwrite', 'obs')
#     remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
#     spurious_event = userargs.get('spurious_event', False)  # if True, epoch_key is used for removal, epoch_key + "_safe" is used for PC calculation
#     screen_high_power = userargs.get('screen_high_power', None)  # if True, the epochs with high power would not be used for PC calculation. If None, no screening is performed. If false, only the epochs with high power would be used for PC calculation.
    
    
#     if spurious_event:
#         # screener = dataset[f"{epoch_key}_screener"]
#         # orig_data = orig_data[screener]
#         orig_data = dataset[f"{epoch_key}_safe"].get_data(picks=picks) # #ep, #ch, len(ep)
#     else:
#         orig_data = dataset[epoch_key].get_data(picks=picks) # #ep, #ch, len(ep)
    
#     if not screen_high_power is None:
#         epoch_power = np.sum(orig_data**2, axis=(1,2)) # #ep
#         power_med = np.median(epoch_power)
#         power_mad = np.median(np.abs(epoch_power - power_med))
#         threshold = power_med + 3*power_mad
#         orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]
    
#     orig_data = np.transpose(orig_data, (1, 2, 0))  # (#ch, #len(ep), #ep)
#     pca_mean = np.mean(orig_data, axis=1) * int(remove_mean)    # (#ch, #ep)
#     dirty_data = orig_data - pca_mean[:, None, :]  # (#ch, #len(ep), #ep)
    
#     pca = PCA(n_components=npc) if not force_mean_pc0 else PCA(n_components=npc-1)
    
#     all_pcs = np.zeros((dirty_data.shape[0], dirty_data.shape[1], npc))  # (#ch, #len(ep), #pc)
#     for ch in range(dirty_data.shape[0]):
#         if force_mean_pc0:
#             pc0 = np.mean(dirty_data[ch], axis=-1, keepdims=True)    # len(ep), 1
#             detrended = dirty_data[ch] - pc0    # (len(ep), #ep)
#             pcs = pca.fit_transform(detrended)
#             pcs = np.concatenate([pc0, pcs], axis=-1)  # (len(ep), #pc)
#         else:
#             pcs = pca.fit_transform(dirty_data[ch])
#         all_pcs[ch] = pcs
    
#     del orig_data, dirty_data, pca_mean
#     all_pcs = torch.from_numpy(all_pcs)
#     orig_data = dataset[epoch_key].get_data(picks=picks)  # #ep, #ch, len(ep)
#     orig_data = torch.from_numpy(np.transpose(orig_data, (1, 2, 0))) # (#ch, len(ep), #ep)
#     pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)
#     dirty_data = orig_data - pca_mean[:, None, :]  # (#ch, #len(ep), #ep)
#     noise = lstsq(all_pcs, dirty_data)[0]   # (#ch, #pc, #ep)
#     noise = all_pcs @ noise + pca_mean[:, None, :]  # (#ch, #len(ep), #ep)
#     cleaned = np.array(orig_data - noise).transpose(2, 0, 1)  # #ep, #ch, len(ep)
#     pc_name = f"pc_{epoch_key}"
#     noise_name = f"noise_{epoch_key}"
#     picks_name = f"picks_{epoch_key}"
        
#     # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
#     # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
#     # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."

#     while True:
#         if pc_name in dataset:
#             pc_name = pc_name + "_"
#             continue
#         if noise_name in dataset:
#             noise_name = noise_name + "_"
#             continue
#         if picks_name in dataset:
#             picks_name = picks_name + "_"
#             continue
#         break
        
#     dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
        
#     dataset[pc_name] = all_pcs
#     dataset[picks_name] = picks
#     dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
#     dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
#     dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)
#     return dataset

# def epoch_pca_scipy_flatten(dataset, userargs):
#     epoch_key = userargs.get('epoch_key', 'tr_ep')
#     npc = userargs.get('npc', 3)
#     force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
#     picks = userargs.get('picks', 'eeg')
#     overwrite = userargs.get('overwrite', 'obs')
#     remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
#     ignore_spurious_event = userargs.get('ignore_spurious_event', False)  # if True, epoch_key is used for removal, epoch_key + "_screener" is used for PC calculation
#     screen_high_power = userargs.get('screen_high_power', None)  # if True, the epochs with high power would not be used for PC calculation. If None, no screening is performed. If false, only the epochs with high power would be used for PC calculation.
    
#     if ignore_spurious_event:
#         # screener = dataset[f"{epoch_key}_screener"]
#         # orig_data = orig_data[screener]
#         orig_data = dataset[f"{epoch_key}_safe"].get_data(picks=picks) # #ep, #ch, len(ep)
#     else:
#         orig_data = dataset[epoch_key].get_data(picks=picks) # #ep, #ch, len(ep)
    
#     if not screen_high_power is None:
#         epoch_power = np.sum(orig_data**2, axis=(1,2)) # #ep
#         power_med = np.median(epoch_power)
#         power_mad = np.median(np.abs(epoch_power - power_med))
#         threshold = power_med + 3*power_mad
#         orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]
    
#     reshaped_data = orig_data.reshape(-1, orig_data.shape[-1]).T  # len(ep), #ch*#ep
#     pca_mean = np.mean(reshaped_data, axis=0) * int(remove_mean)    # (#ch*#ep)
#     dirty_data = reshaped_data - pca_mean[None, :]
#     pca = PCA(n_components=npc) if not force_mean_pc0 else PCA(n_components=npc-1)
    
#     if force_mean_pc0:
#         pc0 = np.mean(reshaped_data, axis=1, keepdims=True)
#         detrended = reshaped_data - pc0
#         all_pcs = pca.fit_transform(detrended)
#         all_pcs = np.concatenate([pc0, all_pcs], axis=-1) # (len(ep), #pc)
#     else:
#         all_pcs = pca.fit_transform(reshaped_data)  # (len(ep), #pc)
    
#     del orig_data, reshaped_data, dirty_data, pca_mean
#     orig_data = dataset[epoch_key].get_data(picks=picks)  # #ep, #ch, len(ep)
#     # trans_data = np.transpose(orig_data, (1, 2, 0))
#     # trans_data = trans_data.reshape(trans_data.shape[1], -1)   # len(ep), #ch*#ep,
#     trans_data = orig_data.reshape(-1, orig_data.shape[-1]).T  # len(ep), #ch*#ep
#     pca_mean = np.mean(trans_data, axis=0) * int(remove_mean)    # (#ch*#ep)
#     dirty_data = trans_data - pca_mean[None, :]
    
#     noise = scipy_lstsq(all_pcs, dirty_data)[0]   # #pc, #ch*#ep
#     noise = all_pcs @ noise + pca_mean[None, :]  # len(ep), #ch*#ep
#     cleaned = np.array(trans_data - noise).T.reshape(*orig_data.shape)
#     pc_name = f"pc_{epoch_key}"
#     noise_name = f"noise_{epoch_key}"
#     picks_name = f"picks_{epoch_key}"
        
#     # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
#     # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
#     # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."

#     while True:
#         if pc_name in dataset:
#             pc_name = pc_name + "_"
#             continue
#         if noise_name in dataset:
#             noise_name = noise_name + "_"
#             continue
#         if picks_name in dataset:
#             picks_name = picks_name + "_"
#             continue
#         break
        
#     dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
        
#     dataset[pc_name] = np.repeat(all_pcs[None], orig_data.shape[1], axis=0)
#     dataset[picks_name] = picks
#     dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
#     dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
#     dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)
#     return dataset

    
def epoch_pca(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    npc = userargs.get('npc', 3)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'obs')
    remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
    pc_from_spurious = userargs.get('pc_from_spurious', True)  # if True, the PC is calculated from the spurious events, else it is calculated from the safe epochs.
    apply_to_spurious = userargs.get('apply_to_spurious', True)  # if True, the PC is applied to the spurious events, else it is not.
    screen_high_power = userargs.get('screen_high_power', None)  # if True, the epochs with high power would not be used for PC calculation. If None, no screening is performed. If false, only the epochs with high power would be used for PC calculation.
    
    
    if pc_from_spurious:
        orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks)) # #ep, #ch, len(ep)
    else:
        orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))
        # screener = dataset[f"{epoch_key}_screener"]
        # orig_data = orig_data[screener]
    
    if not screen_high_power is None:
        epoch_power = torch.sum(orig_data**2, dim=(1,2)) # #ep
        power_med = epoch_power.median()
        power_mad = torch.median(torch.abs(epoch_power - power_med))
        threshold = power_med + 3*power_mad
        orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]
        
    # orig_data = orig_data.reshape(*orig_data.shape[1:], orig_data.shape[0])
    orig_data = orig_data.permute(1, 2, 0)  # #ch, len(ep), #ep
    
    # epoch_std = orig_data.std(dim=(0,1))
    # normal_ep = epoch_std < (epoch_std.mean() + 3*epoch_std.std())
    # orig_data = orig_data[..., normal_ep]  # #ch, len(ep), #ep
    
    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean.unsqueeze(1)
    if force_mean_pc0:  # TODO: fit is false here, but when fitting, fit becomes true. could be problematic.
        pc0 = torch.mean(dirty_data, dim=2).unsqueeze(-1)  # #ch, len(ep), 1
        detrended = dirty_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)   # #ch, len(ep), q;    # #ch, q
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1) # #ch, len(ep), #pc
    else:   
        U, S, _ = torch.pca_lowrank(dirty_data)   # #ch, len(ep), q;    # #ch, q
        all_pcs = U[..., :npc]*S[..., None, :npc]
        
    del orig_data, dirty_data, U, S  # free memory
    if apply_to_spurious:
        orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # #ep, #ch, len(ep)
    else:
        orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))
    orig_data = orig_data.permute(1, 2, 0)  # #ch, len(ep), #ep
    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean.unsqueeze(1)  # #ch, len(ep), #ep
    noise = lstsq(all_pcs, dirty_data)[0]   # #ch, #pc, #ep
    noise = all_pcs @ noise + pca_mean.unsqueeze(1)  # #ch, len(ep), #ep
    
    cleaned = np.array((orig_data - noise).permute(2, 0, 1))
        
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"
        
    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."

    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
        
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)
    return dataset
    # if method == 'mne':
    #     ecg = find_ecg_events(dataset['raw'], ch_name=bcg_name, l_freq=5, h_freq=15)
    #     if ecg[0].size == 0:
    #         ecg = find_ecg_events(dataset['raw'], ch_name=None, l_freq=5, h_freq=15)
    #         print("Warning: No R peaks detected. Please check the ECG channel. Using other channels for reference.")
    #         if ecg[0].size == 0:
    #             raise AssertionError("No R peaks detected. Please check the data.")
    #     ecg = np.unique(ecg[0], axis=0)
    #     if correct:
    #         raise NotImplementedError("Correction for MNE method is not implemented yet.")
    #         ecg, spurious_ecg = qrs_correction(ecg, dataset['raw'], dataset['raw'].get_data(picks='EKG').squeeze(), new_event_idx=999)
    # elif method == 'kteo':
        # else:
    #     ecg = QRSDetector(dataset['raw'], ch_name=bcg_name, l_freq=l_freq, h_freq=h_freq).get_events(correction=correct, method=method)
    #     raise NotImplementedError(f"Method {method} not implemented for QRS detection.")

def qrs_detect(dataset, userargs):
    delay = userargs.get('delay', 0.0)  # if use EKG, delay is better to be 0.21, if use EEG, use 0.0
    bcg_name = userargs.get('bcg_name', 'pca')
    bcg2raw = userargs.get('bcg2raw', True)  # if True and bcg_name!=ECG, the calculated BCG epochs will be added to the raw data as a new channel.
    l_freq = userargs.get('l_freq', None)
    h_freq = userargs.get('h_freq', None)
    # method = userargs.get('method', 'kteo')
    random = userargs.get('random', False)
    ssp = userargs.get('ssp', 0)
    epoch_len = userargs.get('epoch_len', 1.0)  # if smaller than 2, denotes the epoch length in multiples of the median R-R interval, otherwise in ms
    corr = userargs.get('corr', [0.8, 0.4, 5])
    
    assert corr[0] > corr[1], "corr_start should be greater than corr_end."
    
    fs = dataset['raw'].info['sfreq']
    kteo, ecg_data = kteager_detect(dataset["raw"], filt_emg=False, filt_kteo=True, picks=bcg_name, l_freq=5, h_freq=15)
    if bcg2raw:
        assert bcg_name not in ['ECG', 'EKG', 'ecg'], f"bcg2raw could only be true if a new ECG channel is created based on other channels, currently {bcg_name} is used as the ECG channel. Please use a different source."
        new_info = mne.create_info(ch_names=['AECG'], sfreq=fs, ch_types=['ecg'])
        ecg_chs = mne.pick_types(dataset["raw"].info, ecg=True, exclude=[])
        ecg_names = [dataset["raw"].ch_names[p] for p in ecg_chs]
        dataset["raw"].set_channel_types({name: 'misc' for name in ecg_names})
        dataset['raw'].add_channels([mne.io.RawArray(ecg_data[None, :], new_info, first_samp=dataset['raw'].first_samp)], force_update_info=True)
    
    peaks = np.array(panPeakDetect(kteo, fs))
    peaks += dataset['raw'].first_samp
    ecg = np.column_stack([peaks, np.zeros(len(peaks)), 999*np.ones(len(peaks))]).astype(np.int64)

    safe_ecg, spurious_ecg = qrs_correction(ecg, dataset['raw'], ecg_data, (10*(len(mne.pick_types(dataset['raw'].info, eeg=True, exclude='bads'))+1)//2) / dataset['raw'].info['sfreq'] / 0.8, new_event_idx=999, corr_thres=corr)
    
    r_list = spurious_ecg[:,0]
    half_ep_size = (np.median(np.diff(r_list)) * epoch_len / 2 / dataset['raw'].info['sfreq']) if epoch_len < 2 else epoch_len / 2 / 1000.0
    if l_freq is not None or h_freq is not None:
        ep_raw = copy.deepcopy(dataset['raw']).filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params={'ftype': 'butter', 'order': 4})
    else:
        ep_raw = dataset['raw']
    dataset['bcg_ep_safe'] = mne.Epochs(ep_raw, events=safe_ecg, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=999, baseline=None, proj=False)
    dataset['bcg_ep'] = mne.Epochs(ep_raw, events=spurious_ecg, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=999, baseline=None, proj=False)

    if random:
        # randomly sample same number of epochs, with the same length of bcg_ep
        rand_tp_list = np.sort(np.random.choice(np.arange(np.min(r_list), np.max(r_list)), size=len(r_list), replace=False))
        
        events = rand_tp_list.reshape(-1, 1)
        events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        dataset['bcg_ep_rand'] = mne.Epochs(dataset['raw'], events=events, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=1, baseline=None, proj=False)
    
    ep = dataset['bcg_ep_safe'] if not random else dataset['bcg_ep_rand']

    if ssp > 0:
        proj = mne.compute_proj_epochs(ep, n_grad=0, n_mag=0, n_eeg=ssp, verbose=True)
        
        dataset['raw'].add_proj(proj)
        dataset['raw'].apply_proj()
    
    return dataset
    
def bcg_removal(dataset, userargs):
    method = userargs.get('method', 'obs')
    npc = userargs.get('npc', 3)
    overwrite = userargs.get('overwrite', 'obs')
    filt = userargs.get('filt', [1, 40])
    remove_mean = userargs.get('remove_mean', True)
    picks = userargs.get('picks', 'eeg')
    filt_fit_target = userargs.get('filt_fit_target', False)

    if method == 'obs':
        dataset = epoch_pca(dataset, userargs={'epoch_key': 'bcg_ep', 'npc': npc, 'force_mean': True, 'overwrite': overwrite, 'filt': filt, 'tmin':dataset['bcg_ep'].tmin, 'remove_mean': remove_mean, 'picks': picks, 'filt_fit_target': filt_fit_target})
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
    return dataset

def bcg_ep_ica(dataset, userargs):
    """perform ICA on the BCG epochs to remove artifacts.

    This function applies ICA to the BCG epochs in the dataset, excluding components based on their explained variance ratio.
    It modifies the raw data in the dataset by applying the ICA solution.

    Args:
        dataset (_type_): _description_
        userargs (_type_): _description_

    Returns:
        _type_: _description_
    """
    picks = userargs.get('picks', 'eeg')
    seed = userargs.get('seed', 42)
    max_iter = userargs.get('max_iter', 'auto')
    n_components = userargs.get('n_components', .999) # number of components to keep, default is 0.999, which means keep all components that explain at least 99.9% of the variance
    qrs_event_id = userargs.get('qrs_event_id', 999)
    downsample = userargs.get('downsample', 1)    # only set this if the "bcg_ep" is not downsampled!!!
    flatten_epoch_frange = userargs.get('flatten_epoch_frange', [1,40])  # the flattened epochs will be band pass filtered to the specified frequency range, e.g. [1, 40] for 1-40Hz bandpass filtering
    foi = userargs.get('freq_of_interest', [6, 12])  # the frequency range where you're sure not expecting any peaks within. default set to [6,12] Hz
    peak_threshold = userargs.get('peak_threshold', 3) # threshold for peak detection in psd, default is 3, which means only peaks with at least 3 times the value of the mean of the surrounding frequencies will be considered as peaks.
    ctps_threshold = userargs.get('ctps_threshold', 0.15)
    
    assert 'bcg_ep' in dataset, "Please run qrs_detect first to create bcg_ep."
    
    ev = copy.deepcopy(dataset["bcg_ep"].events)
    
    ev[:,0] = ev[:,0] / downsample
    ev = ev.astype(np.int64)
    bcg_ep = mne.Epochs(dataset['raw'], events=ev, tmin=dataset["bcg_ep"].tmin, tmax=dataset["bcg_ep"].tmax, event_id=qrs_event_id, baseline=None, proj=False)
    bcg_ep.load_data()  # Ensure the data is loaded before applying ICA
    
    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=seed)
    ica.fit(bcg_ep)
    
    exclude_list = []
    data = ica._transform_epochs(bcg_ep, concatenate=True)
    data = mne.io.RawArray(data, mne.create_info(data.shape[0], sfreq=dataset['raw'].info['sfreq'], ch_types='eeg'))
    psd = data.compute_psd(fmin=flatten_epoch_frange[0], fmax=flatten_epoch_frange[1], n_fft=int(np.round(dataset['raw'].info['sfreq']*20)))  # 20: resolution of 20 bins within 1Hz, 0.05 later = 1/20, 0.1 = 2/20
    
    epoch_freq = 1 / (bcg_ep.tmax-bcg_ep.tmin+1/dataset['raw'].info['sfreq'])
    
    ratio = []
    for idx in range(int(np.floor(foi[0]/epoch_freq)), int(np.ceil(foi[1]/epoch_freq))):
        ratio.append(psd.get_data(fmin=idx*epoch_freq-0.05, fmax=idx*epoch_freq+0.05).max(axis=1) / psd.get_data(fmin=(idx-1)*epoch_freq+0.1, fmax=(idx+1)*epoch_freq-0.1).mean(axis=1))
    ratio = np.array(ratio)
    exclude_list = np.unique(np.where(np.array(ratio)>peak_threshold)[1])
    
    if ctps_threshold is not None:
        # exclude_list.extend(np.where(ica.find_bads_ecg(bcg_ep)[1]>ctps_threshold)[0])
        exclude_list = np.union1d(exclude_list, ica.find_bads_ecg(bcg_ep, threshold=ctps_threshold)[0])
    # exclude_list = np.unique(exclude_list)    
    
    ica.exclude = exclude_list
    dataset['raw_before_ica'] = copy.deepcopy(dataset['raw'])
    dataset['raw'] = ica.apply(dataset['raw'])
    return dataset