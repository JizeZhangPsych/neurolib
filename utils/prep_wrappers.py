import os, re, math, random, copy, pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch.linalg import lstsq

import mne
from mne.preprocessing import find_ecg_events
from osl_ephys.report.preproc_report import plot_channel_dists # usage: plot_channel_dists(raw, savebase)
from .util import ensure_dir, proc_userargs
from .eeg import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, filename2subj, HeteroStudy as Study, Pathfinder, find_spurious_channels, pcs_plot
from .qrs import qrs_correction, QRSDetector


def initialize(dataset, userargs):
    dataset['pf'] = Pathfinder(**userargs)
    
    if 'Trigger' in dataset['raw'].ch_names:
        dataset['raw'].drop_channels(['Trigger'])
    
    subject = filename2subj(dataset['raw'].filenames[0])
    if subject == '2111':   # radiographer error, one more session recorded after 400s
        try:
            dataset['raw'] = dataset['raw'].crop(tmin=0, tmax=400)  
        except ValueError:
            print("Warning: Subject 2111 has no data after 400s, so no cropping is needed.")
    if subject == '4121':   # Accidental overwriting of resting stage EEG file after computer prompted an overwriting towards the end of the recording. So a very small file is recorded.
        raise Exception("Subject 4121 has its eeg file corrupted: Accidental overwriting of resting stage EEG file after computer prompted an overwriting towards the end of the recording. So a very small file is recorded.")
    # 3112
    
    dataset['raw'].drop_channels(['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2'], on_missing='warn')
    print("Warning: F11, F12, FT11, FT12, Cb1, Cb2 are dropped from the raw data, as no gel is used in these channels.")
    return dataset

def debug_init(dataset, userargs):
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
        'res_mult': 32,
        'max_freq': 50,
        'qrs_event': False,
        'key_to_print': None,
        'always_print': ['EKG'],    # must be name, 'eeg' is not allowed
        'std_channel_pick': 'eeg',
        'print_pcs': True,
        'print_noise': True,
        'ds_name': 'staresina'
    }
    userargs = proc_userargs(userargs, default_args)
    
    fs = dataset['raw'].info['sfreq']
    picks = dataset[f"picks_{userargs['key_to_print']}"] if f"picks_{userargs['key_to_print']}" in dataset else 'eeg'
    subject = filename2subj(dataset['raw'].filenames[0], ds_name=userargs['ds_name'])
    save_fdr = os.path.join(dataset['pf'].get_fdr_dict()['prep'], "ckpt", subject, userargs['ckpt_name'])
    ensure_dir(save_fdr)
    
    if userargs['key_to_print'] is None:
        userargs['print_noise'] = userargs['print_pcs'] = False
    
    psd = psd_plot([dataset['raw']], [userargs['ckpt_name']], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], save_pth=os.path.join(save_fdr, f"psd.png"), picks=picks)
    std = np.mean(np.std(dataset['raw'].get_data(picks=userargs['std_channel_pick'], reject_by_annotation='omit'), axis=1))
    plot_channel_dists(dataset['raw'], os.path.join(save_fdr, f"std={std:.4e}.png"))
    
    def print_ch(ch_name):
        extra_str = "temp"
        print_fdr = os.path.join(save_fdr, extra_str)
        ensure_dir(print_fdr)
        
        psd_plot([dataset['raw']], [ch_name], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], picks=ch_name, save_pth=os.path.join(print_fdr, f"{ch_name}_psd.png"))
        temp_plot(dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}.png"), name=ch_name)
        temp_plot(dataset['raw'], ch_name, fs=fs, start=100*fs, length=10*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_100-110.png"),name=ch_name)
        if 'ckpt_raw' in dataset:
            if dataset['ckpt_raw'].info['sfreq'] != dataset['raw'].info['sfreq']:
                dataset['ckpt_raw'].resample(dataset['raw'].info['sfreq'])
            if not userargs['qrs_event']:
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}_diff.png"),name=ch_name)
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=100*fs, length=10*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_100-110_diff.png"), name=ch_name)
            else:
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}_diff.png"),name=ch_name, events=dataset['bcg_ep'].events, event_id=999)
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=100*fs, length=10*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_100-110_diff.png"), name=ch_name, events=dataset['bcg_ep'].events, event_id=999)
    
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
            psd_plot([dataset[f"noise_{userargs['key_to_print']}"]], [ch_name], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], picks=[ch_name], save_pth=os.path.join(noise_fdr_name, f"{ch_name}_psd.png"))
            temp_plot(dataset[f"noise_{userargs['key_to_print']}"], ch_name, fs=fs, save_pth=os.path.join(noise_fdr_name, f"{ch_name}.png"), name=ch_name)
        
        psd_plot([dataset[f"noise_{userargs['key_to_print']}"]], ['noise'], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], save_pth=os.path.join(noise_fdr_name, f"noise_psd.png"), picks=channel_to_print)
    
    dataset['ckpt_raw'] = copy.deepcopy(dataset['raw'])
    return dataset

def set_channel_montage(dataset, userargs):
    correct_sign = userargs.get('correct_sign', True)
    dpo_files = Study([
        os.path.join(dataset['pf'].get_fdr_dict()['base'], "sub-{subj}/eeg/sub-{subj}_ses-{ses}_run-{run}_{foo}rest{foo2}block-{block}.cdt.dpo"),
        os.path.join(dataset['pf'].get_fdr_dict()['base'], "sub-{subj}/eeg/sub-{subj}_ses-{ses}_run-{run}_block-{block}{foo}rest{foo2}.cdt.dpo"),
        os.path.join(dataset['pf'].get_fdr_dict()['base'], "sub-{subj}/ses-{ses}/eeg/sub-{subj}_ses-{ses}_run-{run}_block-{block}{foo1}rest{foo2}.cdt.dpo")
    ])
    subject = filename2subj(dataset['raw'].filenames[0])
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
    custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    dataset["raw"].set_montage(custom_montage)

    return dataset

def crop_TR(dataset, userargs):
    """
    Crops the dataset to the TRs of the fMRI data.
    userargs{event_reference: bool} - If True, after cropping, the event would be overwritten to the event in dataset["raw"].
    """
    
    # event_reference = userargs.get('event_reference', False)
    freq = userargs.get('freq', 5000)
    TR = userargs.get('TR', 1.14)
    tmin = userargs.get('tmin', -0.04*1.14)

    # def crop_eeg_to_tr(eeg, change_onset=True):   
    def crop_eeg_to_tr(eeg):   
        try:
            trig = mne.events_from_annotations(eeg)[1]['1200002']
        except KeyError:
            trig = mne.events_from_annotations(eeg)[1]['TR']
        
        start_point = end_point = -1
        for timepoint, _, trig_value in mne.events_from_annotations(eeg)[0]:
            if trig_value == trig:
                if start_point == -1:
                    start_point = timepoint - eeg.first_samp
                end_point = timepoint+TR*freq - eeg.first_samp
                
        eeg = eeg.crop(tmin=start_point/freq+tmin, tmax=end_point/freq)  
        return eeg
    
    dataset["raw"] = crop_eeg_to_tr(dataset["raw"])
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
    tmax = userargs.get('tmax', 0.97*1.14)
    random = userargs.get('random', False)
    
    if event == 'slice':
        try:
            event_id = mne.events_from_annotations(dataset['raw'])[1]['1200002']
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
            
            subject = filename2subj(dataset['raw'].filenames[0])
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
        try:
            event_id = mne.events_from_annotations(dataset['raw'])[1]['1200002']
        except KeyError:
            event_id = mne.events_from_annotations(dataset['raw'])[1]['TR']
        events = mne.events_from_annotations(dataset['raw'])[0]
        epoch_name = 'tr_ep'
        
        if random:
            epoch_name = 'tr_ep_rand'
            tr_tp_list = events[events[:,-1]==event_id][:,0]            
            rand_tp_list = np.sort(np.random.choice(np.arange(np.min(tr_tp_list), np.max(tr_tp_list)), size=len(tr_tp_list), replace=False))
            events = rand_tp_list.reshape(-1, 1)
            events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        
    elif event == 'He132':  # tmin and tmax are not used
        try:
            event_id = mne.events_from_annotations(dataset['raw'])[1]['132']
        except KeyError:
            event_id = mne.events_from_annotations(dataset['raw'])[1]['128']
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

    while True:
        if epoch_name in dataset:
            epoch_name = epoch_name + "_"
        else:
            break

    dataset[epoch_name] = mne.Epochs(dataset['raw'], events=events, tmin=tmin, tmax=tmax, event_id=event_id, baseline=None, proj=False)
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
    
    assert not do_align, "Alignment not implemented yet."
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
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
        U, S, _ = torch.pca_lowrank(torch.Tensor(spurious_data))
        all_pcs = U[..., :npc]*S[..., None, :npc]   # #win, #ch, len(ep), #pc
    
    padding = torch.repeat_interleave(all_pcs[0:1], window_length-1, dim=0)
    all_pcs = torch.cat([padding, all_pcs], dim=0)    # 29+#win, #ch, len(ep), #pc
    
    noise = lstsq(all_pcs, det_orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
    noise = (all_pcs @ noise)[...,0] + pca_mean.unsqueeze(-1)  # 29+#win, #ch, len(ep)  # [...,0] means squeezing the last dim, not squeeze() for the case #ch=1
    cleaned = np.array(orig_data - noise)
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    # if pc_name in dataset:
        # pc_name = pc_name + "_"
    # if noise_name in dataset:
        # noise_name = noise_name + "_"
    # if picks_name in dataset:
        # picks_name = picks_name + "_"
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset
    

def epoch_pca(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    npc = userargs.get('npc', 3)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'obs')
    remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks)) # #ep, #ch, len(ep)
    orig_data = orig_data.reshape(*orig_data.shape[1:], orig_data.shape[0])  # #ch, len(ep), #ep
    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    spurious_data = orig_data - pca_mean.unsqueeze(1)
    if force_mean_pc0:
        pc0 = torch.mean(spurious_data, dim=2).unsqueeze(-1)  # #ch, len(ep), 1
        detrended = spurious_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)   # #ch, len(ep), q;    # #ch, q
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1) # #ch, len(ep), #pc
    else:   
        U, S, _ = torch.pca_lowrank(spurious_data)   # #ch, len(ep), q;    # #ch, q
        all_pcs = U[..., :npc]*S[..., None, :npc]
        
    noise = lstsq(all_pcs, spurious_data)[0]   # #ch, #pc, #ep
    noise = all_pcs @ noise + pca_mean.unsqueeze(1)  # #ch, len(ep), #ep
    cleaned = np.array((orig_data - noise).reshape(orig_data.shape[2], *orig_data.shape[:2]))
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"
        
    assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    # if pc_name in dataset:
        # pc_name = pc_name + "_"
    # if noise_name in dataset:
        # noise_name = noise_name + "_"
    # if picks_name in dataset:
        # picks_name = picks_name + "_"
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
        
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)
    return dataset


def qrs_detect(dataset, userargs):
    delay = userargs.get('delay', 0.21)
    bcg_name = userargs.get('bcg_name', 'EKG')
    l_freq = userargs.get('l_freq', 7)
    h_freq = userargs.get('h_freq', 40)
    correct = userargs.get('correct', True)
    method = userargs.get('method', 'mne')
    random = userargs.get('random', False)
    
    if method == 'mne':
        ecg = find_ecg_events(dataset['raw'], ch_name=bcg_name, l_freq=l_freq, h_freq=h_freq)
        if ecg[0].size == 0:
            ecg = find_ecg_events(dataset['raw'], ch_name=None, l_freq=l_freq, h_freq=h_freq)
            print("Warning: No R peaks detected. Please check the ECG channel. Using other channels for reference.")
            if ecg[0].size == 0:
                raise AssertionError("No R peaks detected. Please check the data.")
        ecg = np.unique(ecg[0], axis=0)
        if correct:
            ecg = qrs_correction(ecg, dataset['raw'], new_event_idx=999)
    else:
        ecg = QRSDetector(dataset['raw'], ch_name=bcg_name, l_freq=l_freq, h_freq=h_freq).get_events(correction=correct, method=method)
    
    r_list = ecg[:,0]
    half_ep_size = np.median(np.diff(r_list)) * 0.75 / dataset['raw'].info['sfreq']
 
    dataset['bcg_ep'] = mne.Epochs(dataset['raw'], events=ecg, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=999, baseline=None, proj=False)
    
    if random:
        # randomly sample same number of epochs, with the same length of bcg_ep
        rand_tp_list = np.sort(np.random.choice(np.arange(np.min(r_list), np.max(r_list)), size=len(r_list), replace=False))
        
        events = rand_tp_list.reshape(-1, 1)
        events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        dataset['bcg_ep_rand'] = mne.Epochs(dataset['raw'], events=events, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=1, baseline=None, proj=False)
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