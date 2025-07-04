import os, re, glob, copy, pickle
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
from .eeg import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, filename2subj, HeteroStudy as Study, Pathfinder, find_spurious_channels, pcs_plot, pick_indices, SingletonEEG
from .qrs import kteager_detect, qrs_correction, QRSDetector
from ecgdetectors import panPeakDetect
from mne.preprocessing import ICA

spurious_subject_list = ['13121', '8111', '8112', '8121', '17111', '17112', '31111', '31112', '31121']

def initialize(dataset, userargs):
    ds_name = userargs.get('ds_name', 'staresina')
    
    if ds_name == 'irene':  # for irene, dev_head_t is note set, so we need to set it to identity
        dataset['raw'].info['dev_head_t'] = SingletonEEG("/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/edfs/sub-003_ses-01_run-01_block-01_task-resting_convert.cdt.edf").info['dev_head_t']
    
    dataset['pf'] = Pathfinder(**userargs)
    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']
    
    if 'Trigger' in dataset['raw'].ch_names:
        dataset['raw'].drop_channels(['Trigger'])
    
    subject = filename2subj(dataset['raw'].filenames[0], ds_name=ds_name)
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
        'res_mult': 32,
        'max_freq': 50,
        'qrs_event': False,
        'key_to_print': None,
        'always_print': ['EKG'],    # must be name, 'eeg' is not allowed
        'std_channel_pick': 'eeg',
        'print_pcs': True,
        'print_noise': True,
        'ds_name': 'staresina',
        'dB': True,  # whether to plot psd in dB scale
        'focus_range': [100, 110],  # in seconds, for temp_plot
    }
    userargs = proc_userargs(userargs, default_args)
    
    fs = dataset['raw'].info['sfreq']
    picks = dataset[f"picks_{userargs['key_to_print']}"] if f"picks_{userargs['key_to_print']}" in dataset else 'eeg'
    subject = filename2subj(dataset['raw'].filenames[0], ds_name=userargs['ds_name'])
    save_fdr = os.path.join(dataset['pf'].get_fdr_dict()['prep'], "ckpt", subject, userargs['ckpt_name'])
    ensure_dir(save_fdr)
    
    if userargs['key_to_print'] is None:
        userargs['print_noise'] = userargs['print_pcs'] = False
    
    psd = psd_plot([dataset['raw']], [userargs['ckpt_name']], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], save_pth=os.path.join(save_fdr, f"psd.png"), picks=picks, dB=userargs['dB'])
    std = np.mean(np.std(dataset['raw'].get_data(picks=userargs['std_channel_pick'], reject_by_annotation='omit'), axis=1))
    plot_channel_dists(dataset['raw'], os.path.join(save_fdr, f"std={std:.4e}.png"))
    
    def print_ch(ch_name):
        extra_str = "temp"
        print_fdr = os.path.join(save_fdr, extra_str)
        ensure_dir(print_fdr)
        
        psd_plot([dataset['raw']], [ch_name], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], picks=ch_name, save_pth=os.path.join(print_fdr, f"{ch_name}_psd.png"), dB=userargs['dB'])
        temp_plot(dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}.png"), name=ch_name)
        temp_plot(dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}.png"), name=ch_name)
        if 'ckpt_raw' in dataset:
            if dataset['ckpt_raw'].info['sfreq'] != dataset['raw'].info['sfreq']:
                dataset['ckpt_raw'].resample(dataset['raw'].info['sfreq'])
            if not userargs['qrs_event']:
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}_diff.png"),name=ch_name)
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}_diff.png"), name=ch_name)
            else:
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=os.path.join(print_fdr, f"{ch_name}_diff.png"),name=ch_name, events=dataset['bcg_ep'].events, event_id=999)
                temp_plot_diff(dataset['ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=os.path.join(print_fdr, f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}_diff.png"), name=ch_name, events=dataset['bcg_ep'].events, event_id=999)
    
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
            psd_plot([dataset[f"noise_{userargs['key_to_print']}"]], [ch_name], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], picks=[ch_name], save_pth=os.path.join(noise_fdr_name, f"{ch_name}_psd.png"), dB=userargs['dB'])
            temp_plot(dataset[f"noise_{userargs['key_to_print']}"], ch_name, fs=fs, save_pth=os.path.join(noise_fdr_name, f"{ch_name}.png"), name=ch_name)
        
        psd_plot([dataset[f"noise_{userargs['key_to_print']}"]], ['noise'], res_mult=userargs['res_mult'], fs=fs, figsize=(10, 3), fmax=userargs['max_freq'], save_pth=os.path.join(noise_fdr_name, f"noise_psd.png"), picks=channel_to_print, dB=userargs['dB'])
    
    dataset['ckpt_raw'] = copy.deepcopy(dataset['raw'])
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
    elif ds_name == 'irene':
        subject = filename2subj(dataset['raw'].filenames[0], ds_name=ds_name)
        subj_dict = parse_subj(subject, True, ds_name='irene')
        
        bcg_file_pth = f"/ohba/pi/knobre/irene/data_for_jize/curry_clean/visit{subj_dict['visit']}/s{subj_dict['subj']}/s{subj_dict['subj']}_mrEEG{subj_dict['visit']}_block{subj_dict['block']}_mr_bcg_clean.cdt"
        if not os.path.exists(bcg_file_pth):
            bcg_file_pth = f"/ohba/pi/knobre/irene/data_for_jize/curry_clean/visit{subj_dict['visit']}/s{subj_dict['subj']}/s{subj_dict['subj']}_mrEEG_visit{subj_dict['visit']}_block{subj_dict['block']}_mr_bcg_clean.cdt"
        
        bcg_file = mne.io.read_raw_curry(bcg_file_pth, preload=True)
        dataset["raw"].set_montage(copy.deepcopy(bcg_file.get_montage()))
        del bcg_file

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
        event_name = '1200002'
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
        tmax = min(end_point/freq+num_edge_TR*TR, eeg.tmax)
        eeg = eeg.crop(tmin=new_tmin, tmax=tmax)  
        return eeg
    
    dataset["raw"] = crop_eeg_to_tr(dataset["raw"], tmin=tmin, num_edge_TR=num_edge_TR)
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
    event_name = userargs.get('event_name', None)
    epoch_name_diy = userargs.get('epoch_name', None)   # if None, will be set to event + '_ep' or event + '_ep_rand' if random is True
    
    if event == 'slice':
        if event_name is None:
            event_name = '1200002'
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
        if event_name is None:
            event_name = '1200002'
        try:
            event_id = mne.events_from_annotations(dataset['raw'])[1][str(event_name)]
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
    spurious_event = userargs.get('spurious_event', False)  # if True, epoch_key is used for removal, epoch_key + "_screener" is used for PC calculation
    
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
    
    padding = torch.repeat_interleave(all_pcs[0:1], window_length-1, dim=0)
    all_pcs = torch.cat([padding, all_pcs], dim=0)    # 29+#win, #ch, len(ep), #pc
    
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


def epoch_aas(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep

    all_pcs = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
    
    padding = torch.repeat_interleave(all_pcs[0:1], window_length-1, dim=0)
    all_pcs = torch.cat([padding, all_pcs], dim=0)    # 29+#win, #ch, len(ep), #pc
    
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
    
    

def epoch_pca(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    npc = userargs.get('npc', 3)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'obs')
    remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
    spurious_event = userargs.get('spurious_event', False)  # if True, epoch_key is used for removal, epoch_key + "_screener" is used for PC calculation
    screen_high_power = userargs.get('screen_high_power', None)  # if True, the epochs with high power would not be used for PC calculation. If None, no screening is performed. If false, only the epochs with high power would be used for PC calculation.
    
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks)) # #ep, #ch, len(ep)
    if spurious_event:
        screener = dataset[f"{epoch_key}_screener"]
        orig_data = orig_data[screener]
    
    if not screen_high_power is None:
        epoch_power = torch.sum(orig_data**2, dim=(1,2)) # #ep
        power_med = epoch_power.median()
        power_mad = torch.median(torch.abs(epoch_power - power_med))
        threshold = power_med + 3*power_mad
        orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]
        
    
    # orig_data = orig_data.reshape(*orig_data.shape[1:], orig_data.shape[0])
    orig_data = orig_data.permute(1, 2, 0)  # #ch, len(ep), #ep
    
    epoch_std = orig_data.std(dim=(0,1))
    normal_ep = epoch_std < (epoch_std.mean() + 3*epoch_std.std())
    orig_data = orig_data[..., normal_ep]  # #ch, len(ep), #ep
    
    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean.unsqueeze(1)
    if force_mean_pc0:
        pc0 = torch.mean(dirty_data, dim=2).unsqueeze(-1)  # #ch, len(ep), 1
        detrended = dirty_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)   # #ch, len(ep), q;    # #ch, q
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1) # #ch, len(ep), #pc
    else:   
        U, S, _ = torch.pca_lowrank(dirty_data)   # #ch, len(ep), q;    # #ch, q
        all_pcs = U[..., :npc]*S[..., None, :npc]
        
    # if spurious_event:
    del orig_data, dirty_data, U, S  # free memory
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # #ep, #ch, len(ep)
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


def qrs_detect(dataset, userargs):
    delay = userargs.get('delay', 0.0)  # if use EKG, delay is better to be 0.21, if use EEG, use 0.0
    bcg_name = userargs.get('bcg_name', 'pca')
    l_freq = userargs.get('l_freq', 5)
    h_freq = userargs.get('h_freq', 15)
    correct = userargs.get('correct', True)
    method = userargs.get('method', 'kteo')
    random = userargs.get('random', False)
    median_mult = userargs.get('median_mult', 0.75)
    
    if method == 'mne':
        ecg = find_ecg_events(dataset['raw'], ch_name=bcg_name, l_freq=l_freq, h_freq=h_freq)
        if ecg[0].size == 0:
            ecg = find_ecg_events(dataset['raw'], ch_name=None, l_freq=l_freq, h_freq=h_freq)
            print("Warning: No R peaks detected. Please check the ECG channel. Using other channels for reference.")
            if ecg[0].size == 0:
                raise AssertionError("No R peaks detected. Please check the data.")
        ecg = np.unique(ecg[0], axis=0)
        if correct:
            raise NotImplementedError("Correction for MNE method is not implemented yet.")
            ecg, spurious_ecg = qrs_correction(ecg, dataset['raw'], dataset['raw'].get_data(picks='EKG').squeeze(), new_event_idx=999)
    elif method == 'kteo':
        fs = dataset['raw'].info['sfreq']
        kteo, ecg_data = kteager_detect(dataset["raw"], filt_emg=False, filt_kteo=True, picks=bcg_name, l_freq=l_freq, h_freq=h_freq)
        peaks = np.array(panPeakDetect(kteo, fs))
        peaks += dataset['raw'].first_samp
        ecg = np.column_stack([peaks, np.zeros(len(peaks)), 999*np.ones(len(peaks))]).astype(np.int64)
        if correct:
            safe_event_screener, spurious_ecg = qrs_correction(ecg, dataset['raw'], ecg_data, new_event_idx=999)
    else:
        ecg = QRSDetector(dataset['raw'], ch_name=bcg_name, l_freq=l_freq, h_freq=h_freq).get_events(correction=correct, method=method)
        raise NotImplementedError(f"Method {method} not implemented for QRS detection.")
    
    r_list = spurious_ecg[:,0]
    half_ep_size = np.median(np.diff(r_list)) * median_mult / dataset['raw'].info['sfreq']
 
    # dataset['bcg_ep_safe'] = mne.Epochs(dataset['raw'], events=ecg, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=999, baseline=None, proj=False)
    dataset['bcg_ep'] = mne.Epochs(dataset['raw'], events=spurious_ecg, tmin=delay-half_ep_size, tmax=delay+half_ep_size, event_id=999, baseline=None, proj=False)
    dataset['bcg_ep_screener'] = [ep_idx for ep_idx, event_idx in enumerate(safe_event_screener) if event_idx in dataset['bcg_ep'].selection]
    
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

def bcg_ep_ica(dataset, userargs):
    threshold = userargs.get('threshold', 0.05)
    picks = userargs.get('picks', 'eeg')
    seed = userargs.get('seed', 42)
    max_iter = userargs.get('max_iter', 'auto')
    n_components = userargs.get('n_components', 20)
    qrs_event_id = userargs.get('qrs_event_id', 999)
    downsample = userargs.get('downsample', 1)    # only set this if the "bcg_ep" is not downsampled!!!
    
    assert 'bcg_ep' in dataset, "Please run qrs_detect first to create bcg_ep."
    
    ev = copy.deepcopy(dataset["bcg_ep"].events)
    
    ev[:,0] = ev[:,0] / downsample
    ev = ev.astype(np.int64)
    bcg_ep = mne.Epochs(dataset['raw'], events=ev, tmin=dataset["bcg_ep"].tmin, tmax=dataset["bcg_ep"].tmax, event_id=qrs_event_id, baseline=None, proj=False)
    bcg_ep.load_data()  # Ensure the data is loaded before applying ICA
    
    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=seed)
    ica.fit(bcg_ep)
    
    exclude_list = []
    for i in range(n_components):
        var = ica.get_explained_variance_ratio(bcg_ep, components=[i], ch_type=picks)
        if var['eeg'] > threshold:
            exclude_list.append(i)
        else:
            break
    
    ica.exclude = exclude_list
    dataset['raw_before_ica'] = copy.deepcopy(dataset['raw'])
    dataset['raw'] = ica.apply(dataset['raw'])
    return dataset