#%%
import os, sys, copy, glob
import numpy as np
import matplotlib.pyplot as plt
from osl_ephys.preprocessing import run_proc_chain, run_proc_batch
from natsort import natsorted as sorted
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.getcwd()))
from utils import psd_plot, temp_plot, temp_plot_diff, Pathfinder, QRSDetector
from utils.prep_wrappers import set_channel_montage, crop_TR, set_channel_type_raw, epoch_pca, qrs_detect, create_epoch, epoch_sw_pca, ckpt_report, initialize, debug_init, epoch_aas, bcg_ep_ica, epoch_impulse_removal, impulse_removal, voltage_correction, mreeg_bad_channels, summary, init_tracer, mid_crop
from utils.metric import psd_band_ratio

continue_interrupt = True
debug=False
SLICE_FREQ = 1/0.07  # Hz, frequency for the first slice in init_tracer
#%%
if not debug:
    config = {
        'preproc': [
            {'initialize': {'prep': 'after_prep_sts'}},
            {'init_tracer': {
                'psd_slice': partial(psd_band_ratio, band1=[SLICE_FREQ-1, SLICE_FREQ+1], band2='beta', fn1=np.mean),
                'psd_2slice': partial(psd_band_ratio, band1=[SLICE_FREQ*2-1, SLICE_FREQ*2+1], band2=[20, 35], fn1=np.mean),
            }},
            {'set_channel_types': {'VEOG': 'eog', 'HEOG': 'eog', 'EKG': 'ecg', 'EMG': 'emg'}},
            {'set_channel_montage': {}},
            {'notch_filter': {'freqs': 50}},
            {'crop_TR': {'tmin': -0.0456, 'TR': 1.14}},
            {'ckpt_report': {'ckpt_name': 'raw', 'focus_range': [0, 10], 'dB': False}},
            {'create_epoch': {'event': 'TR', 'tmin': -0.0456, 'tmax': 1.1058, 'correct_trig': True}},
            {'epoch_pca': {'epoch_key': 'tr_ep', 'npc': 5, 'force_mean': False, 'overwrite': 'new', 'remove_mean': True, 'picks': 'all'}},
            {'ckpt_report': {'ckpt_name': 'after_pca_removal', 'key_to_print': 'tr_ep', 'dB': False}},
            {'epoch_aas': {'epoch_key': 'tr_ep', 'overwrite': 'new', 'picks': 'all', 'window_length': 10, 'fit': False}},
            {'voltage_correction': {}},
            {'ckpt_report': {'ckpt_name': 'after_aas_removal', 'key_to_print': 'tr_ep_', 'dB': False}},
            {'filter': {'l_freq': 0.5, 'h_freq': 45, 'method': 'iir', 'iir_params': {'order': 6, 'ftype': 'butter'}}},
            {'crop_TR': {'tmin': -0.0456, 'TR': 1.14, 'num_edge_TR': 5}},
            {'mid_crop': {'length': 250}},
            {'resample': {'sfreq': 250}},
            {'ckpt_report': {'ckpt_name': 'after_filt', 'dB': False}},
            {'qrs_detect': {'delay': 0, 'bcg_name': 'pca', 'epoch_len': 1, 'ssp': 5}},
            {'ckpt_report': {'ckpt_name': 'after_ssp', 'qrs_event': True, 'dB': False}},
            {'bad_segments': {'segment_len': 250, 'picks': 'eeg', 'significance_level': 0.1}},
            {'bad_segments': {'segment_len': 250, 'picks': 'eeg', 'mode': 'diff', 'significance_level': 0.1}},
            {'mreeg_bad_channels': {'mode': 'power'}},
            {'mreeg_bad_channels': {'mode': 'power', 'l_freq': 15}},
            {'mreeg_bad_channels': {'mode': 'freq', 'tight_win': 1, 'freq': SLICE_FREQ, 'broad_win': [-1, 4]}},
            {'mreeg_bad_channels': {'mode': 'freq', 'tight_win': 1, 'freq': SLICE_FREQ*2, 'alpha': 0.025}},
            {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
            {'bad_segments': {'segment_len': 2500, 'picks': 'eog'}},
            {'ckpt_report': {'ckpt_name': 'after_bads', 'qrs_event': True, 'dB': False}},
            {'bcg_ep_ica': {}},
            {'ckpt_report': {'ckpt_name': 'after_epoched_ica', 'qrs_event': True, 'dB': False}},
            {'ica_raw': {'n_components': 30, 'picks': 'eeg', 'l_freq': 1}},
            {'ica_autoreject': {'eogmeasure': 'correlation', 'eogthreshold': 0.35, 'ecgmethod': 'ctps', 'ecgthreshold': 'auto', 'apply': True}},
            {'interpolate_bads': {}},
            {'ckpt_report': {'ckpt_name': 'after_ica', 'dB': False}},
            {'set_eeg_reference': {'projection': True}},
            {'summary': {'ds_name': 'staresina'}}
        ]
    }
    
    


    file_list = []
    subject_list = []
    pf = Pathfinder(prep=config['preproc'][0]['initialize']['prep'])
    for subject in pf.subject_list:
        file_list.append(pf.get_eeg_file(subject, "raw", ".edf"))
        subject_list.append(subject)

    pth = pf.get_fdr_dict()
    if continue_interrupt:
        full_file_list = file_list.copy()
        file_list = []
        full_subject_list = subject_list.copy()
        subject_list = []
        
        # Check if files already processed
        finished_list = glob.glob(f'{pth["prep"]}/*/*_preproc-raw.fif')
        finished_list = [full_string.split('/')[-2] for full_string in finished_list]
        error_list = glob.glob(f'{pth["prep"]}/logs/*.error.log')
        error_list = [full_string.split('/')[-1].split('_')[0] for full_string in error_list]
        for subject, preproc in zip(full_subject_list, full_file_list):
            if subject in finished_list:
                print(f"WARNING: {subject} already finished, skipping")
            elif subject in error_list:
                print(f"WARNING: {subject} had an error, skipping")
            else:
                file_list.append(preproc)
                subject_list.append(subject) 
        
    run_proc_batch(config, file_list, subjects=subject_list, outdir=pth["prep"], gen_report=False, overwrite=True, extra_funcs=[set_channel_montage, crop_TR, set_channel_type_raw, epoch_pca, qrs_detect, create_epoch, epoch_sw_pca, ckpt_report, initialize, epoch_aas, bcg_ep_ica, epoch_impulse_removal, impulse_removal, voltage_correction, mreeg_bad_channels, summary, init_tracer, mid_crop])

else:
    config = """
        preproc:
            - initialize: {prep: after_prep_sts}
            - set_channel_types: {VEOG: eog, HEOG: eog, EKG: ecg, EMG: emg}
            - set_channel_montage: {}
            - notch_filter: {freqs: 50}
            - notch_filter: {freqs: 50}
            - crop_TR: {tmin: -0.0456, TR: 1.14}
            - ckpt_report: {ckpt_name: raw, focus_range: [0, 10], dB: false}
            - create_epoch: {event: TR, tmin: -0.0456, tmax: 1.1058, correct_trig: true}
            - epoch_pca: {epoch_key: tr_ep, npc: 5, force_mean: false, overwrite: new, remove_mean: true, picks: all}
            - ckpt_report: {ckpt_name: after_pca_removal, key_to_print: tr_ep, dB: false}
            - epoch_aas: {epoch_key: tr_ep, overwrite: new, picks: all, window_length: 10, fit: false}
            - voltage_correction: {}
            - ckpt_report: {ckpt_name: after_aas_removal, key_to_print: tr_ep_, dB: false}
            - filter: {l_freq: 1, h_freq: 40, method: iir, iir_params: {order: 4, ftype: butter}}
            - crop_TR: {tmin: -0.0456, TR: 1.14, num_edge_TR: 5}
            - resample: {sfreq: 250}
            - qrs_detect: {delay: 0, bcg_name: pca, epoch_len: 1, ssp: 0}
            - ckpt_report: {ckpt_name: after_filt, dB: false}
            - bad_segments: {segment_len: 250, picks: eeg, significance_level: 0.1}
            - bad_segments: {segment_len: 250, picks: eeg, mode: diff, significance_level: 0.1}
            - bad_segments: {segment_len: 2500, picks: eog}
            - mreeg_bad_channels: {mode: power}
            - mreeg_bad_channels: {mode: power, l_freq: 15}
            - bad_channels: {picks: eeg, significance_level: 0.1}
            - interpolate_bads: {}
            # - create_epoch: {event: He132}
            # - epoch_pca: {epoch_key: he_ep, npc: 3, force_mean: false, overwrite: new, remove_mean: false, picks: all}
            # - ckpt_report: {ckpt_name: after_he_removal, key_to_print: he_ep, dB: false}
            # - epoch_pca: {epoch_key: bcg_ep, npc: 5, force_mean: false, overwrite: even, remove_mean: true, picks: eeg, spurious_event: true, screen_high_power: true}

            # - ckpt_report: {ckpt_name: after_bcg_removal, qrs_event: true, dB: false}
            # - bcg_ep_ica: {}
            # - ica_raw:            {n_components: 30, picks: eeg, l_freq: 1}
            # - ica_autoreject:     {eogmeasure: correlation, eogthreshold: 0.35, ecgmethod: ctps, ecgthreshold: auto, apply: true}
            # - interpolate_bads: {}
            # - set_eeg_reference: {projection: true}
            # - ckpt_report: {ckpt_name: after_epoched_ica, dB: false}
    """
    
    #%%
    pf = Pathfinder(prep=config.split("- debug_init: {prep: ")[1].split("}")[0])
    subject = "1121"

    raw = pf.get_eeg_file(subject, "raw", ".edf")
    pth = pf.get_fdr_dict()

    dataset = run_proc_chain(config, raw, subject=subject, outdir=pth['debug'], ret_dataset=True, gen_report=False, overwrite=True, extra_funcs=[set_channel_montage, crop_TR, set_channel_type_raw, create_epoch, epoch_pca, qrs_detect, epoch_sw_pca, ckpt_report, debug_init, epoch_aas, bcg_ep_ica])

    #%%
    temp_plot(dataset['raw'], 0, length=5000*299)
    psd_plot([dataset['raw']], fmax=40)
    psd_plot([dataset['raw']], dB=False, fmax=40)
    psd_plot([dataset['raw']], fmax=40, res_mult=8)
    psd_plot([dataset['raw']], dB=False, fmax=40, res_mult=8)
    #%%
    dataset_pca = epoch_pca(copy.deepcopy(dataset), {"epoch_key": "tr_ep", "npc": 3, "force_mean": False, "overwrite": "new", "remove_mean": True, "picks": "all"})
    dataset_pcavar = epoch_pca(copy.deepcopy(dataset_pca), {"epoch_key": "tr_ep", "npc": 3, "force_mean": False, "overwrite": "new", "remove_mean": True, "picks": "all", "screen_impedance": True})
    dataset_pcavar_aas = epoch_aas(copy.deepcopy(dataset_pcavar), {"epoch_key": "tr_ep", "remove_mean": False, "window_length": 10, "fit": False, "picks": "all"})
    dataset_pcavar_aas_pcav = epoch_pca(copy.deepcopy(dataset_pcavar_aas), {"epoch_key": "tr_ep", "npc": 3, "force_mean": False, "overwrite": "new", "remove_mean": True, "picks": "all", "screen_impedance": True})
    #%%
    temp_plot(dataset_pcavar_aas['raw'], 0, length=5000*299)
    psd_plot([dataset_pcavar_aas['raw']], fmax=40)
    psd_plot([dataset_pcavar_aas['raw']], dB=False, fmax=40)
    psd_plot([dataset_pcavar_aas['raw']], fmax=40, res_mult=8)
    psd_plot([dataset_pcavar_aas['raw']], dB=False, fmax=40, res_mult=8)
        #%%
    temp_plot(dataset_pcavar_aas_pcav['raw'], 0, length=5000*299)
    psd_plot([dataset_pcavar_aas_pcav['raw']], fmax=40)
    psd_plot([dataset_pcavar_aas_pcav['raw']], dB=False, fmax=40)
    psd_plot([dataset_pcavar_aas_pcav['raw']], fmax=40, res_mult=8)
    psd_plot([dataset_pcavar_aas_pcav['raw']], dB=False, fmax=40, res_mult=8)
    #%%
    temp_plot(dataset_pcavar_aas['raw'], 0, length=5000*10, start=5000*100)
    temp_plot(dataset_pcavar_aas_pcav['raw'], 0, length=5000*10, start=5000*100)
    #%%
    dataset_pca2 = epoch_pca(copy.deepcopy(dataset_pca), {"epoch_key": "tr_ep", "npc": 3, "force_mean": False, "overwrite": "new", "remove_mean": True, "picks": "all"})
    
    # %%
    temp_plot(dataset_pca2['raw'], 0, length=5000*299)
    psd_plot([dataset_pca2['raw']], fmax=40)
    psd_plot([dataset_pca2['raw']], dB=False, fmax=40)
    psd_plot([dataset_pca2['raw']], fmax=40, res_mult=8)
    psd_plot([dataset_pca2['raw']], dB=False, fmax=40, res_mult=8)

    # %%
    temp_plot(dataset_pca2['raw'], 0, start=5000*100, length=5000*10)
    #%%
    dataset_pca2_aas = epoch_aas(copy.deepcopy(dataset_pca2), {"epoch_key": "tr_ep", "remove_mean": False, "window_length": 10, "fit": False, "picks": "all"})
    dataset_pca2_aas2 = epoch_aas(copy.deepcopy(dataset_pca2_aas), {"epoch_key": "tr_ep", "remove_mean": False, "window_length": 10, "fit": False, "picks": "all"})
    
    #%%
    # dataset_pca2 = create_epoch(dataset_pca2, {"event": "slice", "tmin": 0, "tmax": 0.07})
    dataset_pca2_saas_16 = epoch_aas(copy.deepcopy(dataset_pca2), {"epoch_key": "slice_ep", "remove_mean": False, "window_length": 16, "fit": False, "picks": "all"})
    dataset_pca2_saas_32 = epoch_aas(copy.deepcopy(dataset_pca2), {"epoch_key": "slice_ep", "remove_mean": False, "window_length": 32, "fit": False, "picks": "all"})
    dataset_pca2_saas_160 = epoch_aas(copy.deepcopy(dataset_pca2), {"epoch_key": "slice_ep", "remove_mean": False, "window_length": 160, "fit": False, "picks": "all"})
    
    #%%
    dataset_pca2_daas_16 = epoch_aas(copy.deepcopy(dataset_pca2_saas_16), {"epoch_key": "tr_ep", "remove_mean": False, "window_length": 10, "fit": False, "picks": "all"})
    dataset_pca2_daas_32 = epoch_aas(copy.deepcopy(dataset_pca2_saas_32), {"epoch_key": "tr_ep", "remove_mean": False, "window_length": 10, "fit": False, "picks": "all"})
    dataset_pca2_daas_160 = epoch_aas(copy.deepcopy(dataset_pca2_saas_160), {"epoch_key": "tr_ep", "remove_mean": False, "window_length": 10, "fit": False, "picks": "all"})
    
    
    # %%
    temp_plot(dataset_pca2_aas2['raw'], 0, length=5000*299)
    psd_plot([dataset_pca2_aas2['raw']], fmax=40)
    psd_plot([dataset_pca2_aas2['raw']], dB=False, fmax=40)
    psd_plot([dataset_pca2_aas2['raw']], fmax=40, res_mult=8)
    psd_plot([dataset_pca2_aas2['raw']], dB=False, fmax=40, res_mult=8)

    # %%
    temp_plot(dataset_pca2_aas['raw'], 0, start=5000*100, length=5000*10)
    # %%
    dataset_pca2_aas_pca = epoch_pca(copy.deepcopy(dataset_pca2_aas), {"epoch_key": "tr_ep", "npc": 3, "force_mean": False, "overwrite": "new", "remove_mean": True, "picks": "all"})
    
    # %%
    temp_plot(dataset_pca2_aas_pca['raw'], 0, length=5000*299)
    psd_plot([dataset_pca2_aas_pca['raw']], fmax=40)
    psd_plot([dataset_pca2_aas_pca['raw']], dB=False, fmax=40)
    psd_plot([dataset_pca2_aas_pca['raw']], fmax=40, res_mult=8)
    psd_plot([dataset_pca2_aas_pca['raw']], dB=False, fmax=40, res_mult=8)
    # %%
    temp_plot(dataset_pca2['raw'], 0, start=5000*100, length=5000*10)
    temp_plot(dataset_pca2_aas_pca['raw'], 0, start=5000*100, length=5000*10)
    temp_plot_diff(dataset_pca2['raw'], dataset_pca2_aas_pca['raw'], 0, start=5000*100, length=5000*10)
    # %%
    temp_plot(dataset_pca2['raw'], 0, start=5000*0, length=5000*299)
    temp_plot(dataset_pca2_aas_pca['raw'], 0, start=5000*0, length=5000*299)
    temp_plot_diff(dataset_pca2['raw'], dataset_pca2_aas_pca['raw'], 0, start=5000*0, length=5000*299)
    # %%
    temp_plot(dataset_pca2_aas_pca['raw'], 'EKG', start=5000*100, length=5000*10)
# %%
