import numpy as np
import torch
import mne
from mne.annotations import _annotations_starts_stops
from mne.filter import filter_data
from mne.utils import sum_squared
from mne.preprocessing.ecg import _get_ecg_channel_index, _make_ecg
import scipy.signal as signal
from scipy.signal import welch
# from ecgdetectors import Detectors


def kteager_detect(raw, slice_period=None, tr_period=None, filt_emg=False, filt_kteo=True, picks='pca', l_freq=5, h_freq=15, filter_length=0.04):
    """Detect QRS complexes using K. Teager's method."""
    fs = round(raw.info['sfreq'])

    if slice_period is not None:
        raise NotImplementedError("slice_period is not implemented yet. plans to ignore harmonics of slice/tr period when calculating f_d for value k, to minimize the effect of residual GA to qrs detection")
    if tr_period is not None:
        raise NotImplementedError("tr_period is not implemented yet")
    if picks == 'mean':
        # Use mean of EEG channels as ECG source
        ecg_data = raw.get_data(picks='eeg', reject_by_annotation='NaN')
        ecg_data = np.nan_to_num(ecg_data, nan=0.0).mean(axis=0, keepdims=True)
    elif picks == 'pca':
        ecg_data = raw.get_data(picks='eeg', reject_by_annotation='NaN')
    else:
        ecg_data = raw.get_data(picks=picks, reject_by_annotation='NaN')
        ecg_data = np.nan_to_num(ecg_data, nan=0.0)
    
    # bandpass 7-40, fmrib code put this after MA filter, unsure why
    ecg_data = mne.filter.filter_data(ecg_data, sfreq=fs, l_freq=l_freq, h_freq=h_freq, verbose=False)
    
    def ma_filter(data, interval):
        """Apply moving average filter to data."""
        data = np.nan_to_num(data, nan=0.0)
        b = round(interval * fs)
        b = np.ones(b) / b
        return signal.filtfilt(b, [1], data)

    if filt_emg:
        # EMG noise suppression, 0.028 s MA filter
        ecg_data = ma_filter(ecg_data, 0.02)
        ecg_data = ma_filter(ecg_data, 0.028)
        
    if picks == 'pca':
        ecg_data = np.nan_to_num(ecg_data, nan=0.0)
        ecg_data = torch.tensor(ecg_data)
        ecg_data = ecg_data - torch.mean(ecg_data, dim=0).unsqueeze(0)  # #ch, #timepoints
        U, S, _ = torch.pca_lowrank(ecg_data.T)   # [#timepoints, q]; [q,]
        ecg_data = U[:, :1]*S[None, :1]  # #timepoints, 1
        ecg_data = np.array(ecg_data.T)
        
    # calculate k value
    n_fft = np.power(2, np.round(np.log2(fs))).astype(np.int64)
    freqs, psd = welch(ecg_data, fs, nperseg=n_fft)
    f_d = freqs[np.argmax(psd)]
    k = round(fs / f_d / 4)
    
    ecg_data = ecg_data.squeeze()
    kteo = ecg_data[k:-k]**2 - ecg_data[:-2*k]*ecg_data[2*k:]
    kteo[-1] = 0
    if filt_kteo:
        # MA filter with 0.04 s
        kteo = ma_filter(kteo, filter_length)
    kteo[kteo < 0] = 0  
    padded_kteo = np.zeros_like(ecg_data)
    padded_kteo[k:-k] = kteo
    return (padded_kteo, ecg_data)
    
    # # mfr calculation
    # ms350 = round(0.35 * fs)
    # ms300 = round(0.3 * fs)
    # ms50 = round(0.05 * fs)
    # ms1200 = round(1.2 * fs)
    # msWait = round(0.55 * fs)

    # M = np.zeros(len(kteo))
    # R = np.zeros(len(kteo))
    # F = np.zeros(len(kteo))
    # F2 = np.zeros(len(kteo))
    # MFR = np.zeros(len(kteo))
    # Mc = 0.45
    # R5 = np.ones(5) * round(nfft/np.argmax(psd))
    # M5 = Mc * np.ones(5) * np.max(kteo[fs:fs*6])

    # M[:5*fs] = np.mean(M5)
    # newM5 = np.mean(M5)
    # F[:ms350] = np.mean(kteo[fs:fs+ms350])
    # F2[:ms350] = np.mean(kteo[fs:fs+ms350])
    
    # detect_flag = False
    # timer1 = 0
    # peaks = []
    # for n in range(len(kteo)):
    #     timer1 += 1
    #     if len(peaks) >= 2:
    #         if detect_flag:
    #             detect_flag = False
    #             M[n] = np.mean(M5)
    #             Mdec = (M[n] - M[n]*Mc) / (ms1200-msWait)
    #             Rdec = Mdec/1.4
    #         elif timer1 <= msWait or timer1 > ms1200:
    #             M[n] = M[n-1]
    #         elif timer1 == msWait + 1:
    #             M[n] = M[n-1] - Mdec
    #             newM5 = Mc * np.max(kteo[n-msWait:n])
    #             newM5 = min(newM5, 1.5 * M5[4])
    #             M5 = np.roll(M5, -1)
    #             M5[-1] = newM5
    #         elif timer1 > msWait+1 and timer1 <= ms1200:
    #             M[n] = M[n-1] - Mdec
    #     if n > ms350:
    #         F[n] = F[n-1] + (np.max(kteo[n+1-ms50:n+1]) - np.max(kteo[n+1-ms350:n+1-ms300])) / 150
    #         F2[n] = F[n] - np.mean(kteo[fs:fs+ms350+1]) + newM5
        
    #     Rm = np.mean(R5)
    #     if timer1 <= round(2*Rm/3):
    #         R[n] = 0
    #     elif len(peaks)>=2:
    #         R[n] = R[n-1] - Rdec
        
    #     MFR[n] = M[n] + F2[n] + R[n]
        
    #     if kteo[n] >= MFR[n] and (timer1 > msWait or len(peaks)==0):
    #         peaks.append(n)
    #         if len(peaks) > 1:
    #             R5 = np.roll(R5, -1)
    #             R5[-1] = peaks[-1] - peaks[-2]
    #         detect_flag = True
    #         timer1 = -1
    
    # if debug:
    #     return peaks, kteo
    # return peaks



def qrs_correction(ecg_event, raw, ecg_signal, max_heart_rate=160, min_heart_rate=40, new_event_idx=998, corr_thres=0.5, local_maxima=None):
    """
    Correct ECG events of the raw EEG data, based on the provided ecg signal.
    Parameters:
        - ecg_event: Array of ECG events, shape (N, 3), where N is the number of events.
        - raw: MNE Raw object containing the EEG data.
        - ecg_signal: Array of shape (N,) representing the ECG signal, to be used for event detection.
        - max_heart_rate: Maximum heart rate in beats per minute (default: 160). Any events that are too close to each other will be considered as false positives, and would be removed.
        - min_heart_rate: Minimum heart rate in beats per minute (default: 40). Any events that are too far apart will be considered as false negatives, and would be interpolated.
        - new_event_idx: Index to use for the new events in the MNE Raw object (default: 998).
        - corr_thres: Threshold for the Pearson correlation coefficient to consider an event as safe (default: 0.5). The correlation is calculated in the ecg or eeg signal, not the ecg.
        - local_maxima: None or array of local maxima of the energy operator of the ECG signal. eneygy could be calculated using the K. Teager's method, or Pan-Tompkins algorithm, or any other method that returns a 1D non-negative array of the same length as the ECG signal.
    Returns:
        - new_events: Array of safe ECG events, shape (M, 3), where M is the number of safe events.
        - spurious_events: Array of all events, including safe and unsafe ones, shape (K, 3), where K is the total number of events detected, K >= M.
    """
    event_timing = ecg_event[:, 0]
    event_length = np.diff(event_timing)
    
    # step 1: calc median and std of event length, with too short & long events ignored
    # long events MUST be ignored as they might be caused by bad segments detection
    min_length = 60.0 / max_heart_rate * raw.info['sfreq']
    max_length = 60.0 / min_heart_rate * raw.info['sfreq']
    inrange_length = event_length[(event_length<max_length) & (event_length>min_length)]
    med, std = np.median(inrange_length), np.std(inrange_length)
    if med < 1.5*min_length:
        med = 1.5*min_length    # note that it is possible that false positive are too many, leading to a very low median, which is not desired.
        
    # step 2: fp removal: remove events that are too close to each other
    # using fp_flag+1, the one removed is always the second one for coding laziness (empirically the second one is usually fp)
    # for the harsh_flag calculating template, both the surrounding events are removed for robustness
    low_thres = max(min_length, med-std*3, 0.75*med)    # 0.75 = 1.5/2
    fp_flag = event_length < low_thres
    
    fp_harsh_flag = np.zeros(len(fp_flag)+1, dtype=bool)
    fp_idx = np.where(fp_flag)[0]
    fp_harsh_flag[fp_idx] = True
    fp_harsh_flag[fp_idx + 1] = True
    
    event_removed = []
    last_fp_len = 0     # 0 if last event is not fp, else is len(last event)
    for event_idx in range(len(fp_flag)):
        if fp_flag[event_idx]:
            last_fp_len += event_length[event_idx]
            if last_fp_len > low_thres: # this if statement only happens if consecutive fp occurs.
                last_fp_len = 0
            else:                       # the first fp in a consecutive fp sequence is always fallen here
                event_removed.append(event_idx+1)
            continue
        last_fp_len = 0
    
    mask = np.ones(len(fp_flag)+1, dtype=bool)
    mask[event_removed] = False
    event_timing = event_timing[mask]
    
    # the code above and below differs in assumption for a consecutive events of ooooxx..xxooo, where x are fp
    # code above assume the heartbeats is within one of the fps
    # code below is a more conservative version used when ecg_signal has super low quality, assuming none fp is usable
    # event_timing = np.insert(event_timing[1:][~fp_flag], 0, event_timing[0])
    
    # step 3: average heartbeat template calculation, only consider safe events
    fn_idx = np.where((event_length > min(max_length, 1.5*med)))[0] # 1.5 = 0.75*2
    fn_harsh_flag = np.zeros(len(fp_flag)+1, dtype=bool)
    fn_harsh_flag[fn_idx] = True
    fn_harsh_flag[fn_idx + 1] = True
    harsh_flag = np.logical_or(fp_harsh_flag, fn_harsh_flag)
    safe_event = ecg_event[:, 0][~harsh_flag]
    
    search_range = round(med / 3)
    half_win_range = round(med / 2)
    # ecg_data =raw.get_data(picks='ecg')
    safe_windows = []
    for ev in safe_event:
        start = ev - half_win_range - raw.first_samp
        end = ev + half_win_range + 1 - raw.first_samp
        if start >= 0 and end <= raw._data.shape[1]:
            safe_windows.append(ecg_signal[start:end])
    
    if len(safe_windows) < ecg_event.shape[0] // 10:
        for ev in ecg_event[:,0]:
            start = ev - half_win_range - raw.first_samp
            end = ev + half_win_range + 1 - raw.first_samp
            if start >= 0 and end <= raw._data.shape[1]:
                safe_windows.append(ecg_signal[start:end])
    tmplt = np.mean(np.array(safe_windows), axis=0).squeeze()
    
    # step 4: move each event a little bit to maximize the pearson correlation with the template
    def pearson_corr(windows: np.ndarray, template: np.ndarray):
        """
        windows: shape (N, L) — N windows of length L
        template: shape (L,) — single QRS template
        returns: shape (N,) — correlation of each window with the template
        """
        # Normalize template
        return np.array([np.abs(np.corrcoef(window, template)[0,1]) for window in windows])

    def align_with_template(event_time_list, corr_thres=0.5):  
        event_time_list = np.array(event_time_list, dtype=np.int64)
        old_corr = 0    # for recording the last correlation value=
        new_event_list = []
        for ev in event_time_list:
            ev_pos = ev - raw.first_samp
            if (ev_pos-search_range-half_win_range<0) or (ev_pos+search_range+half_win_range>raw._data.shape[1]):
                continue    # if the event is too close to the edge of the signal, remove it
            
            win_pos_list = np.arange(ev_pos-search_range, ev_pos+search_range+1)
            
            if local_maxima is not None:
                win_pos_list = np.intersect1d(win_pos_list, local_maxima)
                if len(win_pos_list) == 0:
                    continue    # if no local maxima in the search range, remove it, leaving fn removal to deal with it
            
            window_arr = np.stack([ecg_signal[pos-half_win_range:pos+half_win_range+1] for pos in win_pos_list])
            corr = pearson_corr(window_arr, tmplt)
            
            if np.max(corr) < corr_thres:
                continue    # if no correlation above threshold, remove it, leaving fn removal to deal with it
            
            best_pos = win_pos_list[np.argmax(corr)]
            
            # if we stop here, the event could be too close to the previous event, so we need to check it
            if len(new_event_list) > 0 and best_pos + raw.first_samp - new_event_list[-1][0] < low_thres:
                if np.max(corr) < old_corr:
                    continue
                else:
                    # if the correlation is better than the last one, we remove the last event
                    new_event_list = new_event_list[:-1]
            
            old_corr = np.max(corr)
            new_event_list.append([best_pos+raw.first_samp, 0, new_event_idx])
            
        return np.unique(np.array(new_event_list), axis=0)
    new_event = align_with_template(event_timing, corr_thres=corr_thres)
    
    # step 5: fn removal: add events between two far away events
    # also at start & end of the signal
    # also align them if iterations > 1
    event_length = np.diff(new_event[:, 0])
    inrange_length = event_length[(event_length<max_length)]
    if len(inrange_length) > 0:
        new_med = np.median(inrange_length)
        med = med if np.abs(med-0.8*raw.info['sfreq']) < np.abs(new_med-0.8*raw.info['sfreq']) else new_med

    thres = min(max_length, 1.5*med)
    fn_pos_list = np.where(event_length>thres)[0]
    fn_list = []
    for fn_pos in fn_pos_list:
        hb_in_between = round(event_length[fn_pos] / med)  # number of hearbeats in between, hb_in_between = number of missing events + 1
        fn_length = round(event_length[fn_pos]/hb_in_between)
        fn = (np.arange(1, hb_in_between) * fn_length + new_event[fn_pos][0]).astype(np.int64)
        fn_list.append(fn)
    if new_event[0, 0] - raw.first_samp > thres:
        missing_events = np.floor((new_event[0, 0] - raw.first_samp) / med)
        fn = new_event[0, 0] - (np.arange(missing_events)+1) * med
        fn_list.append(fn)
    if new_event[-1, 0] - raw.first_samp + thres < raw._data.shape[1]:
        missing_events = np.floor((raw._data.shape[1] - new_event[-1, 0]) / med)
        fn = new_event[-1, 0] + (np.arange(missing_events)+1) * med
        fn_list.append(fn)
    
    new_event = np.unique(new_event,axis=0)
    new_event = new_event[np.argsort(new_event[:, 0])]
    
    if len(fn_list) > 0:
        fn_list = np.concatenate(fn_list)
        
        if local_maxima is not None:
            new_fn_list = []
            for fn_pos in fn_list:
                closest_idx = np.abs(local_maxima - fn_pos).argmin()
                distance = np.abs(local_maxima[closest_idx] - fn_pos)
                
                if distance < med / 5: 
                    new_fn_list.append(local_maxima[closest_idx])
                else:
                    new_fn_list.append(fn_pos)  
                
            fn_list = np.array(new_fn_list, dtype=np.int64) 
                    
        # if iterations > 1:  # if last iteration, don't align fn_list with template
            # fn_list = align_with_template(fn_list, corr_thres=corr_thres)
        fn_event = np.column_stack([fn_list, np.zeros(len(fn_list)), new_event_idx*np.ones(len(fn_list))]).astype(np.int64)
        spurious_event = np.concatenate([new_event, fn_event])
        spurious_event = np.unique(spurious_event, axis=0)
        spurious_event = spurious_event[np.argsort(spurious_event[:, 0])]
    else:
        spurious_event = new_event.copy()
    
    return (new_event, spurious_event)
    # return (new_event, spurious_event) if iterations <= 0 else qrs_correction(new_event, raw, operator, max_heart_rate, min_heart_rate, new_event_idx, iterations=iterations, corr_thres=corr_thres)

class QRSDetector:
    def __init__(self, raw, ch_name=None, tstart=0.0,
                    l_freq=7, h_freq=40, 
                    filter_length='auto', reject_by_annotation=True, shortest_seg=10):
        """Initialize QRSDetector with raw data and parameters."""
                
        self.tstart = tstart
        self.fs = raw.info['sfreq']
        # self.detectors = Detectors(self.fs)
        
        # step 1: extract ECG data, refer to mne.preprocessing.ecg
        skip_by_annotation = ('edge', 'bad') if reject_by_annotation else ()
        del reject_by_annotation
        idx_ecg = _get_ecg_channel_index(ch_name, raw)
        if idx_ecg is not None:
            print('Using channel %s to identify heart beats.'
                        % raw.ch_names[idx_ecg])
            ecg = raw.get_data(picks=idx_ecg)
        else:
            ecg, _ = _make_ecg(raw, start=None, stop=None)
        assert ecg.ndim == 2 and ecg.shape[0] == 1
        
        self.unfiltered_ecg = ecg.copy()
        
        ecg = ecg[0]
        # Deal with filtering the same way we do in raw, i.e. filter each good
        # segment
        self.onsets, self.ends = _annotations_starts_stops(
            raw, skip_by_annotation, 'reject_by_annotation', invert=True)
        mask = (self.ends-self.onsets) > shortest_seg * self.fs
        self.onsets = self.onsets[mask]
        self.ends = self.ends[mask]
        
        self.ecgs = list()
        for si, (start, stop) in enumerate(zip(self.onsets, self.ends)):
            # Only output filter params once (for info level), and only warn
            # once about the length criterion (longest segment is too short)
            # self.ecgs.append(filter_data(
            #     ecg[start:stop], raw.info['sfreq'], l_freq, h_freq, [0],
            #     filter_length, copy=False))
            self.ecgs.append(ecg[start:stop])
    
    def _combine_ecgs(self, ecg_events):
        # map ECG events back to original times
        ecg_len = len(np.concatenate(self.ecgs))
        remap = np.empty(ecg_len, int)
        offset = 0
        for start, stop in zip(self.onsets, self.ends):
            this_len = stop - start
            assert this_len >= 0
            remap[offset:offset + this_len] = np.arange(start, stop)
            offset += this_len
        assert offset == ecg_len

        if ecg_events.size > 0:
            ecg_events = remap[ecg_events]
        else:
            ecg_events = np.array([])

        n_events = len(ecg_events)
        duration_sec = ecg_len / self.fs - self.tstart
        duration_min = duration_sec / 60.
        average_pulse = n_events / duration_min
        print("Number of ECG events detected : %d (average pulse %d / "
                    "min.)" % (n_events, average_pulse))
        return ecg_events
        
    def get_events(self, event_id=999, method='Elgendi et al (Two average)', return_ecg=False, correction=True):
        """Get QRS events from the ECG data."""
        
        ecg_func = dict(self.detectors.get_detector_list())[method]
        ecg_events = []
        for ecg in self.ecgs:
            ecg_events.append(ecg_func(ecg))
        
        event = self._combine_ecgs(ecg_events)
        event = np.stack_columns([event, np.zeros(len(event)), event_id])
        if correction:
            event = qrs_correction(event, raw=self.raw, new_event_idx=event_id)
        if return_ecg:
            return event, self.ecg
        return event
