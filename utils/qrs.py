import numpy as np
import torch
import mne
from mne.annotations import _annotations_starts_stops
from mne.filter import filter_data
from mne.utils import sum_squared
from mne.preprocessing.ecg import _get_ecg_channel_index, _make_ecg

# from ecgdetectors import Detectors

def qrs_correction(ecg_event, raw, max_heart_rate=160, min_heart_rate=40, new_event_idx=998):
    event_timing = ecg_event[:, 0]
    event_length = np.diff(event_timing)
    
    # step 1: calc median and std of event length, with too short & long events ignored
    # long events MUST be ignored as they might be caused by bad segments detection
    min_length = 60.0 / max_heart_rate * raw.info['sfreq']
    max_length = 60.0 / min_heart_rate * raw.info['sfreq']
    inrange_length = event_length[(event_length<max_length) & (event_length>min_length)]
    med, std = np.median(inrange_length), np.std(inrange_length)
        
    # step 2: fp removal: remove events that are too close to each other
    # in fp_flag, the one removed is always the second one for coding laziness
    # in harsh_flag, both are removed for robustness
    low_thres = max(min_length, med-std*3)
    fp_flag = event_length < low_thres
    
    fp_harsh_flag = np.zeros(len(fp_flag) + 1, dtype=bool)
    fp_idx = np.where(fp_flag)[0]
    fp_harsh_flag[fp_idx] = True
    fp_harsh_flag[fp_idx + 1] = True
    fp_harsh_flag[fp_idx - 1] = True
    
    event_removed = []
    last_fp_len = 0     # 0 if last event is not fp, else is len(last event)
    for event_idx in range(len(fp_flag)):
        if fp_flag[event_idx]:
            last_fp_len += event_length[event_idx]
            if last_fp_len > low_thres:
                last_fp_len = 0
            else:
                event_removed.append(event_idx+1)
            continue
        last_fp_len = 0
    
    mask = np.ones(len(fp_flag)+1, dtype=bool)
    mask[event_removed] = False
    event_timing = event_timing[mask]
    
    # the code above and below differs in assumption for a consecutive events of ooooxx..xxooo, where x are fp
    # code above assume the heartbeats is within one of the fps
    # code below is a more conservative version, assuming none fp is usable
    # event_timing = np.insert(event_timing[1:][~fp_flag], 0, event_timing[0])
    
    # step 3: average heartbeat template calculation, only consider safe events
    fn_harsh_flag = np.zeros(len(fp_flag) + 1, dtype=bool)
    fn_idx = np.where(event_length > min(max_length, 1.5*med))[0]
    fn_harsh_flag[fn_idx] = True
    fn_harsh_flag[fn_idx + 1] = True
    fn_harsh_flag[fn_idx - 1] = True
    harsh_flag = np.logical_or(fp_harsh_flag, fn_harsh_flag)
    safe_event = ecg_event[:, 0][~harsh_flag]
    
    search_range = round(med / 3)
    half_win_range = round(med / 2)
    ecg_data =raw.get_data(picks='ecg')
    safe_windows = []
    for ev in safe_event:
        start = ev - half_win_range - raw.first_samp
        end = ev + half_win_range + 1 - raw.first_samp
        if start >= 0 and end <= raw._data.shape[1]:
            safe_windows.append(ecg_data[0, start:end])
    
    tmplt = np.mean(np.array(safe_windows), axis=0).squeeze()
    
    # step 4: move each event a little bit to maximize the pearson correlation with the template
    def pearson_corr(windows: np.ndarray, template: np.ndarray):
        """
        windows: shape (N, L) — N windows of length L
        template: shape (L,) — single QRS template
        returns: shape (N,) — correlation of each window with the template
        """
        # Normalize template
        template_norm = template - template.mean()
        windows_norm = windows - windows.mean(axis=1, keepdims=True)

        return np.sum(windows_norm * template_norm, axis=1) / (windows.std(axis=1)*template.std())

    def align_with_template(event_time_list):  
        event_time_list = event_time_list.astype(np.int64)
        new_event_list = []
        for ev in event_time_list:
            ev_pos = ev - raw.first_samp
            if (ev_pos-search_range-half_win_range<0) or (ev_pos+search_range+half_win_range>raw._data.shape[1]):
                continue
            
            win_pos_list = np.arange(ev_pos-search_range, ev_pos+search_range+1)
            window_arr = np.stack([ecg_data[0, pos-half_win_range:pos+half_win_range+1] for pos in win_pos_list])
            corr = pearson_corr(window_arr, tmplt)
            best_pos = win_pos_list[np.argmax(corr)]
            new_event_list.append([best_pos+raw.first_samp, 0, new_event_idx])
        return np.unique(np.array(new_event_list), axis=0)
    new_event = align_with_template(event_timing)
    
    # step 5: fn removal: add events between two far away events
    # also at start & end of the signal
    # also align them
    event_length = np.diff(new_event[:, 0])
    med = np.median(event_length[(event_length<max_length)])
    thres = min(max_length, 1.5*med)
    fn_pos_list = np.where(event_length>thres)[0]
    fn_list = []
    for fn_pos in fn_pos_list:
        hb_in_between = np.round(event_length[fn_pos] / med)  # number of hearbeats in between, hb_in_between = number of missing events + 1
        fn_length = round(event_length[fn_pos]/hb_in_between)
        fn = (np.arange(1, hb_in_between) * fn_length + new_event[fn_pos][0]).astype(np.int64)
        fn_list.append(fn)
    if new_event[0, 0] - raw.first_samp > thres:
        missing_events = np.floor((new_event[0, 0] - raw.first_samp) / med)
        fn = new_event[0, 0] - (np.arange(missing_events)+1) * med
        fn_list.append(fn)
    if new_event[-1, 0] + thres < raw._data.shape[1]:
        missing_events = np.floor((raw._data.shape[1] - new_event[-1, 0]) / med)
        fn = new_event[-1, 0] + (np.arange(missing_events)+1) * med
        fn_list.append(fn)
    fn_list = np.concatenate(fn_list) if len(fn_list) > 0 else np.array([])
    new_event = np.concatenate([new_event, align_with_template(fn_list)])
    new_event = new_event[np.argsort(new_event[:, 0])]
    return new_event

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
