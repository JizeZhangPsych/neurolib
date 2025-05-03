function processed_data = fastr(eeg_path, varargin)
    eegdata, header = readedf(eeg_path);
    EEG = struct();
    EEG.data = eegdata;
    EEG.srate = header.samplerate(1);
    EEG.pnts = size(eegdata, 2);
    EEG.events = [];
    processed_data = fmrib_fastr(EEG, varargin{:});
end