if ~ischar(EEG.event(1).type)
    for i = 1:length(EEG.event)
      
        if isnumeric(EEG.event(i).type) % Check if it's a number
            if EEG.event(i).type>60000
                EEG.event(i).type = 66666;
            end
            EEG.event(i).type = num2str(EEG.event(i).type); % Convert to char list
        end
    end
    tmpevent = EEG.event;
end