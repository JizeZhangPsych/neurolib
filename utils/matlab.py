import os, shutil
import mne.io
import scipy.io
import matlab.engine


class MatlabInstance:
    def __init__(self, plugin_list) -> None:
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('/ohba/pi/mwoolrich/jzhang/neurolib')
        for plugin_name in plugin_list:
            self.plugin(plugin_name)
    
    def plugin(self, name='eeglab'):
        if name == 'eeglab':
            self.eng.addpath('/ohba/pi/mwoolrich/jzhang/eeglab')
            self.eng.eeglab()
        else:
            raise NotImplementedError()
    
    def run(self, function_parts, name='foobar.mat', mode='to_python'):
        if mode == 'disk':
            self.eng.matlabfunc2disk(function_parts, name, nargout=1)
        elif mode == 'to_python':
            self.eng.matlabfunc2disk(function_parts, name, nargout=1)
            return_mat = scipy.io.loadmat(name)
            os.remove(name)
            return return_mat
        else:
            raise NotImplementedError()
        
    def end(self):
        self.eng.quit()
        del self.eng
        del self

