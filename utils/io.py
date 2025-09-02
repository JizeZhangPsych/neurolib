import os
import pickle


def load_pkl(file_path):
    """Load data using pickle.

    Parameters
    ----------
    file_path : str
        File path containing the data to be loaded.

    Returns
    -------
    data : Any
        Loaded data.
    """

    # Load input data
    with open(file_path, "rb") as input_path:
        data = pickle.load(input_path)
    input_path.close()
    
    return data

def save_pkl(data, file_path):
    """Save data using pickle.

    Parameters
    ----------
    data : Any
        Data object to be saved.
    file_path : str
        File path where the data will be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save input data
    with open(file_path, "wb") as save_path:
        pickle.dump(data, save_path)
    save_path.close()