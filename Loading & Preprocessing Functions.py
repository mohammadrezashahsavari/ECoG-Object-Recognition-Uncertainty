# ecog_utils.py
#
# This script is a utility module. It is NOT meant to be run directly.
# It contains functions for loading, converting, and preprocessing ECoG data.
#
# To use these functions, import them into your own analysis script
# or Jupyter Notebook. See the example usage guide at the bottom of this file.
# ------------------------------------------------------------------------------

import numpy as np
import mne

def load_ecog_data(filepath, subject_idx, session_idx):
    """
    Loads ECoG data for a specific subject and session from the .npz file.

    Args:
        filepath (str): Path to the 'faceshouses.npz' data file.
        subject_idx (int): The index of the subject to load.
        session_idx (int): The index of the session to load.

    Returns:
        dict or None: A dictionary with the session data, or None if loading fails.
    """
    print(f"Loading data for subject {subject_idx}, session {session_idx}...")
    try:
        # Load the .npz file and access the main data structure via the 'dat' key.
        alldat = np.load(filepath, allow_pickle=True)['dat']
        subject_data = alldat[subject_idx][session_idx]
        print("Data loaded successfully.")
        return subject_data
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except IndexError:
        print(f"Error: Subject index {subject_idx} or session index {session_idx} is out of bounds.")
        return None

def create_mne_raw_object(session_data, sfreq):
    """
    Creates a complete MNE Raw object from session data.

    This function packages the voltage data, electrode locations (montage),
    and stimulus event markers (annotations) into a standardized MNE object.

    Args:
        session_data (dict): The dictionary loaded by `load_ecog_data`.
        sfreq (int): The sampling frequency of the data (in Hz).

    Returns:
        mne.io.RawArray: A fully configured MNE Raw object.
    """
    print("Creating MNE Raw object...")
    # MNE expects data in shape (n_channels, n_times), so we transpose the voltage array.
    data = session_data['V'].T
    n_channels = data.shape[0]

    # Create standard metadata
    ch_names = [f"ECOG_{i+1:03}" for i in range(n_channels)]
    ch_types = ['ecog'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create the core MNE Raw object
    raw = mne.io.RawArray(data, info)

    # Add electrode locations (montage) for spatial analysis and plotting
    print("  - Adding electrode locations (montage)...")
    locs = session_data['locs']
    ch_pos = {ch_names[i]: locs[i] for i in range(n_channels)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='mni_tal')
    raw.set_montage(montage)
    
    # Add stimulus onsets and descriptions as annotations
    print("  - Adding stimulus annotations...")
    onsets_in_seconds = session_data['t_on'] / sfreq
    durations = np.full_like(onsets_in_seconds, 1.0) # Each stimulus has a 1s duration
    
    descriptions = []
    stim_categories = session_data['stim_cat'] # 1=house, 2=face
    noise_levels = session_data['stim_noise'] # 0-100 scale
    for i, category_id in enumerate(stim_categories):
        category = 'house' if category_id == 1 else 'face'
        noise = np.squeeze(noise_levels[i]) / 100.0 # Convert to 0-1 scale
        descriptions.append(f"{category}/noise_{noise:.1f}")
    
    annotations = mne.Annotations(onset=onsets_in_seconds, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    
    print("MNE Raw object created successfully.")
    return raw


def preprocess_ecog(raw, params):
    """
    Applies a standard preprocessing pipeline to an MNE Raw object.

    The pipeline includes:
    - Removing specified channels
    - Band-pass filtering
    - Notch filtering
    - Re-referencing

    Args:
        raw (mne.io.Raw): The MNE Raw object to preprocess.
        params (dict): A dictionary of preprocessing parameters.
                       Keys: 'l_freq', 'h_freq', 'notch_freqs', 'rereference', 'exclude_channels'.

    Returns:
        mne.io.Raw: The preprocessed MNE Raw object.
    """
    print("Applying ECoG preprocessing...")
    # Work on a copy to keep the original data intact
    raw_copy = raw.copy()

    # 0. Remove specified channels if provided
    if 'exclude_channels' in params and params['exclude_channels']:
        # Find which of the channels to exclude are present in the data
        channels_to_drop = [ch for ch in params['exclude_channels'] if ch in raw_copy.ch_names]
        
        if channels_to_drop:
            print(f"  - Removing channels: {channels_to_drop}")
            raw_copy.drop_channels(channels_to_drop)
        else:
            print("  - Note: Channels specified in 'exclude_channels' not found in data.")

    # 1. Apply a band-pass filter
    if 'l_freq' in params and 'h_freq' in params and params['l_freq'] is not None and params['h_freq'] is not None:
        print(f"  - Applying band-pass filter: {params['l_freq']}-{params['h_freq']} Hz")
        raw_copy.filter(l_freq=params['l_freq'], h_freq=params['h_freq'])

    # 2. Apply a notch filter for power line noise and its harmonics
    if 'notch_freqs' in params and params['notch_freqs'] is not None:
        print(f"  - Applying notch filter at: {params['notch_freqs']} Hz")
        raw_copy.notch_filter(freqs=params['notch_freqs'])

    # 3. Re-reference to common average reference (CAR)
    if params.get('rereference') == 'car':
        print("  - Applying Common Average Reference (CAR)")
        raw_copy.set_eeg_reference('average', projection=False)

    print("Preprocessing complete.")
    return raw_copy


# =====================================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
#                  EXAMPLE USAGE (COPY THIS INTO YOUR ANALYSIS SCRIPT)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# =====================================================================================
if __name__ == '__main__':
    # This block is for demonstration purposes only.
    # To use the functions, import them into your own script or notebook as shown below.
    
    # --------------------------------------------------------------------------------
    # In your analysis script (e.g., `my_analysis.py` or a Jupyter cell), do the following:
    # --------------------------------------------------------------------------------
    
    # --- 1. Import the functions from this file ---
    # Make sure `ecog_utils.py` is in the same directory or in your Python path.
    # from ecog_utils import load_ecog_data, create_mne_raw_object, preprocess_ecog
    
    # --- 2. Define your settings ---
    FILEPATH = r'C:\path\to\your\data\faceshouses.npz'  # 👈 IMPORTANT: Change this path!
    SAMPLING_RATE = 1000  # Hz
    
    # Define preprocessing parameters - you can create multiple dictionaries to test different pipelines.
    PREPROCESS_PARAMS_A = {
    'l_freq': 0.5,
    'h_freq': 290.0,
    #'notch_freqs': [60, 120, 180, 240],
    #'rereference': 'car',
    'exclude_channels': ['ECoG1', 'ECoG2', 'ECoG3'] # Channels to be removed
    }
    
    # --- 3. HOW TO PROCESS A SINGLE SUBJECT ---
    print("--- Example: Processing a single subject ---")
    subject_id = 0
    session_id = 1  # Session 0 = noiseless, Session 1 = noisy trials
    
    # Load the specified subject's data
    session_data = load_ecog_data(FILEPATH, subject_id, session_id)
    
    if session_data:
        # Convert the loaded data into an MNE Raw object
        raw = create_mne_raw_object(session_data, SAMPLING_RATE)
        
        # Apply the preprocessing pipeline
        raw_preprocessed = preprocess_ecog(raw, PREPROCESS_PARAMS_A)
        
        print(f"\n✅ Successfully processed Subject {subject_id}, Session {session_id}.")
        print("Preprocessed MNE Raw object is now ready for analysis:")
        print(raw_preprocessed)
        # You can now use `raw_preprocessed` for epoching, ERPs, time-frequency analysis, etc.
        # raw_preprocessed.plot_psd()
    
    
    # --- 4. HOW TO LOOP OVER ALL SUBJECTS AND SESSIONS ---
    print("\n\n--- Example: Looping over all subjects and sessions ---")
    
    # The dataset has 2 subjects (indices 0, 1) and 2 sessions (indices 0, 1)
    N_SUBJECTS = 2
    N_SESSIONS = 2
    
    all_preprocessed_data = {}  # A dictionary is a good way to store the results
    
    for sub_idx in range(N_SUBJECTS):
        all_preprocessed_data[sub_idx] = {} # Create a nested dictionary for sessions
        for ses_idx in range(N_SESSIONS):
            print(f"\n🔄 Processing Subject {sub_idx}, Session {ses_idx}...")
            
            # Step A: Load data
            current_session_data = load_ecog_data(FILEPATH, sub_idx, ses_idx)
            
            if current_session_data:
                # Step B: Create MNE object
                raw_obj = create_mne_raw_object(current_session_data, SAMPLING_RATE)
                
                # Step C: Preprocess data with your chosen parameters
                preprocessed_obj = preprocess_ecog(raw_obj, PREPROCESS_PARAMS_A)
                
                # Step D: Store the final object for later use
                all_preprocessed_data[sub_idx][ses_idx] = preprocessed_obj
                print(f"✅ Finished and stored data for Subject {sub_idx}, Session {ses_idx}.")
                
    print("\n\nLoop finished! All preprocessed data is stored in the `all_preprocessed_data` dictionary.")
    # You can now easily access any preprocessed data, for example:
    # subject_1_session_0_data = all_preprocessed_data[1][0]
    # print(subject_1_session_0_data)