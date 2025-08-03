import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt

# --- Configuration based on the notebook ---
# We are analyzing session 2 (index 1), which contains the noise trials.
SUBJECT_INDEX = 1
SESSION_INDEX = 1 
SAMPLING_RATE = 1000
# NOTE: You can change the noise_level here to a specific float (e.g., 0.5, 0.7)
# or use 'all' to combine all noise levels.
NOISE_LEVEL_TO_PLOT = 'all' # 0.0,  0.1,  0.2,  ...  ,  1.00  or 'all'

# NOTE: Face/House selective channels might differ in this session,
# but we use the old ones for consistency in the example plots.
FACE_CH_IDX = 46
HOUSE_CH_IDX = 43

def load_ecog_data(filepath, subject_idx, session_idx):
    """
    Loads ECoG data for a specific subject and session from the .npz file.
    """
    print(f"Loading data for subject {subject_idx}, session {session_idx}...")
    try:
        alldat = np.load(filepath, allow_pickle=True)['dat']
        subject_data = alldat[subject_idx][session_idx]
        print("Data loaded successfully.")
        return subject_data
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

def create_mne_raw_object(session_data, sfreq):
    """
    Creates a complete MNE Raw object from the session data, including
    channel info, montage, and detailed stimulus annotations with noise levels.

    Args:
        session_data (dict): The dictionary loaded by load_ecog_data.
        sfreq (int): The sampling frequency of the data.

    Returns:
        mne.io.RawArray: A fully configured MNE Raw object.
    """
    print("Creating complete MNE Raw object...")
    # --- Create the basic Raw object ---
    data = session_data['V'].T
    n_channels = data.shape[0]
    ch_names = [f"ECOG_{i+1:03}" for i in range(n_channels)]
    ch_types = ['ecog'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # --- Add electrode locations (montage) ---
    locs = session_data['locs']
    ch_pos = {ch_names[i]: locs[i] for i in range(n_channels)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='mni_tal')
    raw.set_montage(montage)
    
    # --- Add stimulus onsets as annotations with noise levels ---
    print("Adding detailed stimulus annotations with noise levels...")
    onsets_in_seconds = session_data['t_on'] / sfreq
    # CORRECTED: Duration for session 2 is 1000ms (1.0s)
    durations = np.full_like(onsets_in_seconds, 1.0)
    
    # Create descriptive labels like "face/noise_0.7"
    descriptions = []
    # CORRECTED: Use the correct keys for the noise session
    stim_categories = session_data['stim_cat']
    noise_levels = session_data['stim_noise']
    for i, category_id in enumerate(stim_categories):
        # CORRECTED: 1 = house, 2 = face
        category = 'house' if category_id == 1 else 'face'
        # CORRECTED: Convert noise from 0-100 to 0-1 scale and squeeze
        noise = np.squeeze(noise_levels[i]) / 100.0
        descriptions.append(f"{category}/noise_{noise:.1f}")
    
    annotations = mne.Annotations(onset=onsets_in_seconds,
                                  duration=durations,
                                  description=descriptions)
    raw.set_annotations(annotations)
    
    print("Complete MNE Raw object created.")
    return raw

def preprocess_ecog(raw, params):
    """
    Applies standard preprocessing steps to ECoG data.
    - Band-pass filter
    - Notch filter for power line noise and its harmonics
    - Re-references to common average reference (CAR)
    
    Args:
        raw (mne.io.Raw): The raw MNE object.
        params (dict): A dictionary of preprocessing parameters.
    
    Returns:
        mne.io.Raw: The preprocessed MNE Raw object.
    """
    print("Applying ECoG preprocessing...")
    raw_copy = raw.copy()

    # 1. Apply a band-pass filter
    if 'l_freq' in params and 'h_freq' in params:
        print(f"Applying band-pass filter from {params['l_freq']} to {params['h_freq']} Hz...")
        raw_copy.filter(l_freq=params['l_freq'], h_freq=params['h_freq'])

    # 2. Apply a notch filter for power line noise and its harmonics
    if 'notch_freqs' in params:
        print(f"Applying notch filter at {params['notch_freqs']} Hz...")
        raw_copy.notch_filter(freqs=params['notch_freqs'])

    # 3. Re-reference to common average reference (CAR)
    if params.get('rereference') == 'car':
        print("Applying Common Average Reference (CAR)...")
        raw_copy.set_eeg_reference('average', projection=False)

    print("Preprocessing complete.")
    return raw_copy

def extract_broadband_power(raw):
    """
    Extracts high-gamma broadband power and returns it as a new MNE Raw object.
    
    Args:
        raw (mne.io.Raw): The input MNE Raw object.
    
    Returns:
        mne.io.Raw: A new Raw object containing the broadband power data.
    """
    print("Extracting broadband power...")
    sfreq = raw.info['sfreq']
    V = raw.get_data().T.astype('float32') 
    
    # --- Signal processing to get broadband power envelope ---
    b, a = signal.butter(3, [50], btype='high', fs=sfreq)
    V = signal.filtfilt(b, a, V, axis=0)
    V = np.abs(V)**2
    b, a = signal.butter(3, [10], btype='low', fs=sfreq)
    V = signal.filtfilt(b, a, V, axis=0)
    V = V / V.mean(axis=0)
    
    # --- Create a new MNE Raw object for the broadband data ---
    info = raw.info.copy()
    info['description'] = 'Broadband Power'
    # MNE expects data as (channels, time)
    raw_broadband = mne.io.RawArray(V.T, info)
    # Carry over the annotations to the new Raw object
    raw_broadband.set_annotations(raw.annotations)
    
    print("Broadband power extraction complete.")
    return raw_broadband

def plot_broadband_evoked(raw_broadband, tmin=-0.2, tmax=0.4, n_channels_to_plot=50, noise_level='all'):
    """
    Calculates and plots the evoked broadband response for each channel,
    with an option to select a specific noise level.
    
    Args:
        raw_broadband (mne.io.Raw): The MNE Raw object containing broadband power.
        tmin (float): Start time of the epoch in seconds.
        tmax (float): End time of the epoch in seconds.
        n_channels_to_plot (int): Number of channels to include in the plot.
        noise_level (str or float): The noise level to plot, or 'all'.
    """
    print(f"Calculating and plotting evoked broadband responses for noise level: {noise_level}...")
    events, event_id = mne.events_from_annotations(raw_broadband)
    epochs = mne.Epochs(raw_broadband, events, event_id, tmin=tmin, tmax=tmax,
                        preload=True, baseline=(None, 0))
    
    title_suffix = ""
    try:
        if noise_level == 'all':
            evoked_face = epochs['face'].average()
            evoked_house = epochs['house'].average()
            title_suffix = "(All Noise Levels)"
        else:
            face_query = f'face/noise_{noise_level:.1f}'
            house_query = f'house/noise_{noise_level:.1f}'
            evoked_face = epochs[face_query].average()
            evoked_house = epochs[house_query].average()
            title_suffix = f"(Noise Level: {noise_level:.1f})"
    except KeyError:
        print(f"Warning: Noise level {noise_level} not found in data. Skipping plot.")
        return

    plt.figure(figsize=(20, 10))
    for i in range(n_channels_to_plot):
        ax = plt.subplot(5, 10, i + 1)
        ax.plot(evoked_house.times * 1000, evoked_house.data[i], label='House')
        ax.plot(evoked_face.times * 1000, evoked_face.data[i], label='Face')
        ax.set_title(raw_broadband.ch_names[i])
        ax.set_ylim([0, 4])
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Normalized Power')
            ax.set_xlabel('Time (ms)')
    
    plt.suptitle(f"Average Broadband Response to Faces vs. Houses {title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_broadband_epochs_image(raw_broadband, face_ch_idx, house_ch_idx, tmin=-0.2, tmax=0.4, noise_level='all'):
    """
    Plots heatmaps of all trials for specified channels using MNE's plot_image,
    with an option to select a specific noise level.
    
    Args:
        raw_broadband (mne.io.Raw): The MNE Raw object containing broadband power.
        face_ch_idx (int): Index of the face-selective channel.
        house_ch_idx (int): Index of the house-selective channel.
        tmin (float): Start time of the epoch in seconds.
        tmax (float): End time of the epoch in seconds.
        noise_level (str or float): The noise level to plot, or 'all'.
    """
    print(f"Plotting broadband trial images for selective channels for noise level: {noise_level}...")
    events, event_id = mne.events_from_annotations(raw_broadband)
    epochs = mne.Epochs(raw_broadband, events, event_id, tmin=tmin, tmax=tmax,
                        preload=True, baseline=(None, 0))
    
    title_suffix = ""
    epochs_to_plot = epochs
    try:
        if noise_level != 'all':
            # Use a formatted string that matches the annotation description
            query = f'noise_{noise_level:.2f}' 
            # Check for both face and house categories with this noise level
            if any(query in desc for desc in epochs.event_id):
                 epochs_to_plot = epochs[query]
                 title_suffix = f" (Noise Level: {noise_level:.2f})"
            else:
                # If a specific noise level is requested but not found, we can be more specific
                print(f"Warning: Events with noise level query '{query}' not found in data. Plotting all noise levels instead.")
                title_suffix = " (All Noise Levels - Requested level not found)"
        else:
            title_suffix = " (All Noise Levels)"
    except KeyError:
        # This catch is for cases where the query itself is malformed or another issue arises
        print(f"Warning: Could not select epochs for noise level {noise_level}. Plotting all available epochs.")
        title_suffix = " (All Noise Levels - Error in selection)"

    # Create a specific title and plot for the face-selective channel
    face_title = f'Face-Selective Channel (Index: {face_ch_idx}){title_suffix}'
    epochs_to_plot.plot_image(picks=[face_ch_idx],
                              order=np.argsort(epochs_to_plot.events[:, 2]), # Sort by event type
                              vmin=0, vmax=7,
                              cmap='magma',
                              title=face_title)

    # Create a specific title and plot for the house-selective channel
    house_title = f'House-Selective Channel (Index: {house_ch_idx}){title_suffix}'
    epochs_to_plot.plot_image(picks=[house_ch_idx],
                              order=np.argsort(epochs_to_plot.events[:, 2]), # Sort by event type
                              vmin=0, vmax=7,
                              cmap='magma',
                              title=house_title)

    plt.show()

    
def plot_channel_erps(raw, tmin=-0.2, tmax=1.0, n_channels_to_plot=50, noise_level='all'):
    """
    Calculates and plots the ERP for each channel, comparing faces and houses,
    with an option to select a specific noise level.
    
    Args:
        raw (mne.io.Raw): The preprocessed MNE Raw object.
        tmin (float): Start time of the epoch in seconds.
        tmax (float): End time of the epoch in seconds.
        n_channels_to_plot (int): Number of channels to include in the plot.
        noise_level (str or float): The noise level to plot, or 'all'.
    """
    print(f"Calculating and plotting ERPs for noise level: {noise_level}...")
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                        preload=True, baseline=(None, 0))
    
    title_suffix = ""
    try:
        if noise_level == 'all':
            evoked_face = epochs['face'].average()
            evoked_house = epochs['house'].average()
            title_suffix = "(All Noise Levels)"
        else:
            face_query = f'face/noise_{noise_level:.1f}'
            house_query = f'house/noise_{noise_level:.1f}'
            evoked_face = epochs[face_query].average()
            evoked_house = epochs[house_query].average()
            title_suffix = f"(Noise Level: {noise_level:.1f})"
    except KeyError:
        print(f"Warning: Noise level {noise_level} not found in data. Skipping plot.")
        return
        
    plt.figure(figsize=(20, 10))
    for i in range(n_channels_to_plot):
        ax = plt.subplot(5, 10, i + 1)
        ax.plot(evoked_house.times * 1000, evoked_house.data[i] * 1e6, label='House')
        ax.plot(evoked_face.times * 1000, evoked_face.data[i] * 1e6, label='Face')
        ax.set_title(raw.ch_names[i])
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Voltage (µV)')
            ax.set_xlabel('Time (ms)')
    
    plt.suptitle(f"Event-Related Potentials (ERPs) to Faces vs. Houses {title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    filepath = r'C:\Users\Mohammadreza\Desktop\Neuromatch CN\Data\faceshouses.npz'
    session_data = load_ecog_data(filepath, SUBJECT_INDEX, SESSION_INDEX)

    if session_data:
        
        # 1. Create a complete MNE Raw object with data, montage, and detailed annotations
        raw = create_mne_raw_object(session_data, SAMPLING_RATE)

        
        # --- BROADBAND ANALYSIS (MNE-based) ---
        print("\n--- Starting Broadband Power Analysis ---")
        # 2. Extract broadband power as a new Raw object
        raw_broadband = extract_broadband_power(raw)
        
        # 3. Plot the evoked broadband responses and trial images
        # Plot for the specified noise level
        plot_broadband_evoked(raw_broadband, n_channels_to_plot=50, noise_level=NOISE_LEVEL_TO_PLOT)
        # Plot the trial images for the specified noise level
        plot_broadband_epochs_image(raw_broadband, FACE_CH_IDX, HOUSE_CH_IDX, noise_level=NOISE_LEVEL_TO_PLOT)
        #

        
        # --- ERP ANALYSIS (New functionality) ---
        print("\n--- Starting Event-Related Potential (ERP) Analysis ---")
        # 4. Define preprocessing parameters
        preprocess_params = {
            'l_freq': 0.5,
            'h_freq': 300.0,
            'notch_freqs': np.arange(60, 241, 60),
            'rereference': 'car'
        }
        
        # 5. Apply preprocessing
        raw_preprocessed = preprocess_ecog(raw.copy(), preprocess_params)

        raw_preprocessed.compute_psd().plot()
        input()
        
        # 6. Calculate and plot ERPs
        # Plot for the specified noise level
        plot_channel_erps(raw_preprocessed, n_channels_to_plot=50, noise_level=NOISE_LEVEL_TO_PLOT)
        
