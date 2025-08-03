#!/usr/bin/env python3
"""
Utility functions for MVPA analysis of ECoG data
"""

import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
import warnings
import scipy.stats as stats
from nilearn import plotting
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cv2
import scipy.signal as signal
warnings.filterwarnings('ignore')


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
    print(" - Adding electrode locations (montage)...")
    locs = session_data['locs']
    ch_pos = {ch_names[i]: locs[i] for i in range(n_channels)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='mni_tal')
    raw.set_montage(montage)
    
    # Add stimulus onsets and descriptions as annotations
    print(" - Adding stimulus annotations...")
    onsets_in_seconds = session_data['t_on'] / sfreq
    durations = np.full_like(onsets_in_seconds, 1.0) # Each stimulus has a 1s duration
    
    descriptions = []
    stim_categories = session_data['stim_cat'] # 1=house, 2=face
    noise_levels = session_data['stim_noise'] # 0-100 scale
    for i, category_id in enumerate(stim_categories):
        category = 'house' if category_id == 1 else 'face'
        noise = np.squeeze(noise_levels[i]) / 100.0 # Convert to 0-1 scale
        # Round to nearest 5% (0.05) for 5% resolution
        noise_rounded = np.round(noise * 20) / 20  # 20 bins for 5% resolution
        descriptions.append(f"{category}/noise_{noise_rounded:.2f}")
    
    annotations = mne.Annotations(onset=onsets_in_seconds, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    
    print("MNE Raw object created successfully.")
    return raw


def preprocess_ecog(raw, params):
    """
    Applies a standard preprocessing pipeline to an MNE Raw object.
    Each step is controlled by the presence and value of keys in the `params` dictionary.
    If a parameter is set to None, the corresponding step is skipped.

    The pipeline includes:
    - Resampling
    - Removing specified channels
    - Band-pass, high-pass, or low-pass filtering
    - Notch filtering
    - Re-referencing

    Args:
        raw (mne.io.Raw): The MNE Raw object to preprocess.
        params (dict): A dictionary of preprocessing parameters.
                       Keys: 'resample_rate', 'exclude_channels', 'l_freq', 
                             'h_freq', 'notch_freqs', 'rereference'.

    Returns:
        mne.io.Raw: The preprocessed MNE Raw object.
    """
    print("Applying ECoG preprocessing...")
    # Work on a copy to keep the original data intact
    raw_copy = raw.copy()

    # 1. Resample data if 'resample_rate' is specified
    resample_rate = params.get('resample_rate')
    if resample_rate is not None:
        print(f"  - Resampling data to {resample_rate} Hz...")
        raw_copy.resample(sfreq=resample_rate)

    # 2. Remove specified channels if provided
    exclude_channels = params.get('exclude_channels')
    if exclude_channels:
        # Find which of the channels to exclude are present in the data
        channels_to_drop = [ch for ch in exclude_channels if ch in raw_copy.ch_names]
        if channels_to_drop:
            print(f"  - Removing channels: {channels_to_drop}")
            raw_copy.drop_channels(channels_to_drop)
        else:
            print("  - Note: Channels specified in 'exclude_channels' not found in data.")

    # 3. Apply filtering (band-pass, high-pass, or low-pass)
    l_freq = params.get('l_freq')
    h_freq = params.get('h_freq')
    
    # Proceed if at least one frequency is specified
    if l_freq is not None or h_freq is not None:
        if l_freq is not None and h_freq is not None:
            print(f"  - Applying band-pass filter: {l_freq}-{h_freq} Hz")
        elif l_freq is not None:
            print(f"  - Applying high-pass filter: {l_freq} Hz")
        else:  # h_freq must be not None
            print(f"  - Applying low-pass filter: {h_freq} Hz")
        
        # Apply the filter
        raw_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    # 4. Apply a notch filter for power line noise and its harmonics
    notch_freqs = params.get('notch_freqs')
    if notch_freqs is not None:
        print(f"  - Applying notch filter at: {notch_freqs} Hz")
        raw_copy.notch_filter(freqs=notch_freqs)

    # 5. Re-reference to common average reference (CAR)
    if params.get('rereference') == 'car':
        print("  - Applying Common Average Reference (CAR)")
        raw_copy.set_eeg_reference('average', projection=False)

    print("Preprocessing complete.")
    return raw_copy


def load_subject_data(config, subject_idx):
    """Load and preprocess data for a single subject"""
    print(f"Loading Subject {subject_idx}, Session {config['session_idx']}...")
    
    # Load raw data
    session_data = load_ecog_data(config['filepath'], subject_idx, config['session_idx'])
    
    if session_data is not None:
        # Create MNE object
        raw = create_mne_raw_object(session_data, config['original_sampling_rate'])
        
        # Preprocess
        raw_preprocessed = preprocess_ecog(raw, config['preprocess_params'])
        
        # Apply processing mode
        processing_mode = config.get('processing_mode', 'ecog')
        
        if processing_mode == 'high_gamma':
            print("Processing mode: High Gamma (Broadband)")
            raw_processed = extract_broadband_power(raw_preprocessed)
        elif processing_mode == 'ecog':
            print("Processing mode: ECoG Signals")
            raw_processed = raw_preprocessed
        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}. Use 'ecog' or 'high_gamma'")
        
        return raw_processed
    else:
        print(f"Failed to load data for Subject {subject_idx}")
        return None



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


def generate_output_folder_name(config):
    """
    Generate output folder name based on configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        str: Formatted folder name
    """
    folder_parts = []
    
    # Base name
    folder_parts.append("mvpa_results")
    
    # Processing mode
    mode = config.get('processing_mode', 'ecog')
    if mode == 'high_gamma':
        folder_parts.append("high_gamma")
    else:
        folder_parts.append("ecog")
    
    # Bandpass frequencies
    preprocess_params = config.get('preprocess_params', {})
    l_freq = preprocess_params.get('l_freq')
    h_freq = preprocess_params.get('h_freq')
    
    if l_freq is not None and h_freq is not None:
        folder_parts.append(f"{l_freq}to{h_freq}Hz")
    elif l_freq is not None:
        folder_parts.append(f"hp{l_freq}Hz")
    elif h_freq is not None:
        folder_parts.append(f"lp{h_freq}Hz")
    else:
        folder_parts.append("raw")
    
    # Sampling rate
    resample_rate = config.get('resample_rate')
    if resample_rate:
        folder_parts.append(f"{resample_rate}Hz")
    
    # Noise range
    noise_range = config.get('noise_range')
    if noise_range is not None:
        min_noise, max_noise = noise_range
        folder_parts.append(f"noise{min_noise:.2f}to{max_noise:.2f}")
    
    # MVPA parameters
    window_ms = config.get('window_length_ms')
    stride_ms = config.get('stride_ms')
    if window_ms and stride_ms:
        folder_parts.append(f"win{window_ms}ms_stride{stride_ms}ms")
    
    return " - ".join(folder_parts)



class MVPAAnalyzer:
    """
    Multi-Variable Pattern Analysis for ECoG data using sliding window SVM
    """
    
    def __init__(self, config):
        self.config = config
        self.sfreq = config.get('resample_rate', config.get('sampling_rate', 1000))
        
        # Convert time parameters to samples
        self.window_length_samples = int(config['window_length_ms'] * self.sfreq / 1000)
        self.stride_samples = int(config['stride_ms'] * self.sfreq / 1000)
        self.baseline_start_samples = int(config['baseline_start_ms'] * self.sfreq / 1000)
        self.baseline_end_samples = int(config['baseline_end_ms'] * self.sfreq / 1000)
        self.analysis_start_samples = int(config['analysis_start_ms'] * self.sfreq / 1000)
        self.analysis_end_samples = int(config['analysis_end_ms'] * self.sfreq / 1000)
        
        processing_mode = config.get('processing_mode', 'ecog')
        print(f"MVPA Configuration:")
        print(f"  Processing mode: {processing_mode}")
        print(f"  Sampling rate: {self.sfreq} Hz")
        print(f"  Window length: {self.window_length_samples} samples ({config['window_length_ms']} ms)")
        print(f"  Stride: {self.stride_samples} samples ({config['stride_ms']} ms)")
        print(f"  Repetitions: {config['n_repetitions']}")
        print(f"  Test size: {config['test_size']}")

    def analyze_subject(self, raw_data, subject_idx):
        """Analyze all channels for a single subject"""
        processing_mode = self.config.get('processing_mode', 'ecog')
        print(f"Extracting epochs for Subject {subject_idx} (Mode: {processing_mode})...")
        
        # Extract epochs and labels (with noise range filtering if specified)
        noise_range = self.config.get('noise_range', None)
        epochs_data, labels, times = self.extract_epochs_and_labels(raw_data, noise_range)
        
        if epochs_data is None:
            print("No valid epochs found!")
            return None
        
        # Get electrode locations
        electrode_locs = self.extract_electrode_locations(raw_data)
        
        print(f"Analyzing {epochs_data.shape[1]} channels individually...")
        
        subject_results = {
            'subject_idx': subject_idx,
            'n_channels': epochs_data.shape[1],
            'n_trials': epochs_data.shape[0],
            'processing_mode': processing_mode,
            'noise_range': noise_range,
            'electrode_locations': electrode_locs,
            'channel_results': {},
            'all_channels_result': None,
            'config_snapshot': {
                'window_length_ms': self.config['window_length_ms'],
                'stride_ms': self.config['stride_ms'],
                'n_repetitions': self.config['n_repetitions'],
                'bandpass': f"{self.config['preprocess_params'].get('l_freq', 'None')}-{self.config['preprocess_params'].get('h_freq', 'None')} Hz",
                'sampling_rate': self.sfreq
            }
        }
        
        # Analyze each channel individually
        for channel_idx, channel_name in enumerate(tqdm(raw_data.ch_names, desc="Individual channels")):
            channel_results = self.analyze_channel(
                epochs_data, labels, channel_idx, channel_name
            )
            
            if channel_results is not None:
                subject_results['channel_results'][channel_name] = channel_results
        
        # Analyze all channels together
        all_channels_result = self.analyze_all_channels(epochs_data, labels)
        if all_channels_result is not None:
            subject_results['all_channels_result'] = all_channels_result
        
        print(f"Completed analysis for {len(subject_results['channel_results'])} individual channels")
        print(f"Completed all-channels analysis: {'Yes' if all_channels_result else 'No'}")
        
        return subject_results
    
    def extract_epochs_and_labels(self, raw, noise_range=None):
        """
        Extract epochs and labels from MNE Raw object
        
        Args:
            raw: MNE Raw object
            noise_range: Noise range to filter as [min, max] (0.0-1.0), None for all levels
            
        Returns:
            tuple: (epochs_data, labels, times)
        """
        # Get annotations (stimulus events)
        annotations = raw.annotations
        
        # Extract stimulus onsets and categories
        onsets = annotations.onset
        descriptions = annotations.description
        
        # Parse labels and filter by noise range
        labels = []
        valid_onsets = []
        
        for i, desc in enumerate(descriptions):
            # Parse description: e.g., 'house/noise_0.70'
            if '/' in desc and 'noise_' in desc:
                category_part, noise_part = desc.split('/')
                category = category_part.lower()
                
                # Extract noise level
                try:
                    current_noise = float(noise_part.split('_')[1])
                except (IndexError, ValueError):
                    continue
                
                # Filter by noise range if specified
                if noise_range is not None:
                    min_noise, max_noise = noise_range
                    if current_noise < min_noise or current_noise > max_noise:
                        continue
                
                # Add to valid trials
                if 'house' in category:
                    labels.append(0)
                    valid_onsets.append(onsets[i])
                elif 'face' in category:
                    labels.append(1)
                    valid_onsets.append(onsets[i])
        
        labels = np.array(labels)
        valid_onsets = np.array(valid_onsets)
        
        if len(labels) == 0:
            range_str = f" (noise range {noise_range})" if noise_range is not None else ""
            print(f" ✗ No valid stimulus events found{range_str}!")
            return None, None, None
        
        range_str = f" in noise range {noise_range}" if noise_range is not None else ""
        print(f" Found {len(labels)} valid trials{range_str}: {np.sum(labels == 0)} houses, {np.sum(labels == 1)} faces")
        
        # Create epochs
        tmin = self.config['baseline_start_ms'] / 1000 # Convert to seconds
        tmax = self.config['analysis_end_ms'] / 1000
        
        # Create events array for MNE
        events = []
        for i, onset in enumerate(valid_onsets):
            sample_idx = int(onset * self.sfreq)
            events.append([sample_idx, 0, labels[i] + 1]) # MNE expects event_id > 0
        
        events = np.array(events)
        event_id = {'house': 1, 'face': 2}
        
        # Create epochs
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)
        
        # Get data and adjust labels back to 0/1
        epochs_data = epochs.get_data() # Shape: (n_epochs, n_channels, n_times)
        epoch_labels = epochs.events[:, 2] - 1 # Convert back to 0/1
        
        # Get time vector
        times = epochs.times
        
        return epochs_data, epoch_labels, times
    
    def extract_electrode_locations(self, raw_data):
        """Extract electrode locations from MNE Raw object"""
        electrode_locs = {}
        
        # Try to get locations from montage first
        if raw_data.info['dig'] is not None:
            for ch_name in raw_data.ch_names:
                ch_idx = raw_data.ch_names.index(ch_name)
                if ch_idx < len(raw_data.info['chs']):
                    loc = raw_data.info['chs'][ch_idx]['loc'][:3]
                    electrode_locs[ch_name] = loc
        
        # If no locations found, try to get from montage
        if not electrode_locs and hasattr(raw_data.info, 'montage') and raw_data.info['montage'] is not None:
            montage = raw_data.info['montage']
            for ch_name in raw_data.ch_names:
                if ch_name in montage.ch_names:
                    pos_idx = montage.ch_names.index(ch_name)
                    electrode_locs[ch_name] = montage.dig[pos_idx]['r']
        
        return electrode_locs
    
    def sliding_window_mvpa_single_channel(self, channel_data, labels, random_seed):
        """
        Perform sliding window MVPA for a single channel
        
        Args:
            channel_data: Shape (n_trials, n_timepoints)
            labels: Shape (n_trials,)
            random_seed: Random seed for train/test split
        
        Returns:
            list: Accuracy scores for each time window
        """
        n_trials, n_timepoints = channel_data.shape
        
        # Calculate analysis window bounds in samples
        analysis_start_idx = max(0, self.analysis_start_samples - self.baseline_start_samples)
        analysis_end_idx = min(n_timepoints, self.analysis_end_samples - self.baseline_start_samples)
        
        # Calculate number of windows
        n_windows = (analysis_end_idx - analysis_start_idx - self.window_length_samples) // self.stride_samples + 1
        
        if n_windows <= 0:
            return []
        
        accuracies = []
        
        # Split data into train/test with fixed random seed
        try:
            train_idx, test_idx = train_test_split(
                range(n_trials), 
                test_size=self.config['test_size'], 
                random_state=random_seed,
                stratify=labels
            )
        except ValueError:
            # If stratification fails, try without it
            train_idx, test_idx = train_test_split(
                range(n_trials), 
                test_size=self.config['test_size'], 
                random_state=random_seed
            )
        
        # Slide window across time
        for window_idx in range(n_windows):
            start_sample = analysis_start_idx + window_idx * self.stride_samples
            end_sample = start_sample + self.window_length_samples
            
            # Extract features for current window
            window_data = channel_data[:, start_sample:end_sample]
            
            # Flatten time dimension (each trial becomes a feature vector)
            X = window_data.reshape(n_trials, -1)
            
            # Split into train/test
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Check if we have both classes in training set
            if len(np.unique(y_train)) < 2:
                accuracies.append(0.5)  # Chance level if only one class
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVM
            svm = SVC(kernel=self.config['svm_kernel'], C=self.config['svm_c'], random_state=random_seed)
            svm.fit(X_train_scaled, y_train)
            
            # Predict and calculate accuracy
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        return accuracies
    
    def analyze_channel(self, epochs_data, labels, channel_idx, channel_name):
        """
        Analyze a single channel with multiple random seeds
        
        Returns:
            dict: Results for this channel
        """
        # Extract data for this channel
        channel_data = epochs_data[:, channel_idx, :]  # Shape: (n_trials, n_timepoints)
        
        all_accuracies = []
        
        # Run analysis with multiple random seeds
        for rep in range(self.config['n_repetitions']):
            random_seed = rep  # Use repetition number as seed for reproducibility
            
            accuracies = self.sliding_window_mvpa_single_channel(
                channel_data, labels, random_seed
            )
            
            if accuracies:  # Only add if we got valid results
                all_accuracies.append(accuracies)
        
        if not all_accuracies:
            return None
        
        # Convert to numpy array and calculate statistics
        all_accuracies = np.array(all_accuracies)  # Shape: (n_repetitions, n_windows)
        
        mean_accuracy = np.mean(all_accuracies, axis=0)
        std_accuracy = np.std(all_accuracies, axis=0)
        
        # Calculate time vector for windows
        analysis_start_idx = max(0, self.analysis_start_samples - self.baseline_start_samples)
        window_times = []
        for window_idx in range(len(mean_accuracy)):
            start_sample = analysis_start_idx + window_idx * self.stride_samples
            center_sample = start_sample + self.window_length_samples // 2
            # Convert back to time relative to stimulus onset
            time_ms = (center_sample + self.baseline_start_samples) * 1000 / self.sfreq
            window_times.append(time_ms)
        
        window_times = np.array(window_times)
        
        results = {
            'channel_name': channel_name,
            'channel_idx': channel_idx,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'all_accuracies': all_accuracies,
            'window_times': window_times,
            'n_repetitions': self.config['n_repetitions'],
            'n_windows': len(mean_accuracy),
            'peak_accuracy': np.max(mean_accuracy),
            'peak_time': window_times[np.argmax(mean_accuracy)]
        }
        
        return results
    
    def sliding_window_mvpa_all_channels(self, epochs_data, labels, random_seed):
        """
        Perform sliding window MVPA using all channels together
        
        Args:
            epochs_data: Shape (n_trials, n_channels, n_timepoints)
            labels: Shape (n_trials,)
            random_seed: Random seed for train/test split
        
        Returns:
            list: Accuracy scores for each time window
        """
        n_trials, n_channels, n_timepoints = epochs_data.shape
        
        # Calculate analysis window bounds in samples
        analysis_start_idx = max(0, self.analysis_start_samples - self.baseline_start_samples)
        analysis_end_idx = min(n_timepoints, self.analysis_end_samples - self.baseline_start_samples)
        
        # Calculate number of windows
        n_windows = (analysis_end_idx - analysis_start_idx - self.window_length_samples) // self.stride_samples + 1
        
        if n_windows <= 0:
            return []
        
        accuracies = []
        
        # Split data into train/test with fixed random seed
        try:
            train_idx, test_idx = train_test_split(
                range(n_trials), 
                test_size=self.config['test_size'], 
                random_state=random_seed,
                stratify=labels
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                range(n_trials), 
                test_size=self.config['test_size'], 
                random_state=random_seed
            )
        
        # Slide window across time
        for window_idx in range(n_windows):
            start_sample = analysis_start_idx + window_idx * self.stride_samples
            end_sample = start_sample + self.window_length_samples
            
            # Extract features for current window (all channels)
            window_data = epochs_data[:, :, start_sample:end_sample]
            
            # Flatten channel and time dimensions (each trial becomes a feature vector)
            X = window_data.reshape(n_trials, -1)
            
            # Split into train/test
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Check if we have both classes in training set
            if len(np.unique(y_train)) < 2:
                accuracies.append(0.5)
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVM
            svm = SVC(kernel=self.config['svm_kernel'], C=self.config['svm_c'], random_state=random_seed)
            svm.fit(X_train_scaled, y_train)
            
            # Predict and calculate accuracy
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        return accuracies

    def analyze_all_channels(self, epochs_data, labels):
        """
        Analyze using all channels together with multiple random seeds
        
        Returns:
            dict: Results for all-channel analysis
        """
        print("Analyzing all channels together...")
        
        all_accuracies = []
        
        # Run analysis with multiple random seeds
        for rep in tqdm(range(self.config['n_repetitions']), desc="All channels"):
            random_seed = rep
            
            accuracies = self.sliding_window_mvpa_all_channels(
                epochs_data, labels, random_seed
            )
            
            if accuracies:
                all_accuracies.append(accuracies)
        
        if not all_accuracies:
            return None
        
        # Convert to numpy array and calculate statistics
        all_accuracies = np.array(all_accuracies)
        
        mean_accuracy = np.mean(all_accuracies, axis=0)
        std_accuracy = np.std(all_accuracies, axis=0)
        
        # Calculate time vector for windows
        analysis_start_idx = max(0, self.analysis_start_samples - self.baseline_start_samples)
        window_times = []
        for window_idx in range(len(mean_accuracy)):
            start_sample = analysis_start_idx + window_idx * self.stride_samples
            center_sample = start_sample + self.window_length_samples // 2
            time_ms = (center_sample + self.baseline_start_samples) * 1000 / self.sfreq
            window_times.append(time_ms)
        
        window_times = np.array(window_times)
        
        results = {
            'channel_name': 'ALL_CHANNELS',
            'channel_idx': -1,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'all_accuracies': all_accuracies,
            'window_times': window_times,
            'n_repetitions': self.config['n_repetitions'],
            'n_windows': len(mean_accuracy),
            'peak_accuracy': np.max(mean_accuracy),
            'peak_time': window_times[np.argmax(mean_accuracy)]
        }
        
        return results
    
    def save_all_results(self, all_results):
        """Save all results to a single file"""
        filepath = os.path.join(self.config['output_dir'], 'all_subjects_mvpa_results.pkl')
        
        with open(filepath, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Also save configuration
        config_filepath = os.path.join(self.config['output_dir'], 'analysis_config.pkl')
        with open(config_filepath, 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"Saved all results to: all_subjects_mvpa_results.pkl")
        print(f"Saved configuration to: analysis_config.pkl")
    
    
    def generate_summary_plots(self, all_results):
        """Generate summary plots across all subjects"""
        
        # Collect all channel results
        all_channel_results = []
        
        for subject_idx in all_results:
            subject_data = all_results[subject_idx]
            if 'channel_results' in subject_data:
                for channel_name, channel_result in subject_data['channel_results'].items():
                    channel_result['subject_idx'] = subject_idx
                    all_channel_results.append(channel_result)
        
        if not all_channel_results:
            print("No results to plot!")
            return
        
        # Plot 1: Average accuracy across all channels and subjects
        plt.figure(figsize=(15, 10))
        
        # Collect all time series
        all_times = []
        all_accuracies = []
        
        for result in all_channel_results:
            all_times.append(result['window_times'])
            all_accuracies.append(result['mean_accuracy'])
        
        # Find common time grid (use the first one as reference)
        if all_times:
            reference_times = all_times[0]
            
            # Plot individual channels (lightly)
            for i, (times, acc) in enumerate(zip(all_times, all_accuracies)):
                if len(times) == len(reference_times):  # Only plot if same length
                    plt.plot(times, acc, alpha=0.05, color='gray', linewidth=0.5)
            
            # Calculate and plot average
            valid_accuracies = [acc for times, acc in zip(all_times, all_accuracies) 
                              if len(times) == len(reference_times)]
            
            if valid_accuracies:
                mean_acc = np.mean(valid_accuracies, axis=0)
                std_acc = np.std(valid_accuracies, axis=0)
                
                plt.plot(reference_times, mean_acc, 'b-', linewidth=3, label='Mean accuracy')
                plt.fill_between(reference_times, mean_acc - std_acc, mean_acc + std_acc, 
                               alpha=0.3, color='blue', label='±1 SD')
        
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance level')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Stimulus onset')
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Classification Accuracy', fontsize=12)
        plt.title('MVPA Results: Average Across All Channels and Subjects (Session 1)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'summary_accuracy_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Summary plots saved!")
    
    def generate_subject_comparison_plots(self, all_results):
        """Generate plots comparing subjects"""
        
        # Plot peak accuracies by subject
        subjects = sorted(all_results.keys())
        subject_peak_accs = []
        subject_labels = []
        
        for subject_idx in subjects:
            subject_data = all_results[subject_idx]
            if 'channel_results' in subject_data:
                peak_accs = [result['peak_accuracy'] for result in subject_data['channel_results'].values()]
                subject_peak_accs.extend(peak_accs)
                subject_labels.extend([f'Subject {subject_idx}'] * len(peak_accs))
        
        if subject_peak_accs:
            plt.figure(figsize=(12, 8))
            
            # Box plot
            import pandas as pd
            df = pd.DataFrame({'Subject': subject_labels, 'Peak Accuracy': subject_peak_accs})
            sns.boxplot(data=df, x='Subject', y='Peak Accuracy')
            sns.stripplot(data=df, x='Subject', y='Peak Accuracy', color='red', alpha=0.5)
            
            plt.axhline(y=0.5, color='red', linestyle='--', label='Chance level')
            plt.title('Peak Classification Accuracy by Subject (All Channels)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output_dir'], 'subject_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def print_summary_statistics(self, all_results):
        """Print summary statistics"""
        print(f"\n{'='*50}")
        print("SUMMARY STATISTICS")
        print(f"{'='*50}")
        
        total_channels = 0
        all_peak_accs = []
        
        for subject_idx in sorted(all_results.keys()):
            subject_data = all_results[subject_idx]
            if 'channel_results' in subject_data:
                n_channels = len(subject_data['channel_results'])
                peak_accs = [result['peak_accuracy'] for result in subject_data['channel_results'].values()]
                
                total_channels += n_channels
                all_peak_accs.extend(peak_accs)
                
                print(f"Subject {subject_idx}:")
                print(f"  - Channels analyzed: {n_channels}")
                print(f"  - Mean peak accuracy: {np.mean(peak_accs):.3f} ± {np.std(peak_accs):.3f}")
                print(f"  - Max peak accuracy: {np.max(peak_accs):.3f}")
                print(f"  - Channels > 60% accuracy: {np.sum(np.array(peak_accs) > 0.6)}")
        
        print(f"\nOverall:")
        print(f"  - Total channels analyzed: {total_channels}")
        print(f"  - Mean peak accuracy: {np.mean(all_peak_accs):.3f} ± {np.std(all_peak_accs):.3f}")
        print(f"  - Max peak accuracy: {np.max(all_peak_accs):.3f}")
        print(f"  - Channels > 60% accuracy: {np.sum(np.array(all_peak_accs) > 0.6)} ({100*np.sum(np.array(all_peak_accs) > 0.6)/len(all_peak_accs):.1f}%)")
        print(f"  - Channels > 70% accuracy: {np.sum(np.array(all_peak_accs) > 0.7)} ({100*np.sum(np.array(all_peak_accs) > 0.7)/len(all_peak_accs):.1f}%)")


    def detect_significant_timepoints(self, all_accuracies, alpha=0.05):
        """Detect significantly above-chance timepoints using t-test"""
        n_windows = all_accuracies.shape[1]
        p_values = np.zeros(n_windows)
        
        for window_idx in range(n_windows):
            window_accuracies = all_accuracies[:, window_idx]
            # One-sample t-test against chance level (0.5)
            t_stat, p_val = stats.ttest_1samp(window_accuracies, 0.5)
            p_values[window_idx] = p_val if t_stat > 0 else 1.0  # Only test for above chance
        
        # Correct for multiple comparisons (Bonferroni)
        significant_mask = p_values < (alpha / n_windows)
        
        return significant_mask, p_values

    def save_subject_results(self, subject_results, subject_idx):
        """Save results for a single subject"""
        filename = f"subject_{subject_idx}_results.pkl"
        filepath = os.path.join(self.config['output_dir'], filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(subject_results, f)
        
        print(f"Saved subject results to {filepath}")

    def generate_subject_plots(self, subject_results, subject_idx):
        """Generate plots for a single subject"""
        print(f"Generating plots for Subject {subject_idx}...")
        
        # Create subject-specific output directory
        subject_dir = os.path.join(self.config['output_dir'], f'subject_{subject_idx}')
        os.makedirs(subject_dir, exist_ok=True)
        
        channel_results = subject_results['channel_results']
        all_channels_result = subject_results.get('all_channels_result')
        
        # Plot 1: Individual channel results
        n_channels = len(channel_results)
        fig, axes = plt.subplots(n_channels + 1, 1, figsize=(12, 4 * (n_channels + 1)))
        
        if n_channels == 1:
            axes = [axes]
        
        for idx, (channel_name, result) in enumerate(channel_results.items()):
            ax = axes[idx]
            
            times = result['window_times']
            mean_acc = result['mean_accuracy']
            std_acc = result['std_accuracy']
            all_accuracies = result['all_accuracies']
            
            # Detect significant timepoints
            sig_mask, p_values = self.detect_significant_timepoints(all_accuracies, self.config['alpha'])
            
            # Plot accuracy curve
            ax.plot(times, mean_acc, 'b-', linewidth=2, label='Mean accuracy')
            ax.fill_between(times, mean_acc - std_acc, mean_acc + std_acc, 
                        alpha=0.3, color='blue', label='±1 SD')
            
            # Mark significant timepoints
            if np.any(sig_mask):
                ax.scatter(times[sig_mask], mean_acc[sig_mask], 
                        color='red', s=50, zorder=5, label='Significant')
            
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Stimulus onset')
            
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Channel {channel_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot all-channels result instead of average
        ax = axes[-1]
        if all_channels_result is not None:
            times = all_channels_result['window_times']
            mean_acc = all_channels_result['mean_accuracy']
            std_acc = all_channels_result['std_accuracy']
            all_accuracies = all_channels_result['all_accuracies']
            
            # Detect significant timepoints
            sig_mask, p_values = self.detect_significant_timepoints(all_accuracies, self.config['alpha'])
            
            ax.plot(times, mean_acc, 'g-', linewidth=3, label='All channels together')
            ax.fill_between(times, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.3, color='green')
            
            # Mark significant timepoints
            if np.any(sig_mask):
                ax.scatter(times[sig_mask], mean_acc[sig_mask], 
                        color='red', s=50, zorder=5, label='Significant')
            
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Stimulus onset')
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Accuracy')
            ax.set_title('All Channels Together')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, 'channel_accuracies.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_brain_video(self, subject_results, subject_idx):
        """Generate brain video showing accuracy over time"""
        print(f"Generating brain video for Subject {subject_idx}...")
        
        subject_dir = os.path.join(self.config['output_dir'], f'subject_{subject_idx}')
        
        channel_results = subject_results['channel_results']
        electrode_locs = subject_results['electrode_locations']
        
        if not channel_results:
            print("No channel results to visualize")
            return
        
        # Get reference times (assume all channels have same time points)
        reference_result = list(channel_results.values())[0]
        times = reference_result['window_times']
        n_timepoints = len(times)
        
        # Prepare electrode locations and accuracies
        locs_list = []
        channel_names = []
        accuracy_matrix = []  # Shape: (n_channels, n_timepoints)
        
        for channel_name, result in channel_results.items():
            if channel_name in electrode_locs:
                locs_list.append(electrode_locs[channel_name])
                channel_names.append(channel_name)
                accuracy_matrix.append(result['mean_accuracy'])
        
        if not locs_list:
            print("No electrode locations found")
            return
        
        locs_array = np.array(locs_list)
        accuracy_matrix = np.array(accuracy_matrix)
        
        # Create frames for video
        frames = []
        
        for time_idx in range(n_timepoints):
            fig = plt.figure(figsize=(10, 8))
            
            current_accuracies = accuracy_matrix[:, time_idx]
            
            # Create color map
            cmap = plt.cm.RdYlBu_r
            norm = plt.Normalize(vmin=0.4, vmax=0.8)
            colors = [cmap(norm(acc)) for acc in current_accuracies]
            
            # Plot brain with electrodes
            try:
                # Convert locations to MNI coordinates if needed
                mni_locs = locs_array  # Assuming already in MNI space
                
                view = plotting.view_markers(
                    mni_locs,
                    marker_labels=[f'{name}\n{acc:.2f}' for name, acc in zip(channel_names, current_accuracies)],
                    marker_color=colors,
                    marker_size=8,
                    title=f'Subject {subject_idx} - Time: {times[time_idx]:.0f} ms'
                )
                
                # Save frame
                frame_path = os.path.join(subject_dir, f'frame_{time_idx:03d}.png')
                view.save_as_html(frame_path.replace('.png', '.html'))
                
                # Convert HTML to image (you might need to implement this)
                # For now, create a simple matplotlib plot
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(mni_locs[:, 0], mni_locs[:, 1], 
                                    c=current_accuracies, cmap=cmap, 
                                    vmin=0.4, vmax=0.8, s=100)
                
                for i, (name, acc) in enumerate(zip(channel_names, current_accuracies)):
                    plt.annotate(f'{name}\n{acc:.2f}', 
                            (mni_locs[i, 0], mni_locs[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, ha='left')
                
                plt.colorbar(scatter, label='Classification Accuracy')
                plt.title(f'Subject {subject_idx} - Time: {times[time_idx]:.0f} ms')
                plt.xlabel('X (mm)')
                plt.ylabel('Y (mm)')
                
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                frames.append(frame_path)
                
            except Exception as e:
                print(f"Error creating frame {time_idx}: {e}")
                continue
        
        # Create video from frames
        if frames:
            video_path = os.path.join(subject_dir, f'subject_{subject_idx}_brain_accuracy.mp4')
            self.create_video_from_frames(frames, video_path, fps=5)
            
            # Clean up frame files
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

    def create_video_from_frames(self, frame_paths, output_path, fps=5):
        """Create video from frame images"""
        if not frame_paths:
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            print("Could not read first frame")
            return
        
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to {output_path}")