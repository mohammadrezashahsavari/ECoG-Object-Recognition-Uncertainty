#!/usr/bin/env python3
"""
MVPA Analysis for ECoG Data
===========================

Main script for running a sliding window SVM classification analysis.
This script is designed to iterate through different levels of stimulus noise,
perform MVPA for each level, and save the results in separate folders.

Workflow:
1.  Define a base configuration for the analysis.
2.  Specify a list of noise ranges to investigate.
3.  Loop through each noise range:
    a. Create a unique output directory for the current analysis.
    b. Initialize the MVPA analyzer with the current settings.
    c. Loop through each subject:
        i.   Load and preprocess the ECoG data.
        ii.  Run the sliding window MVPA.
        iii. Save results and generate plots for the subject.
4.  Announce completion.
"""

import os
import numpy as np
from utils import load_subject_data, MVPAAnalyzer, generate_output_folder_name

def main():
    """
    Main function to configure and run the MVPA analysis pipeline.
    """
    # =============================================================================
    # 1. ANALYSIS CONFIGURATION
    # =============================================================================
    # All analysis parameters are defined in this dictionary.
    # Modify these values to change the behavior of the analysis.
    config = {
        # --- Data Parameters ---
        # Path to your .npz data file.
        'filepath': r'C:\Users\Mohammadreza\Desktop\Neuromatch CN\Data\faceshouses.npz',
        'original_sampling_rate': 1000,  # Original sampling rate of the data in Hz.
        'n_subjects': 7,                  # Number of subjects to analyze.
        'session_idx': 1,                 # Session index to use from the data file.

        # --- Preprocessing Parameters ---
        # These are passed to the `preprocess_ecog` function.
        # Set any parameter to `None` to deactivate that step.
        'preprocess_params': {
            'resample_rate': 600,         # Target sampling rate in Hz. Set to None to skip resampling.
            'l_freq': 0.5,               # High-pass cutoff in Hz.
            'h_freq': 290,               # Low-pass cutoff in Hz.
            'notch_freqs': None,          # Frequencies for notch filter, e.g., [60, 120].
            'rereference': None,          # Type of re-referencing, e.g., 'car' for common average.
            'exclude_channels': []        # List of channel names to exclude, e.g., ['ECOG_123'].
        },

        # --- Processing Mode ---
        # 'ecog': Use the preprocessed ECoG voltage signals directly.
        # 'high_gamma': Extract and use the high-gamma band power (broadband).
        'processing_mode': 'ecog',

        # --- MVPA Parameters ---
        'window_length_ms': 50,  # Duration of the sliding window in milliseconds.
        'stride_ms': 10,         # Step size of the sliding window in milliseconds.
        'n_repetitions': 10,      # Number of random train/test splits for robust accuracy estimates.
        'test_size': 0.3,        # Proportion of data to use for the test set (e.g., 0.3 = 30%).
        'svm_kernel': 'rbf',     # Kernel for the Support Vector Machine classifier.
        'svm_c': 1.0,            # Regularization parameter for the SVM.

        # --- Analysis Time Window ---
        # Times are relative to stimulus onset (0 ms).
        'baseline_start_ms': -200, # Start of the window for baseline correction
        'baseline_end_ms': 0, # End of the window for baseline correction
        
        'analysis_start_ms': -200,    # Start of the epoch window.
        'analysis_end_ms': 500,    # End of the epoch window.
        
        # --- Statistical and Output Parameters ---
        'alpha': 0.05,                   # Significance level for statistical tests.
        'generate_brain_videos': False,  # Set to True to generate brain animations (can be slow).
    }

    # =============================================================================
    # 2. DEFINE NOISE LEVELS FOR ANALYSIS
    # =============================================================================
    # The script will loop through this list, running a full analysis for each entry.
    # Each entry is a [min, max] pair defining the noise range (0.0 to 1.0).
    noise_ranges_to_analyze = [
        [0.0, 0.2],
        [0.2, 0.4],
        [0.4, 0.6],
        [0.6, 0.8],
        [0.8, 1.0]
    ]

    # =============================================================================
    # 3. RUN THE ANALYSIS PIPELINE
    # =============================================================================
    print("Starting the MVPA analysis pipeline...")

    for noise_range in noise_ranges_to_analyze:
        # --- Setup for the current noise level ---
        print(f"\n{'='*80}")
        print(f"STARTING ANALYSIS FOR NOISE RANGE: {noise_range}")
        print(f"{'='*80}\n")

        # Create a copy of the config for this run to avoid overwriting.
        current_config = config.copy()
        current_config['noise_range'] = noise_range
        
        # The sampling rate for the analyzer is the resampled rate if it exists.
        if current_config['preprocess_params']['resample_rate'] is not None:
            current_config['sampling_rate'] = current_config['preprocess_params']['resample_rate']
        else:
            current_config['sampling_rate'] = current_config['original_sampling_rate']

        # Generate and create a unique output directory for this analysis run.
        current_config['output_dir'] = generate_output_folder_name(current_config)
        os.makedirs(current_config['output_dir'], exist_ok=True)

        print("Current Configuration:")
        print(f"  - Processing mode: {current_config['processing_mode']}")
        print(f"  - Sampling rate: {current_config['sampling_rate']} Hz")
        print(f"  - Noise range: {current_config['noise_range']}")
        print(f"  - Output directory: {current_config['output_dir']}\n")

        # Initialize the MVPA analyzer with the configuration for this run.
        analyzer = MVPAAnalyzer(current_config)

        # --- Process each subject for the current noise level ---
        for subject_idx in range(current_config['n_subjects']):
            print(f"\n--- Processing Subject {subject_idx} ---")

            # Check if results already exist for this subject under this configuration.
            result_file = os.path.join(current_config['output_dir'], f'subject_{subject_idx}_results.pkl')
            if os.path.exists(result_file):
                print(f"Results already exist for Subject {subject_idx}. Skipping.")
                continue

            # Load and preprocess data for this subject.
            print("Loading and preprocessing data...")
            raw_data = load_subject_data(current_config, subject_idx)

            if raw_data is None:
                print(f"Failed to load or process data for Subject {subject_idx}. Skipping.")
                continue

            # Run the MVPA analysis.
            subject_results = analyzer.analyze_subject(raw_data, subject_idx)

            if subject_results:
                # Save results and generate plots.
                analyzer.save_subject_results(subject_results, subject_idx)
                analyzer.generate_subject_plots(subject_results, subject_idx)
                if current_config['generate_brain_videos']:
                    analyzer.generate_brain_video(subject_results, subject_idx)
            else:
                print(f"Analysis for Subject {subject_idx} did not yield any results.")

    print(f"\n{'='*80}")
    print("MVPA ANALYSIS PIPELINE COMPLETE!")
    print("All specified noise levels have been analyzed.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
