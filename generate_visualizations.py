import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
from pathlib import Path
from scipy import ndimage
from scipy import stats
import pandas as pd
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy.interpolate import griddata
import warnings
try:
    from nilearn import plotting
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available for brain visualization")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings('ignore')

class NoiseRangeVisualizer:
    def __init__(self, base_results_dir, config):
        """
        Initialize the noise range visualizer
        
        Args:
            base_results_dir: Base directory containing all noise range results
            config: Configuration dictionary with visualization parameters
        """
        self.base_results_dir = Path(base_results_dir)
        self.config = config
        
        # Find all noise range directories
        self.noise_dirs = self._find_noise_directories()
        print(f"Found {len(self.noise_dirs)} noise range directories:")
        for noise_range, path in self.noise_dirs.items():
            print(f"  {noise_range}: {path.name}")
        
        # Load all results
        self.all_results = self._load_all_results()
        
        # Create output directory
        self.output_dir = self.base_results_dir / "NoiseRange_Visualizations"
        self.output_dir.mkdir(exist_ok=True)
    
    def _find_noise_directories(self):
        """Find all directories with noise range patterns"""
        noise_dirs = {}
        
        # Pattern: noise0.00to0.20, noise0.20to0.40, etc.
        pattern = "*noise*to*"
        for dir_path in self.base_results_dir.glob(pattern):
            if dir_path.is_dir():
                # Extract noise range from directory name
                dir_name = dir_path.name
                noise_part = [part for part in dir_name.split(' - ') if 'noise' in part][0]
                # Parse: noise0.00to0.20 -> (0.0, 0.2)
                range_str = noise_part.replace('noise', '')
                min_val, max_val = range_str.split('to')
                noise_range = (float(min_val), float(max_val))
                noise_dirs[noise_range] = dir_path
        
        return dict(sorted(noise_dirs.items()))
    
    def _load_all_results(self):
        """Load results from all noise range directories"""
        all_results = {}
        
        for noise_range, dir_path in self.noise_dirs.items():
            print(f"\nLoading results for noise range {noise_range}...")
            
            # Find results files
            results_files = list(dir_path.glob("subject_*_results.pkl"))
            
            noise_results = {}
            for results_file in results_files:
                # Extract subject index from filename
                filename = results_file.stem
                try:
                    subject_idx = int(filename.split('_')[1])  # subject_X_results.pkl
                except (IndexError, ValueError):
                    print(f"  Warning: Could not extract subject index from {filename}")
                    continue
                
                # Load results
                try:
                    with open(results_file, 'rb') as f:
                        subject_data = pickle.load(f)
                    noise_results[subject_idx] = subject_data
                    print(f"  Loaded subject {subject_idx}")
                except Exception as e:
                    print(f"  Error loading {results_file}: {e}")
            
            all_results[noise_range] = noise_results
        
        return all_results
    
    def _apply_smoothing_1d(self, accuracy_data, sigma=1.0):
        """Apply 1D Gaussian smoothing along time axis only"""
        if not self.config.get('apply_smoothing', True):
            return accuracy_data
        
        # Ensure we're working with numpy array
        data = np.array(accuracy_data)
        
        # Apply 1D smoothing only along the last axis (time)
        if data.ndim == 1:
            return ndimage.gaussian_filter1d(data, sigma=sigma)
        else:
            # For 2D data (channels x time), smooth along axis 1 (time)
            return ndimage.gaussian_filter1d(data, sigma=sigma, axis=-1)
    
    def _detect_significant_timepoints(self, accuracies, threshold=0.55, min_duration=3):
        """
        Detect significant time points where accuracy is consistently above threshold
        
        Args:
            accuracies: Array of accuracy values over time
            threshold: Minimum accuracy threshold for significance
            min_duration: Minimum number of consecutive time points above threshold
        
        Returns:
            significant_mask: Boolean array indicating significant time points
        """
        above_threshold = accuracies > threshold
        
        # Find consecutive runs above threshold
        significant_mask = np.zeros_like(above_threshold, dtype=bool)
        
        if np.any(above_threshold):
            # Find start and end of runs
            diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Keep only runs that meet minimum duration
            for start, end in zip(starts, ends):
                if end - start >= min_duration:
                    significant_mask[start:end] = True
        
        return significant_mask
    
    def _extract_electrode_locations(self, subject_data):
        """Extract electrode locations from subject data with robust error handling"""
        try:
            # Method 1: Check for 'electrode_locations' key
            if 'electrode_locations' in subject_data:
                electrode_locs = subject_data['electrode_locations']
                print(f" Found electrode_locations key, type: {type(electrode_locs)}")
                
                # Handle different possible formats
                if isinstance(electrode_locs, dict):
                    print(f" Electrode locations keys: {list(electrode_locs.keys())}")
                    
                    # Try to extract coordinates from dictionary
                    coords = []
                    channel_names = []
                    
                    for channel_name, loc_data in electrode_locs.items():
                        if isinstance(loc_data, dict):
                            # Look for coordinate keys
                            if all(key in loc_data for key in ['x', 'y', 'z']):
                                coords.append([loc_data['x'], loc_data['y'], loc_data['z']])
                                channel_names.append(channel_name)
                            elif all(key in loc_data for key in ['X', 'Y', 'Z']):
                                coords.append([loc_data['X'], loc_data['Y'], loc_data['Z']])
                                channel_names.append(channel_name)
                        elif isinstance(loc_data, (list, tuple, np.ndarray)) and len(loc_data) >= 3:
                            coords.append(loc_data[:3])
                            channel_names.append(channel_name)
                    
                    if coords:
                        return np.array(coords), channel_names
                    else:
                        print(" Could not extract coordinates from electrode_locations dict")
                
                elif isinstance(electrode_locs, (list, tuple, np.ndarray)):
                    # If it's already an array-like structure
                    electrode_locs = np.array(electrode_locs)
                    print(f" Electrode locations shape: {electrode_locs.shape}")
                    
                    if electrode_locs.ndim == 2 and electrode_locs.shape[1] >= 3:
                        # Generate channel names
                        n_channels = electrode_locs.shape[0]
                        channel_names = [f"CH_{i+1}" for i in range(n_channels)]
                        return electrode_locs[:, :3], channel_names
                    else:
                        print(" Electrode locations array has unexpected shape")
            
            # Method 2: Check for MNE raw data structure
            if 'raw' in subject_data:
                raw = subject_data['raw']
                print(f" Found raw data, type: {type(raw)}")
                
                # Check if it's an MNE Raw object
                if hasattr(raw, 'info') and hasattr(raw.info, 'chs'):
                    print(f" Found MNE raw.info.chs with {len(raw.info.chs)} channels")
                    
                    coords = []
                    channel_names = []
                    
                    for ch_idx, ch_info in enumerate(raw.info.chs):
                        if 'loc' in ch_info and ch_info['loc'] is not None:
                            loc = ch_info['loc']
                            if len(loc) >= 3:
                                # MNE stores locations in the first 3 elements
                                coords.append(loc[:3])
                                channel_names.append(ch_info.get('ch_name', f'CH_{ch_idx+1}'))
                    
                    if coords:
                        coords_array = np.array(coords)
                        print(f" Extracted {len(coords)} electrode locations from MNE raw")
                        return coords_array, channel_names
                    else:
                        print(" No valid electrode locations found in MNE raw.info.chs")
            
            # Method 3: Check for 'raw_data' key (alternative naming)
            if 'raw_data' in subject_data:
                raw_data = subject_data['raw_data']
                print(f" Found raw_data, type: {type(raw_data)}")
                
                if hasattr(raw_data, 'info') and hasattr(raw_data.info, 'chs'):
                    print(f" Found MNE raw_data.info.chs with {len(raw_data.info.chs)} channels")
                    
                    coords = []
                    channel_names = []
                    
                    for ch_idx, ch_info in enumerate(raw_data.info.chs):
                        if 'loc' in ch_info and ch_info['loc'] is not None:
                            loc = ch_info['loc']
                            if len(loc) >= 3:
                                coords.append(loc[:3])
                                channel_names.append(ch_info.get('ch_name', f'CH_{ch_idx+1}'))
                    
                    if coords:
                        coords_array = np.array(coords)
                        print(f" Extracted {len(coords)} electrode locations from MNE raw_data")
                        return coords_array, channel_names
            
            # Method 4: Look for any key that might contain electrode info
            print(" Searching for electrode location keys in subject_data...")
            for key in subject_data.keys():
                if 'electrode' in key.lower() or 'location' in key.lower() or 'coord' in key.lower():
                    print(f" Found potential electrode key: {key}")
                    data = subject_data[key]
                    
                    if isinstance(data, (list, tuple, np.ndarray)):
                        data = np.array(data)
                        if data.ndim == 2 and data.shape[1] >= 3:
                            n_channels = data.shape[0]
                            channel_names = [f"CH_{i+1}" for i in range(n_channels)]
                            return data[:, :3], channel_names
            
            print(" No electrode_locations found in subject data")
            return None, None
            
        except Exception as e:
            print(f" Error extracting electrode locations: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def generate_level1_visualizations(self):
        """Level 1: Per noise level and per subject"""
        print("\n" + "="*50)
        print("GENERATING LEVEL 1 VISUALIZATIONS")
        print("Per noise level and per subject")
        print("="*50)
        
        level1_dir = self.output_dir / "level1_per_subject_per_noise"
        level1_dir.mkdir(exist_ok=True)
        
        for noise_range, noise_data in self.all_results.items():
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            noise_dir = level1_dir / f"noise_{noise_str}"
            noise_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing noise range {noise_str}...")
            
            for subject_idx, subject_data in noise_data.items():
                print(f"  Subject {subject_idx}...")
                
                # Generate existing plots
                self._generate_subject_channel_heatmap(
                    subject_data, subject_idx, noise_range, noise_dir
                )
                
                self._generate_subject_all_channels_timecourse(
                    subject_data, subject_idx, noise_range, noise_dir
                )
                
                # Generate new plots
                self._generate_subject_channel_grid(
                    subject_data, subject_idx, noise_range, noise_dir
                )
                
                self._generate_brain_visualization(
                    subject_data, subject_idx, noise_range, noise_dir
                )
    
    def _generate_subject_channel_heatmap(self, subject_data, subject_idx, noise_range, output_dir):
        """Generate heatmap for all channels of a single subject"""
        if 'channel_results' not in subject_data:
            print(f"    No channel_results for subject {subject_idx}")
            return
        
        channel_results = subject_data['channel_results']
        if not channel_results:
            print(f"    Empty channel_results for subject {subject_idx}")
            return
        
        # Extract channel data
        channel_names = list(channel_results.keys())
        n_channels = len(channel_names)
        
        # Get time points from first channel
        first_channel = list(channel_results.values())[0]
        if 'window_times' not in first_channel:
            print(f"    No window_times in first channel for subject {subject_idx}")
            return
        
        time_points = first_channel['window_times']
        n_times = len(time_points)
        
        # Create accuracy matrix
        accuracy_matrix = np.zeros((n_channels, n_times))
        
        for i, channel_name in enumerate(channel_names):
            channel_data = channel_results[channel_name]
            if 'mean_accuracy' in channel_data:
                accuracies = np.array(channel_data['mean_accuracy'])
                
                # Apply 1D smoothing along time axis only
                accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
                accuracy_matrix[i, :] = accuracies
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, max(8, n_channels * 0.3)))
        
        # Create heatmap
        im = ax.imshow(accuracy_matrix, aspect='auto', cmap='RdYlBu_r', 
                      vmin=0.4, vmax=0.8, interpolation='bilinear')
        
        # Set labels
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        
        # Set time ticks
        time_ticks = np.arange(0, n_times, max(1, n_times // 10))
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f"{time_points[i]:.0f}" for i in time_ticks])
        
        # Set channel ticks (show every 5th channel if too many)
        if n_channels > 20:
            channel_ticks = np.arange(0, n_channels, 5)
        else:
            channel_ticks = np.arange(n_channels)
        ax.set_yticks(channel_ticks)
        ax.set_yticklabels([channel_names[i] for i in channel_ticks], fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Decoding Accuracy', fontsize=12)
        
        # Add title
        noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
        ax.set_title(f'Subject {subject_idx} - Individual Channel Decoding\n'
                    f'Noise Range: {noise_str}', fontsize=14, fontweight='bold')
        
        # Add stimulus onset line
        stimulus_onset_idx = np.argmin(np.abs(time_points))
        ax.axvline(x=stimulus_onset_idx, color='white', linestyle='--', alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'subject_{subject_idx}_channels_heatmap.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Generated channel heatmap: {filename}")
    
    def _generate_subject_channel_grid(self, subject_data, subject_idx, noise_range, output_dir):
        """Generate 6x10 grid of individual channel accuracies with significant timepoints"""
        if 'channel_results' not in subject_data:
            return
        
        channel_results = subject_data['channel_results']
        if not channel_results:
            return
        
        # Create figure with 6x10 subplots
        fig, axes = plt.subplots(6, 10, figsize=(20, 12))
        axes = axes.flatten()
        
        channel_names = list(channel_results.keys())
        
        for i in range(60):  # Maximum 60 channels
            ax = axes[i]
            
            if i < len(channel_names):
                channel_name = channel_names[i]
                channel_data = channel_results[channel_name]
                
                if 'window_times' in channel_data and 'mean_accuracy' in channel_data:
                    time_points = np.array(channel_data['window_times'])
                    accuracies = np.array(channel_data['mean_accuracy'])
                    
                    # Apply 1D smoothing
                    accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
                    
                    # Detect significant timepoints
                    significant_mask = self._detect_significant_timepoints(accuracies)
                    
                    # Plot accuracy
                    ax.plot(time_points, accuracies, 'b-', linewidth=1, alpha=0.8)
                    
                    # Highlight significant timepoints
                    if np.any(significant_mask):
                        ax.fill_between(time_points, 0.4, 0.8, 
                                      where=significant_mask, alpha=0.3, color='green',
                                      label='Significant')
                    
                    # Add chance level
                    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
                    
                    # Add stimulus onset
                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
                    
                    # Set title
                    ax.set_title(f'{channel_name}', fontsize=8)
                    
                    # Set limits
                    ax.set_ylim(0.4, 0.8)
                    ax.set_xlim(time_points[0], time_points[-1])
                    
                    # Remove tick labels for cleaner look
                    if i < 50:  # Not bottom row
                        ax.set_xticklabels([])
                    if i % 10 != 0:  # Not leftmost column
                        ax.set_yticklabels([])
                    
                    # Add grid
                    ax.grid(True, alpha=0.2)
                    
                else:
                    ax.set_title(f'{channel_name}\n(No Data)', fontsize=8)
                    ax.axis('off')
            else:
                # Empty subplot
                ax.axis('off')
        
        # Add common labels
        fig.text(0.5, 0.02, 'Time (ms)', ha='center', fontsize=12)
        fig.text(0.02, 0.5, 'Decoding Accuracy', va='center', rotation='vertical', fontsize=12)
        
        noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
        fig.suptitle(f'Subject {subject_idx} - Individual Channel Accuracies\n'
                    f'Noise Range: {noise_str}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.98)
        
        # Save figure
        filename = f'subject_{subject_idx}_channel_grid.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Generated channel grid: {filename}")
    
    def _generate_brain_visualization(self, subject_data, subject_idx, noise_range, output_dir):
        """Generate interactive brain visualization with electrode locations on cortex"""
        if not NILEARN_AVAILABLE:
            print(f"   Nilearn not available, skipping brain visualization for subject {subject_idx}")
            return
        
        if 'channel_results' not in subject_data:
            print(f"   No channel_results for subject {subject_idx}")
            return
        
        channel_results = subject_data['channel_results']
        if not channel_results:
            print(f"   Empty channel_results for subject {subject_idx}")
            return
        
        # Extract electrode locations
        print(f"   Extracting electrode locations for subject {subject_idx}...")
        electrode_locs, electrode_names = self._extract_electrode_locations(subject_data)
        
        # Get channel data
        channel_names = list(channel_results.keys())
        first_channel = list(channel_results.values())[0]
        
        if 'window_times' not in first_channel:
            print(f"   No window_times in first channel for subject {subject_idx}")
            return
        
        time_points = np.array(first_channel['window_times'])
        n_times = len(time_points)
        n_channels = len(channel_names)
        
        # If we couldn't extract real electrode locations, generate dummy ones
        if electrode_locs is None:
            print(f"   Generating dummy electrode locations for {n_channels} channels")
            electrode_locs, electrode_names = self._generate_dummy_electrode_locations(n_channels)
        
        # Ensure electrode_locs is a numpy array
        if not isinstance(electrode_locs, np.ndarray):
            electrode_locs = np.array(electrode_locs)
        
        # Check if coordinates are reasonable (not all zeros)
        if np.all(electrode_locs == 0):
            print(f"   Warning: All electrode locations are zero, generating spread locations")
            electrode_locs, electrode_names = self._generate_dummy_electrode_locations(n_channels)
        
        # Convert coordinates to MNI space if needed
        try:
            # Assume coordinates are already in MNI space or convert if needed
            mni_coords = electrode_locs[:, :3]
            
            # Ensure coordinates are in reasonable MNI range (-100 to 100)
            if np.max(np.abs(mni_coords)) > 200:
                print(f"   Warning: Coordinates seem too large, scaling down")
                mni_coords = mni_coords / 10  # Scale down if too large
            
            # If coordinates are too small, scale up
            if np.max(np.abs(mni_coords)) < 10:
                print(f"   Warning: Coordinates seem too small, scaling up")
                mni_coords = mni_coords * 10  # Scale up if too small
                
        except Exception as e:
            print(f"   Error processing coordinates: {e}")
            mni_coords = electrode_locs[:, :3] # Use as-is
        
        # Ensure we don't have more channels than electrode locations
        n_electrodes = min(n_channels, len(electrode_locs))
        
        # Create accuracy matrix
        accuracy_matrix = np.zeros((n_electrodes, n_times))
        
        for i in range(n_electrodes):
            if i < len(channel_names):
                channel_name = channel_names[i]
                channel_data = channel_results[channel_name]
                if 'mean_accuracy' in channel_data:
                    accuracies = np.array(channel_data['mean_accuracy'])
                    accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
                    accuracy_matrix[i, :] = accuracies
        
        try:
            # Create interactive brain visualization with nilearn
            html_content = self._create_brain_html_with_dual_sliders(
                mni_coords[:n_electrodes], 
                accuracy_matrix, 
                channel_names[:n_electrodes],
                time_points, 
                subject_idx,
                noise_range
            )
            
            if html_content is not None:
                # Save as HTML
                noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
                filename = f'subject_{subject_idx}_brain_visualization.html'
                
                with open(output_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"   Generated brain visualization with cortex: {filename}")
            else:
                raise Exception("Failed to generate HTML content")
        
        except Exception as e:
            print(f"   Error generating brain visualization: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Falling back to simple 3D plot for subject {subject_idx}")
            # Fallback to the previous method
            self._generate_simple_3d_visualization(electrode_locs, accuracy_matrix, channel_names, 
                                                time_points, subject_idx, noise_range, output_dir)
            
    
    def _generate_simple_3d_visualization(self, electrode_locs, accuracy_matrix, channel_names, 
                                        time_points, subject_idx, noise_range, output_dir):
        """Generate simple 3D visualization as fallback"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create figure with multiple time points
            n_timepoints = 6
            time_indices = np.linspace(0, len(time_points)-1, n_timepoints, dtype=int)
            
            fig = plt.figure(figsize=(20, 12))
            
            for i, t_idx in enumerate(time_indices):
                ax = fig.add_subplot(2, 3, i+1, projection='3d')
                
                # Get accuracies at this time point
                current_accuracies = accuracy_matrix[:, t_idx]
                
                # Create scatter plot
                scatter = ax.scatter(
                    electrode_locs[:len(channel_names), 0],
                    electrode_locs[:len(channel_names), 1],
                    electrode_locs[:len(channel_names), 2],
                    c=current_accuracies,
                    cmap='RdYlBu_r',
                    vmin=0.4,
                    vmax=0.8,
                    s=60
                )
                
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title(f'Time: {time_points[t_idx]:.0f}ms')
                
                # Add colorbar for first subplot
                if i == 0:
                    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Accuracy')
            
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            fig.suptitle(f'Subject {subject_idx} - Brain Electrode Locations\n'
                        f'Noise Range: {noise_str}', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            filename = f'subject_{subject_idx}_brain_simple_3d.png'
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f" Generated simple 3D brain visualization: {filename}")
            
        except Exception as e:
            print(f" Could not generate simple 3D brain visualization: {e}")

    def _create_brain_html_with_dual_sliders(self, electrode_locs, accuracy_matrix, 
                                        channel_names, time_points, subject_idx, noise_range):
        """Create optimized HTML with brain visualization and dual sliders"""
        
        try:
            from nilearn import plotting
            import matplotlib.cm as cm
            
            # OPTIMIZATION 1: Reduce number of time points significantly
            n_times = len(time_points)
            max_time_points = 10 # Reduced from 30 to 10
            time_step = max(1, n_times // max_time_points)
            sampled_indices = list(range(0, n_times, time_step))
            
            print(f"   Generating {len(sampled_indices)} time points for HTML (reduced from {n_times})")
            
            # OPTIMIZATION 2: Reduce number of electrodes if too many
            max_electrodes = 40 # Limit electrodes for performance
            n_electrodes = min(len(electrode_locs), max_electrodes)
            if len(electrode_locs) > max_electrodes:
                print(f"   Reducing electrodes from {len(electrode_locs)} to {max_electrodes} for performance")
            
            mni_coords = electrode_locs[:n_electrodes]
            reduced_accuracy_matrix = accuracy_matrix[:n_electrodes, :]
            reduced_channel_names = channel_names[:n_electrodes]
            
            # Generate brain views for each sampled time point
            brain_views_data = []
            
            for i, t_idx in enumerate(sampled_indices):
                current_accuracies = reduced_accuracy_matrix[:, t_idx]
                
                # Create marker colors based on accuracy
                normalized_acc = (current_accuracies - 0.4) / (0.8 - 0.4)
                normalized_acc = np.clip(normalized_acc, 0, 1)
                
                # Convert to hex colors using RdYlBu_r colormap
                colormap = cm.get_cmap('RdYlBu_r')
                colors = [colormap(acc) for acc in normalized_acc]
                hex_colors = ['#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
                            for c in colors]
                
                # Create simplified marker labels (shorter text)
                marker_labels = [f'{reduced_channel_names[j][:8]}: {current_accuracies[j]:.2f}' 
                            for j in range(len(mni_coords))]
                
                # Create view with proper parameters
                view = plotting.view_markers(
                    mni_coords,
                    marker_labels=marker_labels,
                    marker_color=hex_colors,
                    marker_size=8
                )
                
                # Get HTML and extract the content properly
                view_html = view._repr_html_()
                
                brain_views_data.append({
                    'html': view_html,
                    'time': time_points[t_idx],
                    'accuracies': current_accuracies
                })
                
                print(f"   Generated brain view {i+1}/{len(sampled_indices)}")
            
            # Create the optimized HTML
            html_content = self._generate_optimized_html(brain_views_data, subject_idx, noise_range)
            
            return html_content
            
        except Exception as e:
            print(f"Error creating brain HTML: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _generate_optimized_html(self, brain_views_data, subject_idx, noise_range):
        """Generate optimized HTML with minimal size and better performance"""
        
        # Create time points array for JavaScript
        time_points_js = [round(view_data['time'], 1) for view_data in brain_views_data]
        
        # Start building HTML
        html_parts = []
        
        # HTML header and styles
        html_parts.append(f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Brain Visualization - Subject {subject_idx}</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 10px;
                background: #f5f5f5;
                padding: 0;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .controls {{
                margin: 15px 0;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .slider-container {{
                margin: 10px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .slider-label {{
                min-width: 100px;
                font-weight: bold;
                font-size: 14px;
            }}
            .slider {{
                flex: 1;
                height: 6px;
                background: #ddd;
                outline: none;
                border-radius: 3px;
            }}
            .slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                background: #4CAF50;
                cursor: pointer;
                border-radius: 50%;
            }}
            .value-display {{
                min-width: 80px;
                font-weight: bold;
                color: #333;
                font-size: 14px;
            }}
            #brain-container {{
                margin: 15px 0;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 15px;
                min-height: 600px;
            }}
            .brain-view {{
                width: 100%;
                height: auto;
                min-height: 500px;
            }}
            .info {{
                background: #e8f4fd;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 3px solid #2196F3;
                font-size: 13px;
            }}
            .accuracy-info {{
                background: #f0f8ff;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Subject {subject_idx} - Brain Electrode Locations</h2>
            <h3>Noise Range: {noise_range[0]:.1f} - {noise_range[1]:.1f}</h3>
        </div>
        
        <div class="info">
            <b>Instructions:</b> Use the time slider to navigate through different time points. 
            Use the opacity slider to adjust brain transparency. 
            Hover over electrodes to see accuracy values.
        </div>
        
        <div class="controls">
            <div class="slider-container">
                <label class="slider-label" for="time-slider">Time Point:</label>
                <input type="range" id="time-slider" class="slider" min="0" max="{len(brain_views_data)-1}" value="0" step="1">
                <span class="value-display" id="time-value">{time_points_js[0]:.0f} ms</span>
            </div>
            
            <div class="slider-container">
                <label class="slider-label" for="opacity-slider">Brain Opacity:</label>
                <input type="range" id="opacity-slider" class="slider" min="10" max="100" value="70" step="5">
                <span class="value-display" id="opacity-value">70%</span>
            </div>
        </div>
        
        <div class="accuracy-info" id="accuracy-info">
            Current time: {time_points_js[0]:.0f} ms | 
            Accuracy range: {brain_views_data[0]['accuracies'].min():.3f} - {brain_views_data[0]['accuracies'].max():.3f}
        </div>
        
        <div id="brain-container">""")
        
        # Add each brain view
        for i, view_data in enumerate(brain_views_data):
            display_style = "block" if i == 0 else "none"
            html_parts.append(f"""
            <div id="brain-view-{i}" class="brain-view" style="display: {display_style};">
                {view_data['html']}
            </div>""")
        
        # Close brain container and add JavaScript
        html_parts.append("""
        </div>
        
        <script>
            // Get DOM elements
            const timeSlider = document.getElementById('time-slider');
            const opacitySlider = document.getElementById('opacity-slider');
            const timeValue = document.getElementById('time-value');
            const opacityValue = document.getElementById('opacity-value');
            const accuracyInfo = document.getElementById('accuracy-info');
            
            // Time points and accuracy data
            const timePoints = """ + str(time_points_js) + """;
            const numViews = """ + str(len(brain_views_data)) + """;
            const accuracyData = """ + str([[float(acc) for acc in view_data['accuracies']] for view_data in brain_views_data]) + """;
            
            // Function to update accuracy info
            function updateAccuracyInfo(timeIndex) {
                const accs = accuracyData[timeIndex];
                const minAcc = Math.min(...accs);
                const maxAcc = Math.max(...accs);
                const avgAcc = accs.reduce((a, b) => a + b, 0) / accs.length;
                
                accuracyInfo.innerHTML = `
                    Current time: ${timePoints[timeIndex].toFixed(0)} ms | 
                    Accuracy range: ${minAcc.toFixed(3)} - ${maxAcc.toFixed(3)} | 
                    Average: ${avgAcc.toFixed(3)}
                `;
            }
            
            // Time slider event listener
            timeSlider.addEventListener('input', function() {
                const timeIndex = parseInt(this.value);
                timeValue.textContent = timePoints[timeIndex].toFixed(0) + ' ms';
                
                // Hide all brain views
                for (let i = 0; i < numViews; i++) {
                    const view = document.getElementById('brain-view-' + i);
                    if (view) {
                        view.style.display = 'none';
                    }
                }
                
                // Show selected brain view
                const selectedView = document.getElementById('brain-view-' + timeIndex);
                if (selectedView) {
                    selectedView.style.display = 'block';
                }
                
                // Update accuracy info
                updateAccuracyInfo(timeIndex);
            });
            
            // Opacity slider event listener
            opacitySlider.addEventListener('input', function() {
                const opacity = parseInt(this.value);
                opacityValue.textContent = opacity + '%';
                
                // Apply opacity to all brain views
                const opacityValue = opacity / 100;
                
                for (let i = 0; i < numViews; i++) {
                    const view = document.getElementById('brain-view-' + i);
                    if (view) {
                        // Find all interactive elements
                        const canvases = view.querySelectorAll('canvas');
                        const svgs = view.querySelectorAll('svg');
                        const webglCanvases = view.querySelectorAll('canvas[data-engine="webgl"]');
                        
                        // Apply opacity to canvas elements
                        canvases.forEach(canvas => {
                            canvas.style.opacity = opacityValue;
                        });
                        
                        // Apply opacity to SVG elements
                        svgs.forEach(svg => {
                            svg.style.opacity = opacityValue;
                        });
                        
                        // Apply opacity to WebGL canvases
                        webglCanvases.forEach(canvas => {
                            canvas.style.opacity = opacityValue;
                        });
                        
                        // Apply to the container itself as fallback
                        const plotlyPlots = view.querySelectorAll('.plotly-graph-div');
                        plotlyPlots.forEach(plot => {
                            plot.style.opacity = opacityValue;
                        });
                    }
                }
            });
            
            // Initialize on page load
            window.addEventListener('load', function() {
                // Wait a bit for the brain views to fully load
                setTimeout(function() {
                    // Trigger initial opacity setting
                    opacitySlider.dispatchEvent(new Event('input'));
                    
                    // Update initial accuracy info
                    updateAccuracyInfo(0);
                    
                    console.log('Brain visualization loaded successfully');
                }, 1000);
            });
            
            // Add error handling
            window.addEventListener('error', function(e) {
                console.error('Error in brain visualization:', e);
            });
        </script>
    </body>
    </html>""")
        
        return ''.join(html_parts)
        

    def _generate_dummy_electrode_locations(self, n_channels):
        """Generate dummy electrode locations for visualization"""
        # Create a grid-like layout for visualization
        grid_size = int(np.ceil(np.sqrt(n_channels)))
        locations = []
        names = []
        
        for i in range(n_channels):
            x = (i % grid_size) * 15 - (grid_size * 15) / 2  # Center the grid
            y = (i // grid_size) * 15 - (grid_size * 15) / 2
            z = np.random.normal(0, 5)  # Add some variation in Z
            locations.append([x, y, z])
            names.append(f'CH_{i+1}')
        
        return np.array(locations), names
    
    def _generate_subject_all_channels_timecourse(self, subject_data, subject_idx, noise_range, output_dir):
        """Generate time course plot for all-channels analysis"""
        if 'all_channels_result' not in subject_data or subject_data['all_channels_result'] is None:
            print(f"    No all_channels_result for subject {subject_idx}")
            return
        
        all_channels_data = subject_data['all_channels_result']
        if 'window_times' not in all_channels_data or 'mean_accuracy' not in all_channels_data:
            print(f"    Missing window_times or mean_accuracy in all_channels_result for subject {subject_idx}")
            return
        
        time_points = np.array(all_channels_data['window_times'])
        accuracies = np.array(all_channels_data['mean_accuracy'])
        std_accuracies = np.array(all_channels_data.get('std_accuracy', np.zeros_like(accuracies)))
        
        # Apply 1D smoothing
        accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
        std_accuracies = self._apply_smoothing_1d(std_accuracies, self.config.get('smoothing_sigma', 1.0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot accuracy
        ax.plot(time_points, accuracies, 'b-', linewidth=2, label='All Channels')
        ax.fill_between(time_points, accuracies - std_accuracies, accuracies + std_accuracies,
                       alpha=0.3, color='blue', label='±1 SD')
        
        # Add chance level
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  linewidth=1, label='Chance Level')
        
        # Add stimulus onset
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.7, 
                  linewidth=1, label='Stimulus Onset')
        
        # Set labels and title
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Decoding Accuracy', fontsize=12)
        noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
        ax.set_title(f'Subject {subject_idx} - All Channels Decoding\n'
                    f'Noise Range: {noise_str}', fontsize=14, fontweight='bold')
        
        # Set limits
        ax.set_ylim(0.4, 0.8)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        filename = f'subject_{subject_idx}_all_channels_timecourse.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Generated all-channels timecourse: {filename}")
    
    def generate_level2_visualizations(self):
        """Level 2: Per noise level across subjects (all-channels only)"""
        print("\n" + "="*50)
        print("GENERATING LEVEL 2 VISUALIZATIONS")
        print("Per noise level across subjects (all-channels)")
        print("="*50)
        
        level2_dir = self.output_dir / "level2_per_noise_across_subjects"
        level2_dir.mkdir(exist_ok=True)
        
        for noise_range, noise_data in self.all_results.items():
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            print(f"\nProcessing noise range {noise_str}...")
            
            # Collect all-channels data across subjects
            subjects_data = []
            subject_indices = []
            
            for subject_idx, subject_data in noise_data.items():
                if ('all_channels_result' in subject_data and 
                    subject_data['all_channels_result'] is not None):
                    
                    all_channels_result = subject_data['all_channels_result']
                    if ('window_times' in all_channels_result and 
                        'mean_accuracy' in all_channels_result):
                        
                        time_points = np.array(all_channels_result['window_times'])
                        accuracies = np.array(all_channels_result['mean_accuracy'])
                        accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
                        
                        subjects_data.append(accuracies)
                        subject_indices.append(subject_idx)
            
            if not subjects_data:
                print(f"  No valid data for noise range {noise_str}")
                continue
            
            # Create combined plot
            self._generate_noise_level_summary(
                subjects_data, subject_indices, time_points, 
                noise_range, level2_dir
            )
    
    def _generate_noise_level_summary(self, subjects_data, subject_indices, time_points, 
                                    noise_range, output_dir):
        """Generate summary plot for a single noise level across subjects"""
        subjects_array = np.array(subjects_data)
        
        # Calculate statistics
        mean_acc = np.mean(subjects_array, axis=0)
        std_acc = np.std(subjects_array, axis=0)
        sem_acc = std_acc / np.sqrt(len(subjects_data))
        
        # Create figure with single subplot (removed redundant second subplot)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot individual subjects + average
        colors = plt.cm.tab10(np.linspace(0, 1, len(subject_indices)))
        for i, (subject_idx, acc) in enumerate(zip(subject_indices, subjects_data)):
            ax.plot(time_points, acc, alpha=0.6, linewidth=1, 
                   color=colors[i], label=f'Subject {subject_idx}')
        
        ax.plot(time_points, mean_acc, 'k-', linewidth=3, label='Mean')
        ax.fill_between(time_points, mean_acc - sem_acc, mean_acc + sem_acc, 
                       alpha=0.2, color='black', label='±SEM')
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Chance Level')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1, label='Stimulus Onset')
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Decoding Accuracy', fontsize=12)
        ax.set_ylim(0.4, 0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
        ax.set_title(f'All Subjects - Noise Range: {noise_str}\n'
                    f'Individual Traces + Mean±SEM (n={len(subjects_data)})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'noise_{noise_str}_across_subjects.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated summary plot: {filename}")
        
        # Also save statistics
        stats_data = {
            'time_points': time_points,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'sem_accuracy': sem_acc,
            'n_subjects': len(subjects_data),
            'subject_indices': subject_indices
        }
        
        stats_filename = f'noise_{noise_str}_statistics.pkl'
        with open(output_dir / stats_filename, 'wb') as f:
            pickle.dump(stats_data, f)
    
    def _is_significantly_above_chance(self, accuracies, threshold=0.55, min_proportion=0.1):
        """
        Check if accuracies are significantly above chance level
        
        Args:
            accuracies: Array of accuracy values
            threshold: Minimum threshold for significance
            min_proportion: Minimum proportion of time points that must be above threshold
        
        Returns:
            bool: True if significantly above chance
        """
        above_threshold = np.mean(accuracies > threshold)
        max_accuracy = np.max(accuracies)
        
        # Consider significant if:
        # 1. At least min_proportion of time points are above threshold, AND
        # 2. Maximum accuracy is above a reasonable threshold
        return above_threshold >= min_proportion and max_accuracy > 0.6
    
    def generate_level3_visualizations(self):
        """Level 3: Across noise levels across subjects"""
        print("\n" + "="*50)
        print("GENERATING LEVEL 3 VISUALIZATIONS")
        print("Across noise levels across subjects")
        print("="*50)
        
        level3_dir = self.output_dir / "level3_across_noise_across_subjects"
        level3_dir.mkdir(exist_ok=True)
        
        # Collect data for all noise ranges
        noise_summaries = {}
        
        for noise_range, noise_data in self.all_results.items():
            # Collect all-channels data across subjects for this noise range
            subjects_data = []
            
            for subject_idx, subject_data in noise_data.items():
                if ('all_channels_result' in subject_data and 
                    subject_data['all_channels_result'] is not None):
                    
                    all_channels_result = subject_data['all_channels_result']
                    if ('window_times' in all_channels_result and 
                        'mean_accuracy' in all_channels_result):
                        
                        accuracies = np.array(all_channels_result['mean_accuracy'])
                        accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
                        subjects_data.append(accuracies)
            
            if subjects_data:
                subjects_array = np.array(subjects_data)
                mean_acc = np.mean(subjects_array, axis=0)
                std_acc = np.std(subjects_array, axis=0)
                sem_acc = std_acc / np.sqrt(len(subjects_data))
                
                # Check if this noise level shows significant decoding
                is_significant = self._is_significantly_above_chance(mean_acc)
                
                noise_summaries[noise_range] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'sem': sem_acc,
                    'n_subjects': len(subjects_data),
                    'is_significant': is_significant
                }
        
        if not noise_summaries:
            print("No valid data for level 3 visualization")
            return
        
        # Get time points (assuming all have same time points)
        first_noise_data = list(self.all_results.values())[0]
        first_subject_data = list(first_noise_data.values())[0]
        time_points = np.array(first_subject_data['all_channels_result']['window_times'])
        
        # Generate comparison plots
        self._generate_noise_comparison_plot(noise_summaries, time_points, level3_dir)
        self._generate_noise_heatmap_comparison(noise_summaries, time_points, level3_dir)
        self._generate_improved_peak_analysis(noise_summaries, time_points, level3_dir)
        self._generate_channel_level_analysis(level3_dir)
    
    def _generate_improved_peak_analysis(self, noise_summaries, time_points, output_dir):
        """Generate improved peak decoding analysis that handles low-significance cases"""
        # Extract peak information only for significant conditions
        peak_data = []
        
        for noise_range, data in noise_summaries.items():
            mean_acc = data['mean']
            is_significant = data['is_significant']
            
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            
            if is_significant:
                # Find peak accuracy and time
                peak_idx = np.argmax(mean_acc)
                peak_accuracy = mean_acc[peak_idx]
                peak_time = time_points[peak_idx]
                
                # Find first time point above 55% after stimulus onset
                onset_idx = np.argmin(np.abs(time_points))
                post_onset_mask = np.arange(len(time_points)) >= onset_idx
                above_chance = mean_acc > 0.55
                
                first_significant_idx = None
                if np.any(above_chance & post_onset_mask):
                    first_significant_idx = np.where(above_chance & post_onset_mask)[0][0]
                    first_significant_time = time_points[first_significant_idx]
                else:
                    first_significant_time = np.nan
                
                # Calculate area under curve above chance
                auc_above_chance = np.trapz(np.maximum(mean_acc - 0.5, 0), time_points)
                
                peak_data.append({
                    'noise_range': noise_str,
                    'noise_min': noise_range[0],
                    'noise_max': noise_range[1],
                    'noise_center': (noise_range[0] + noise_range[1]) / 2,
                    'peak_accuracy': peak_accuracy,
                    'peak_time': peak_time,
                    'first_significant_time': first_significant_time,
                    'auc_above_chance': auc_above_chance,
                    'n_subjects': data['n_subjects'],
                    'is_significant': True
                })
            else:
                # For non-significant conditions, record basic info
                max_accuracy = np.max(mean_acc)
                peak_data.append({
                    'noise_range': noise_str,
                    'noise_min': noise_range[0],
                    'noise_max': noise_range[1],
                    'noise_center': (noise_range[0] + noise_range[1]) / 2,
                    'peak_accuracy': max_accuracy,
                    'peak_time': np.nan,
                    'first_significant_time': np.nan,
                    'auc_above_chance': 0,
                    'n_subjects': data['n_subjects'],
                    'is_significant': False
                })
        
        # Create improved peak analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Separate significant and non-significant conditions
        sig_data = [d for d in peak_data if d['is_significant']]
        nonsig_data = [d for d in peak_data if not d['is_significant']]
        
        # Plot 1: Peak accuracy vs noise level (with significance indication)
        all_noise_centers = [d['noise_center'] for d in peak_data]
        all_peak_accuracies = [d['peak_accuracy'] for d in peak_data]
        all_significances = [d['is_significant'] for d in peak_data]

        # Plot all points connected with a single line
        ax1.plot(all_noise_centers, all_peak_accuracies, 'k-', linewidth=1, alpha=0.5)

        # Overlay colored markers for significance
        sig_data = [d for d in peak_data if d['is_significant']]
        nonsig_data = [d for d in peak_data if not d['is_significant']]

        if sig_data:
            sig_noise_centers = [d['noise_center'] for d in sig_data]
            sig_peak_accuracies = [d['peak_accuracy'] for d in sig_data]
            ax1.scatter(sig_noise_centers, sig_peak_accuracies, color='blue', s=80, 
                    marker='o', label='Significant decoding', zorder=5)

        if nonsig_data:
            nonsig_noise_centers = [d['noise_center'] for d in nonsig_data]
            nonsig_peak_accuracies = [d['peak_accuracy'] for d in nonsig_data]
            ax1.scatter(nonsig_noise_centers, nonsig_peak_accuracies, color='red', s=80, 
                    marker='s', alpha=0.7, label='Non-significant decoding', zorder=5)

        ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Chance level')
        ax1.set_xlabel('Noise Level (center)', fontsize=12)
        ax1.set_ylabel('Peak Accuracy', fontsize=12)
        ax1.set_title('Peak Decoding Accuracy vs Noise Level', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Peak timing vs noise level (include all points, mark significance)
        all_peak_times = [d['peak_time'] if not np.isnan(d['peak_time']) else None for d in peak_data]
        valid_indices = [i for i, t in enumerate(all_peak_times) if t is not None]

        if valid_indices:
            valid_noise_centers = [all_noise_centers[i] for i in valid_indices]
            valid_peak_times = [all_peak_times[i] for i in valid_indices]
            valid_significances = [all_significances[i] for i in valid_indices]
            
            # Plot connecting line
            ax2.plot(valid_noise_centers, valid_peak_times, 'k-', linewidth=1, alpha=0.5)
            
            # Overlay colored markers
            for i, (noise_center, peak_time, is_sig) in enumerate(zip(valid_noise_centers, valid_peak_times, valid_significances)):
                color = 'blue' if is_sig else 'red'
                marker = 'o' if is_sig else 's'
                alpha = 1.0 if is_sig else 0.7
                ax2.scatter(noise_center, peak_time, color=color, s=80, marker=marker, alpha=alpha, zorder=5)
            
            ax2.set_xlabel('Noise Level (center)', fontsize=12)
            ax2.set_ylabel('Peak Time (ms)', fontsize=12)
            ax2.set_title('Peak Timing vs Noise Level\n(Blue=Significant, Red=Non-significant)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=8, label='Significant'),
                            Line2D([0], [0], marker='s', color='red', linestyle='None', markersize=8, alpha=0.7, label='Non-significant')]
            ax2.legend(handles=legend_elements)
        else:
            ax2.text(0.5, 0.5, 'No valid\npeak times', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Peak Timing vs Noise Level', fontweight='bold')
                
        # Plot 3: Area under curve above chance
        if sig_data:
            sig_auc = [d['auc_above_chance'] for d in sig_data]
            ax3.plot(sig_noise_centers, sig_auc, 'mo-', linewidth=2, markersize=8)
            ax3.set_xlabel('Noise Level (center)', fontsize=12)
            ax3.set_ylabel('AUC above chance', fontsize=12)
            ax3.set_title('Decoding Strength vs Noise Level\n(Area Under Curve)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No significant\ndecoding', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Decoding Strength vs Noise Level', fontweight='bold')
        
        # Plot 4: Summary statistics table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for d in peak_data:
            significance_marker = "✓" if d['is_significant'] else "✗"
            table_data.append([
                d['noise_range'],
                f"{d['peak_accuracy']:.3f}",
                f"{d['peak_time']:.0f}" if not np.isnan(d['peak_time']) else "N/A",
                f"{d['first_significant_time']:.0f}" if not np.isnan(d['first_significant_time']) else "N/A",
                significance_marker,
                f"{d['n_subjects']}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Noise Range', 'Peak Acc', 'Peak Time (ms)', 
                                  'Onset Time (ms)', 'Significant', 'N Subjects'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title('Summary Statistics\n(✓ = Significant decoding, ✗ = At chance level)', 
                     fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improved_peak_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed peak data
        peak_df = pd.DataFrame(peak_data)
        peak_df.to_csv(output_dir / 'improved_peak_analysis.csv', index=False)
        
        print("  Generated improved peak analysis")
    
    def _generate_noise_comparison_plot(self, noise_summaries, time_points, output_dir):
        """Generate comparison plot across all noise levels"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(noise_summaries)))
        
        # Plot 1: Mean traces with error bars
        for i, (noise_range, data) in enumerate(noise_summaries.items()):
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            color = colors[i]
            
            linestyle = '-' if data['is_significant'] else '--'
            alpha = 1.0 if data['is_significant'] else 0.6
            
            ax1.plot(time_points, data['mean'], color=color, linewidth=2, 
                    linestyle=linestyle, alpha=alpha,
                    label=f'Noise {noise_str} (n={data["n_subjects"]}){"*" if data["is_significant"] else ""}')
            ax1.fill_between(time_points, 
                           data['mean'] - data['sem'], 
                           data['mean'] + data['sem'], 
                           alpha=0.2 * alpha, color=color)
        
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1, 
                   label='Chance Level')
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1, 
                   label='Stimulus Onset')
        
        ax1.set_ylabel('Decoding Accuracy', fontsize=12)
        ax1.set_ylim(0.4, 0.8)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_title('Decoding Accuracy Across Noise Levels\nMean ± SEM (* = Significant decoding)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 2: Difference from highest noise condition (only for significant conditions)
        baseline_noise = max(noise_summaries.keys())
        baseline_mean = noise_summaries[baseline_noise]['mean']
        
        for i, (noise_range, data) in enumerate(noise_summaries.items()):
            if noise_range == baseline_noise or not data['is_significant']:
                continue
            
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            color = colors[i]
            
            diff = data['mean'] - baseline_mean
            ax2.plot(time_points, diff, color=color, linewidth=2, 
                    label=f'Noise {noise_str} - Baseline')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Accuracy Difference', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        baseline_str = f"{baseline_noise[0]:.1f}-{baseline_noise[1]:.1f}"
        ax2.set_title(f'Difference from Baseline (Noise {baseline_str}) - Significant Conditions Only', 
                     fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'noise_levels_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Generated noise comparison plot")
    
    def _generate_noise_heatmap_comparison(self, noise_summaries, time_points, output_dir):
        """Generate heatmap showing all noise levels"""
        # Create matrix: rows = noise levels, columns = time points
        noise_ranges = sorted(noise_summaries.keys())
        n_noise = len(noise_ranges)
        n_times = len(time_points)
        
        accuracy_matrix = np.zeros((n_noise, n_times))
        noise_labels = []
        
        for i, noise_range in enumerate(noise_ranges):
            accuracy_matrix[i, :] = noise_summaries[noise_range]['mean']
            significance_marker = " *" if noise_summaries[noise_range]['is_significant'] else ""
            noise_labels.append(f"{noise_range[0]:.1f}-{noise_range[1]:.1f}{significance_marker}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 6))
        
        im = ax.imshow(accuracy_matrix, aspect='auto', cmap='RdYlBu_r', 
                      vmin=0.4, vmax=0.8, interpolation='bilinear')
        
        # Set labels
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Noise Level', fontsize=12)
        
        # Set time ticks
        time_ticks = np.arange(0, n_times, max(1, n_times // 15))
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f"{time_points[i]:.0f}" for i in time_ticks])
        
        # Set noise level ticks
        ax.set_yticks(np.arange(n_noise))
        ax.set_yticklabels(noise_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Decoding Accuracy', fontsize=12)
        
        # Add title
        ax.set_title('Decoding Accuracy Heatmap Across Noise Levels\n'
                    'Grand Average Across Subjects (* = Significant decoding)', 
                    fontsize=14, fontweight='bold')
        
        # Add stimulus onset line
        stimulus_onset_idx = np.argmin(np.abs(time_points))
        ax.axvline(x=stimulus_onset_idx, color='white', linestyle='--', 
                  alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'noise_levels_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Generated noise heatmap")
    
    def _generate_channel_level_analysis(self, output_dir):
        """Generate channel-level analysis across noise levels"""
        print("  Generating channel-level analysis...")
        
        # Collect all channel data across noise levels
        all_channel_data = {}
        
        for noise_range, noise_data in self.all_results.items():
            noise_str = f"{noise_range[0]:.1f}-{noise_range[1]:.1f}"
            
            for subject_idx, subject_data in noise_data.items():
                if 'channel_results' not in subject_data:
                    continue
                
                for channel_name, channel_result in subject_data['channel_results'].items():
                    if 'window_times' in channel_result and 'mean_accuracy' in channel_result:
                        key = f"S{subject_idx}_{channel_name}"
                        
                        if key not in all_channel_data:
                            all_channel_data[key] = {}
                        
                        accuracies = np.array(channel_result['mean_accuracy'])
                        accuracies = self._apply_smoothing_1d(accuracies, self.config.get('smoothing_sigma', 1.0))
                        
                        all_channel_data[key][noise_str] = {
                            'accuracies': accuracies,
                            'time_points': np.array(channel_result['window_times']),
                            'peak_accuracy': np.max(accuracies),
                            'peak_time': channel_result['window_times'][np.argmax(accuracies)],
                            'subject_idx': subject_idx,
                            'channel_name': channel_name,
                            'is_significant': self._is_significantly_above_chance(accuracies)
                        }
        
        if not all_channel_data:
            print("    No channel data found")
            return
        
        # Create grand average heatmap for all channels
        self._generate_grand_average_channel_heatmap(all_channel_data, output_dir)
        
        # Create peak accuracy comparison
        self._generate_channel_peak_comparison(all_channel_data, output_dir)
    
    def _generate_grand_average_channel_heatmap(self, all_channel_data, output_dir):
        """Generate grand average heatmap for all channels across noise levels"""
        # Get noise levels
        noise_levels = []
        for channel_data in all_channel_data.values():
            noise_levels.extend(channel_data.keys())
        noise_levels = sorted(list(set(noise_levels)))
        
        # Get time points (assuming all have same time points)
        first_channel = list(all_channel_data.values())[0]
        first_noise = list(first_channel.keys())[0]
        time_points = first_channel[first_noise]['time_points']
        
        # Create separate heatmaps for each noise level
        fig, axes = plt.subplots(len(noise_levels), 1, figsize=(15, 4 * len(noise_levels)))
        if len(noise_levels) == 1:
            axes = [axes]
        
        for noise_idx, noise_level in enumerate(noise_levels):
            # Collect data for this noise level
            channel_accuracies = []
            channel_labels = []
            
            for channel_key, channel_data in all_channel_data.items():
                if noise_level in channel_data:
                    channel_accuracies.append(channel_data[noise_level]['accuracies'])
                    significance_marker = " *" if channel_data[noise_level]['is_significant'] else ""
                    channel_labels.append(f"{channel_key}{significance_marker}")
            
            if not channel_accuracies:
                continue
            
            # Create heatmap
            accuracy_matrix = np.array(channel_accuracies)
            n_channels, n_times = accuracy_matrix.shape
            
            im = axes[noise_idx].imshow(accuracy_matrix, aspect='auto', cmap='RdYlBu_r', 
                                       vmin=0.4, vmax=0.8, interpolation='bilinear')
            
            # Set labels
            axes[noise_idx].set_ylabel('Channels', fontsize=10)
            axes[noise_idx].set_title(f'All Channels - Noise Level: {noise_level}\n(* = Significant decoding)', 
                                     fontsize=12, fontweight='bold')
            
            # Set time ticks
            time_ticks = np.arange(0, n_times, max(1, n_times // 15))
            axes[noise_idx].set_xticks(time_ticks)
            axes[noise_idx].set_xticklabels([f"{time_points[i]:.0f}" for i in time_ticks])
            
            # Set channel ticks (show every 10th channel if too many)
            if n_channels > 20:
                channel_ticks = np.arange(0, n_channels, 10)
                axes[noise_idx].set_yticks(channel_ticks)
                axes[noise_idx].set_yticklabels([channel_labels[i] for i in channel_ticks], fontsize=6)
            else:
                axes[noise_idx].set_yticks(np.arange(n_channels))
                axes[noise_idx].set_yticklabels(channel_labels, fontsize=6)
            
            # Add stimulus onset line
            stimulus_onset_idx = np.argmin(np.abs(time_points))
            axes[noise_idx].axvline(x=stimulus_onset_idx, color='white', linestyle='--', 
                                   alpha=0.8, linewidth=2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[noise_idx], shrink=0.8)
            cbar.set_label('Decoding Accuracy', fontsize=10)
        
        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'grand_average_channel_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("    Generated grand average channel heatmaps")
    
    def _generate_channel_peak_comparison(self, all_channel_data, output_dir):
        """Generate peak accuracy comparison across noise levels"""
        # Extract peak data
        peak_data = []
        
        for channel_key, channel_data in all_channel_data.items():
            for noise_level, data in channel_data.items():
                peak_data.append({
                    'channel': channel_key,
                    'subject_idx': data['subject_idx'],
                    'channel_name': data['channel_name'],
                    'noise_level': noise_level,
                    'peak_accuracy': data['peak_accuracy'],
                    'peak_time': data['peak_time'],
                    'is_significant': data['is_significant']
                })
        
        df = pd.DataFrame(peak_data)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Box plot of peak accuracies by noise level (separated by significance)
        sig_df = df[df['is_significant'] == True]
        nonsig_df = df[df['is_significant'] == False]
        
        if not sig_df.empty:
            sns.boxplot(data=sig_df, x='noise_level', y='peak_accuracy', ax=ax1, color='lightblue')
            ax1.set_title('Peak Accuracy Distribution by Noise Level\n(Significant channels only)', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No significant\nchannels found', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Peak Accuracy Distribution by Noise Level', fontweight='bold')
        
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance Level')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Peak Accuracy')
        ax1.legend()
        
        # Plot 2: Box plot of peak times by noise level (significant channels only)
        if not sig_df.empty:
            sns.boxplot(data=sig_df, x='noise_level', y='peak_time', ax=ax2, color='lightgreen')
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7, label='Stimulus Onset')
            ax2.set_title('Peak Time Distribution by Noise Level\n(Significant channels only)', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No significant\nchannels found', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Peak Time Distribution by Noise Level', fontweight='bold')
        
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Peak Time (ms)')
        ax2.legend()
        
        # Plot 3: Mean peak accuracy by noise level (significant vs non-significant)
        mean_peak_acc_sig = sig_df.groupby('noise_level')['peak_accuracy'].agg(['mean', 'std']).reset_index() if not sig_df.empty else pd.DataFrame()
        mean_peak_acc_nonsig = nonsig_df.groupby('noise_level')['peak_accuracy'].agg(['mean', 'std']).reset_index() if not nonsig_df.empty else pd.DataFrame()
        
        if not mean_peak_acc_sig.empty:
            ax3.errorbar(range(len(mean_peak_acc_sig)), mean_peak_acc_sig['mean'], 
                        yerr=mean_peak_acc_sig['std'], marker='o', linewidth=2, markersize=8, 
                        label='Significant channels', color='blue')
        
        if not mean_peak_acc_nonsig.empty:
            x_offset = 0.1
            ax3.errorbar(np.arange(len(mean_peak_acc_nonsig)) + x_offset, mean_peak_acc_nonsig['mean'], 
                        yerr=mean_peak_acc_nonsig['std'], marker='s', linewidth=2, markersize=8, 
                        label='Non-significant channels', color='red', alpha=0.7)
        
        if not mean_peak_acc_sig.empty:
            ax3.set_xticks(range(len(mean_peak_acc_sig)))
            ax3.set_xticklabels(mean_peak_acc_sig['noise_level'])
        elif not mean_peak_acc_nonsig.empty:
            ax3.set_xticks(range(len(mean_peak_acc_nonsig)))
            ax3.set_xticklabels(mean_peak_acc_nonsig['noise_level'])
        
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance Level')
        ax3.set_title('Mean Peak Accuracy by Noise Level', fontweight='bold')
        ax3.set_xlabel('Noise Level')
        ax3.set_ylabel('Mean Peak Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Percentage of significant channels by noise level
        threshold_data = []
        for noise_level in df['noise_level'].unique():
            noise_df = df[df['noise_level'] == noise_level]
            n_total = len(noise_df)
            n_significant = len(noise_df[noise_df['is_significant'] == True])
            n_above_60 = len(noise_df[noise_df['peak_accuracy'] > 0.6])
            n_above_70 = len(noise_df[noise_df['peak_accuracy'] > 0.7])
            
            threshold_data.append({
                'noise_level': noise_level,
                'total_channels': n_total,
                'n_significant': n_significant,
                'above_60': n_above_60,
                'above_70': n_above_70,
                'pct_significant': 100 * n_significant / n_total,
                'pct_above_60': 100 * n_above_60 / n_total,
                'pct_above_70': 100 * n_above_70 / n_total
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        
        x_pos = np.arange(len(threshold_df))
        width = 0.25
        
        ax4.bar(x_pos - width, threshold_df['pct_significant'], width, 
               label='Significant decoding', alpha=0.7, color='green')
        ax4.bar(x_pos, threshold_df['pct_above_60'], width, 
               label='> 60% accuracy', alpha=0.7, color='blue')
        ax4.bar(x_pos + width, threshold_df['pct_above_70'], width, 
               label='> 70% accuracy', alpha=0.7, color='red')
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(threshold_df['noise_level'])
        ax4.set_title('Channel Performance by Noise Level', fontweight='bold')
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Percentage of Channels')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'channel_peak_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics
        df.to_csv(output_dir / 'channel_peak_statistics.csv', index=False)
        threshold_df.to_csv(output_dir / 'threshold_statistics.csv', index=False)
        
        print("    Generated channel peak comparison")
    
    def generate_all_visualizations(self):
        """Generate all three levels of visualizations"""
        print("Starting comprehensive noise range analysis...")
        print(f"Base directory: {self.base_results_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Generate all levels
        self.generate_level1_visualizations()
        self.generate_level2_visualizations()
        self.generate_level3_visualizations()
        
        print(f"\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*50)

def main():
    """Main function to run the noise range visualization analysis"""
    
    # Configuration
    config = {
        'apply_smoothing': True,  # Apply Gaussian smoothing to accuracies
        'smoothing_sigma': 1.0,   # Sigma for Gaussian smoothing (1D only)
        'figure_dpi': 300,        # DPI for saved figures
        'show_individual_subjects': True,  # Show individual subject traces in level 2
    }
    
    # Set base results directory
    # This should be the parent directory containing all noise range folders
    base_results_dir = r"."
    
    # Check if directory exists
    if not os.path.exists(base_results_dir):
        print(f"Error: Base results directory not found: {base_results_dir}")
        print("Please update the base_results_dir path in the script.")
        return
    
    # Create visualizer and generate all plots
    visualizer = NoiseRangeVisualizer(base_results_dir, config)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()