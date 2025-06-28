import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Function to apply Artifact Subspace Reconstruction (ASR) with automatic cutoff
def asr(data, sampling_rate, reference_window=1, cutoff_multiplier=5):
    """
    Implements ASR algorithm for artifact removal with automatic cutoff determination.
    
    Parameters:
    - data: EEG data (channels x samples).
    - sampling_rate: Sampling rate in Hz.
    - reference_window: Duration (in seconds) to compute reference covariance.
    - cutoff_multiplier: Multiplier for the standard deviation of eigenvalues.
    
    Returns:
    - cleaned_data: Artifact-cleaned EEG data.
    """
    # Step 1: Compute Reference Covariance
    reference_samples = int(reference_window * sampling_rate)
    clean_data_segment = data[:, :reference_samples]  # Assume first N seconds are clean
    ref_cov = np.cov(clean_data_segment)

    # Step 2: Calculate Eigenvalue Statistics
    eigenvalues = np.linalg.eigvals(ref_cov)
    eigen_std = np.std(eigenvalues)  # Standard deviation of eigenvalues
    cutoff = eigen_std * cutoff_multiplier  # Automatic cutoff

    print(f"Computed cutoff: {cutoff:.3f}")

    # Step 3: Detect and Clean Artifacts
    sliding_window_size = int(sampling_rate)  # 1-second sliding window
    cleaned_data = data.copy()
    
    for start in range(0, data.shape[1] - sliding_window_size, sliding_window_size):
        end = start + sliding_window_size
        segment = data[:, start:end]
        seg_cov = np.cov(segment)
        
        # Compare covariance matrices
        eigenvalues = np.linalg.eigvals(np.linalg.inv(ref_cov).dot(seg_cov))
        max_deviation = np.max(eigenvalues)
        
        # Debug: Print max_deviation values to see if artifact detection is working
        if max_deviation > cutoff:  # Artifact detected
            print(f"Artifact detected at segment {start} - {end}: max deviation = {max_deviation:.3f}")
            
            # Step 4: Remove Artifacted Subspace
            u, s, vh = svd(segment, full_matrices=False)
            artifact_subspace = u[:, s > cutoff]
            segment_cleaned = segment - artifact_subspace @ artifact_subspace.T @ segment
            cleaned_data[:, start:end] = segment_cleaned

    return cleaned_data

# Function to load EEG data from a .pkl file
def load_eeg_data(filepath, eeg_columns):
    """
    Loads EEG data from a .pkl file containing a pandas DataFrame.
    
    Parameters:
    - filepath: Path to the .pkl file containing EEG data.
    - eeg_columns: List of channel names.
    
    Returns:
    - eeg_data: EEG data (channels x samples).
    - sampling_rate: Sampling rate in Hz.
    """
    df = pd.read_pickle(filepath)  # Load DataFrame from .pkl file
    eeg_data = df[eeg_columns].values.T  # Extract EEG channels and transpose
    sampling_rate = 1000  # Set the actual sampling rate (adjust if needed)
    return eeg_data, sampling_rate

# Function to plot EEG data (original vs cleaned) side by side
def plot_eeg_side_by_side(original, cleaned, sampling_rate, channel_idx=0, zoom_start=0, zoom_end=None):
    """
    Plots the original and cleaned EEG data side by side for comparison.
    
    Parameters:
    - original: Original EEG data (channels x samples).
    - cleaned: Cleaned EEG data (channels x samples).
    - sampling_rate: Sampling rate in Hz.
    - channel_idx: Index of the channel to plot.
    - zoom_start: Start time for zooming (in seconds).
    - zoom_end: End time for zooming (in seconds).
    """
    time = np.arange(original.shape[1]) / sampling_rate
    
    # Zoom into a specific time window if provided
    if zoom_end is None:
        zoom_end = original.shape[1] / sampling_rate
    
    zoom_mask = (time >= zoom_start) & (time <= zoom_end)
    
    # Create subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Plot Original EEG
    ax1.plot(time[zoom_mask], original[channel_idx, zoom_mask], label="Original EEG", color='blue', linestyle='-', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.set_title(f"Original EEG (Channel {channel_idx + 1})", fontsize=14)
    ax1.axvline(x=zoom_start, color='red', linestyle='--', label="Zoom Start")
    ax1.axvline(x=zoom_end, color='red', linestyle='--', label="Zoom End")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=12)
    
    # Plot Cleaned EEG
    ax2.plot(time[zoom_mask], cleaned[channel_idx, zoom_mask], label="Cleaned EEG", color='green', linestyle='-', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"Cleaned EEG (Channel {channel_idx + 1})", fontsize=14)
    ax2.axvline(x=zoom_start, color='red', linestyle='--', label="Zoom Start")
    ax2.axvline(x=zoom_end, color='red', linestyle='--', label="Zoom End")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Main Execution Script
if __name__ == "__main__":
    # Path to the .pkl file containing EEG data
    filepath = r'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\ImaginedSpeechData\MM05\GAM\df_epochs_filtered.pkl'  # Update with your actual file path

    # List of EEG channels (adjust based on your dataset)
    eeg_columns = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8',
                   'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5',
                   'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'PO3',
                   'PO4', 'O1', 'OZ', 'O2']  # Replace with your actual EEG channels

    # Step 1: Load the EEG data
    eeg_data, sfreq = load_eeg_data(filepath, eeg_columns)

    # Step 2: Apply ASR for artifact removal with automatic cutoff
    eeg_cleaned = asr(eeg_data, sampling_rate=sfreq, reference_window=1, cutoff_multiplier=5)

    # Step 3: Plot the original and cleaned EEG data side by side (first channel)
    plot_eeg_side_by_side(eeg_data, eeg_cleaned, sampling_rate=sfreq, channel_idx=0, zoom_start=0, zoom_end=10)







