import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

def segment_signals_by_events(df, signal_col='eeg_signal', subject_id_col='subject_id'):
    """
    Finds significant peaks in each subject's signal and extracts segments around them.

    Args:
        df (pd.DataFrame): The input DataFrame with continuous signals.
        signal_col (str): The name of the signal column.
        subject_id_col (str): The column identifying unique subjects.

    Returns:
        pd.DataFrame: A new DataFrame containing only the extracted segments.
    """

    # --- Configuration for Peak Finding ---
    # How many time steps to extract before and after the peak (total segment length will be 2 * WINDOW_RADIUS)
    WINDOW_RADIUS = 128
    # How high a peak must be (relative to the signal's range) to be considered significant
    MIN_PEAK_HEIGHT_PERCENTILE = 75
    # How far apart peaks must be (in time steps) to be considered distinct events
    MIN_PEAK_DISTANCE = 256 # Should be at least 2 * WINDOW_RADIUS

    all_segments = []
    total_subjects = len(df[subject_id_col].unique())

    # Group by each subject to process their signal individually
    grouped = df.groupby(subject_id_col)

    print("Starting signal segmentation...")
    for i, (subject_id, subject_df) in enumerate(grouped):
        print(f"  Processing subject {i+1}/{total_subjects}...", end='\r')
        signal = subject_df[signal_col].values

        # Calculate a dynamic threshold for what counts as a "significant" peak
        min_height = np.percentile(np.abs(signal), MIN_PEAK_HEIGHT_PERCENTILE)

        # Find all significant peaks in the signal
        peak_indices, _ = find_peaks(signal, height=min_height, distance=MIN_PEAK_DISTANCE)

        # Also find the troughs by inverting the signal
        trough_indices, _ = find_peaks(-signal, height=min_height, distance=MIN_PEAK_DISTANCE)

        # Combine and sort all found events
        event_indices = sorted(list(set(peak_indices) | set(trough_indices)))

        segment_counter = 0
        for idx in event_indices:
            start = idx - WINDOW_RADIUS
            end = idx + WINDOW_RADIUS

            # Ensure the segment is not out of bounds
            if start >= 0 and end <= len(signal):
                segment_df = subject_df.iloc[start:end].copy()
                # Create a unique ID for each segment
                segment_df['segment_id'] = f"{subject_id}_{segment_counter}"
                all_segments.append(segment_df)
                segment_counter += 1

    print("\nSegmentation complete.")
    if not all_segments:
        print("Warning: No segments were extracted. You may need to adjust the peak finding parameters.")
        return pd.DataFrame()

    return pd.concat(all_segments, ignore_index=True)

# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    INPUT_CSV_PATH = "simulated_eeg_multivariate_dataset.csv"
    OUTPUT_CSV_PATH = "segmented_eeg_dataset.csv"

    # Load the raw, continuous data
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        continuous_data_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
        exit()

    # Run the segmentation process
    segmented_df = segment_signals_by_events(continuous_data_df)

    # Save the new dataset of segments
    if not segmented_df.empty:
        segmented_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSegmented dataset saved to '{OUTPUT_CSV_PATH}'")
        print(f"Original number of rows: {len(continuous_data_df)}")
        print(f"New number of rows: {len(segmented_df)}")
        print(f"Number of unique segments extracted: {segmented_df['segment_id'].nunique()}")