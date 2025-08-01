import pandas as pd
import numpy as np

def segment_signals_with_overlap(df, signal_cols, ancillary_cols, subject_id_col, label_col):
    """
    Segments continuous signals into fixed-size, overlapping windows.

    Args:
        df (pd.DataFrame): The input DataFrame with continuous signals.
        signal_cols (list): List of signal column names.
        ancillary_cols (list): List of ancillary feature column names.
        subject_id_col (str): The column identifying unique subjects.
        label_col (str): The column for the class label.

    Returns:
        pd.DataFrame: A new DataFrame containing the extracted segments.
    """

    # --- Configuration for Segmentation ---
    WINDOW_SIZE = 256  # The length of each segment (e.g., 1 second of data)
    STRIDE = 64        # How far to slide the window for the next segment (75% overlap)

    all_segments = []
    total_subjects = len(df[subject_id_col].unique())

    # Group by each subject to process their signal individually
    grouped = df.groupby(subject_id_col)

    print("Starting signal segmentation with overlapping windows...")
    for i, (subject_id, subject_df) in enumerate(grouped):
        print(f"  Processing subject {i+1}/{total_subjects}...", end='\r')

        # Get the static data for this subject (it's the same for all rows)
        static_data = subject_df.iloc[0]
        label = static_data[label_col]
        ancillary_values = static_data[ancillary_cols]

        # Slide the window across the signal data
        for start in range(0, len(subject_df) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            segment = subject_df.iloc[start:end].copy()

            # Create a new DataFrame for this segment
            # The time_step is now relative to the start of the segment
            segment_data = {'time_step': np.arange(WINDOW_SIZE)}

            # Add the signal data
            for col in signal_cols:
                segment_data[col] = segment[col].values

            # Add the static ancillary data and label
            for col in ancillary_cols:
                segment_data[col] = ancillary_values[col]
            segment_data[label_col] = label
            
            # Create a unique ID for each segment
            segment_data['segment_id'] = f"{subject_id}_{start}"

            all_segments.append(pd.DataFrame(segment_data))

    print("\nSegmentation complete.")
    if not all_segments:
        print("Warning: No segments were extracted.")
        return pd.DataFrame()

    return pd.concat(all_segments, ignore_index=True)

# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    INPUT_CSV_PATH = "simulated_eeg_multivariate_dataset.csv"
    OUTPUT_CSV_PATH = "segmented_eeg_dataset.csv"
    
    # Define which columns are signals vs. static features
    SIGNAL_COLUMNS = ["eeg_signal"]
    ANCILLARY_COLUMNS = ["caffeine_mg", "sugar_g"]
    SUBJECT_ID_COLUMN = "subject_id"
    LABEL_COLUMN = "class"

    # Load the raw, continuous data
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        continuous_data_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
        exit()

    # Run the segmentation process
    segmented_df = segment_signals_with_overlap(
        df=continuous_data_df,
        signal_cols=SIGNAL_COLUMNS,
        ancillary_cols=ANCILLARY_COLUMNS,
        subject_id_col=SUBJECT_ID_COLUMN,
        label_col=LABEL_COLUMN
    )

    # Save the new dataset of segments
    if not segmented_df.empty:
        # Reorder columns for clarity
        final_cols = ['segment_id', 'time_step', LABEL_COLUMN] + SIGNAL_COLUMNS + ANCILLARY_COLUMNS
        segmented_df = segmented_df[final_cols]
        
        segmented_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSegmented dataset saved to '{OUTPUT_CSV_PATH}'")
        print(f"Original number of rows: {len(continuous_data_df)}")
        print(f"New number of rows: {len(segmented_df)}")
        print(f"Number of unique segments extracted: {segmented_df['segment_id'].nunique()}")