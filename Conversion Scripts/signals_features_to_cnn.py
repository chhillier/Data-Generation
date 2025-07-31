import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import List, Tuple, Dict

def process_long_format_to_cnn(
    df: pd.DataFrame, 
    signal_cols: List[str], 
    ancillary_cols: List[str], 
    subject_id_col: str, 
    label_col: str
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Converts a long-format time series DataFrame into a 3D tensor suitable for CNNs.
    Handles multiple signal columns and multiple ancillary columns.

    Args:
        df (pd.DataFrame): The input DataFrame in long format.
        signal_cols (List[str]): The names of the primary time series signal columns.
        ancillary_cols (List[str]): A list of column names for the ancillary features.
        subject_id_col (str): The column name for the unique sample/subject identifier.
        label_col (str): The column name for the target class label.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: A tuple containing:
            - X (np.ndarray): The 3D feature tensor of shape (samples, time_steps, features).
            - y (np.ndarray): The 1D label vector.
            - class_mapping (Dict): A dictionary mapping encoded labels back to original class names.
    """
    print("Starting preprocessing...")

    # --- Step 1: Feature Scaling ---
    # Combine all signal and ancillary columns to scale them at once.
    feature_columns = signal_cols + ancillary_cols
    
    print(f"Scaling features: {feature_columns}")
    scaler = MinMaxScaler()
    # Ensure we only scale existing columns
    cols_to_scale = [col for col in feature_columns if col in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print("Features scaled.")

    # --- Step 2: Label Encoding ---
    print(f"Encoding label column: '{label_col}'")
    label_encoder = LabelEncoder()
    encoded_label_col = f"{label_col}_encoded"
    df[encoded_label_col] = label_encoder.fit_transform(df[label_col])
    class_mapping = {i: c for i, c in enumerate(label_encoder.classes_)}
    print(f"Labels encoded. Mapping: {class_mapping}")

    # --- Step 3: Reshaping the Data ---
    print(f"Grouping by '{subject_id_col}' and reshaping...")
    X = []
    y = []
    
    grouped = df.groupby(subject_id_col)
    
    for _, subject_df in grouped:
        # Extract the values of the feature columns for the current subject
        features = subject_df[feature_columns].values
        
        # The label is the same for all time steps of a subject, so we take the first one
        label = subject_df[encoded_label_col].iloc[0]
        
        X.append(features)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    print("Data reshaped successfully.")
    
    return X, y, class_mapping

# --- Main execution block to demonstrate usage ---
if __name__ == "__main__":
    
    # ===================================================================
    # --- CONFIGURATION: EDIT THESE VARIABLES TO MATCH YOUR DATASET ---
    # ===================================================================
    
    # 1. Path to your input CSV file.
    INPUT_CSV_PATH = "data/simulated_eeg_multivariate_dataset.csv"

    # 2. List of the main signal column(s).
    SIGNAL_COLUMNS = ["eeg_signal"]
    
    # 3. List of the ancillary (static) feature column(s).
    ANCILLARY_COLUMNS = ["caffeine_mg", "sugar_g"]
    
    # 4. The column that identifies unique subjects or samples.
    SUBJECT_ID_COLUMN = "subject_id"
    
    # 5. The column that contains the class labels you want to predict.
    LABEL_COLUMN = "class"
    
    # 6. The prefix for the output files.
    OUTPUT_PREFIX = "outputs/cnn_ready_data"
    
    # ===================================================================
    # --- END OF CONFIGURATION ---
    # ===================================================================

    # Load the data from the specified CSV file
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        input_df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found. Please check the path in the configuration.")
        exit()

    # Process the data using the dynamic function
    X_processed, y_processed, mapping = process_long_format_to_cnn(
        df=input_df,
        signal_cols=SIGNAL_COLUMNS,
        ancillary_cols=ANCILLARY_COLUMNS,
        subject_id_col=SUBJECT_ID_COLUMN,
        label_col=LABEL_COLUMN
    )

    # Verify and report the final shapes
    print("\n--- Preprocessing Complete ---")
    print(f"Final shape of feature tensor (X): {X_processed.shape}")
    print(f"This shape means: ({X_processed.shape[0]} samples, {X_processed.shape[1]} time steps, {X_processed.shape[2]} features)")
    print(f"Final shape of label vector (y): {y_processed.shape}")
    
    # Save the processed arrays to .npy files for easy loading in another script
    X_output_file = f"{OUTPUT_PREFIX}_X.npy"
    y_output_file = f"{OUTPUT_PREFIX}_y.npy"
    
    np.save(X_output_file, X_processed)
    np.save(y_output_file, y_processed)
    
    print(f"\nProcessed data saved to:")
    print(f"Features: {X_output_file}")
    print(f"Labels:   {y_output_file}")
    print(f"Label Mapping: {mapping}")

