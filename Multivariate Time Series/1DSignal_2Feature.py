import numpy as np
import pandas as pd

# --- Simulation Parameters ---
SAMPLING_RATE = 256  # Hz
DURATION = 10  # seconds
N_TIME_STEPS = SAMPLING_RATE * DURATION
N_SUBJECTS_PER_CLASS = 50 # Number of subjects to generate for each class

# --- 1. Baseline EEG Signal Generation ---
def generate_baseline_eeg(n_steps=N_TIME_STEPS, fs=SAMPLING_RATE):
    """
    Generates a baseline 'normal' EEG signal by combining several sine waves
    representing different brainwave frequencies.
    """
    time = np.arange(n_steps) / fs
    
    # Alpha waves (8-12 Hz) - prominent in relaxed, awake states
    alpha_freq = 10
    alpha_amp = 15
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * time)
    
    # Beta waves (13-30 Hz) - prominent in active, alert states
    beta_freq = 20
    beta_amp = 10
    beta = beta_amp * np.sin(2 * np.pi * beta_freq * time)

    # Theta waves (4-7 Hz) - associated with drowsiness/light sleep
    theta_freq = 5
    theta_amp = 5
    theta = theta_amp * np.sin(2 * np.pi * theta_freq * time)

    # Add some random noise to make it look more realistic
    noise = np.random.normal(0, 2, len(time))
    
    return alpha + beta + theta + noise

# --- 2. Functions to Simulate Effects on EEG (with continuous scaling) ---

def apply_caffeine_effect(eeg_signal, caffeine_mg, fs=SAMPLING_RATE):
    """
    Simulates the effect of caffeine based on a continuous dosage (0-400mg).
    The effect scales with the amount of caffeine.
    """
    if caffeine_mg <= 0:
        return eeg_signal
        
    # Normalize caffeine amount to a 0-1 scale for effect strength
    scale_factor = min(caffeine_mg / 400.0, 1.0)
    
    time = np.arange(len(eeg_signal)) / fs
    # Boost beta waves (alertness) proportional to caffeine intake
    beta_boost_freq = 22
    beta_boost_amp = 15 * scale_factor # Stronger beta amplitude with more caffeine
    beta_boost = beta_boost_amp * np.sin(2 * np.pi * beta_boost_freq * time)
    
    # Slightly suppress original signal (alpha suppression) proportional to intake
    suppression = 0.15 * scale_factor # Max suppression of 15% at 400mg
    return eeg_signal * (1 - suppression) + beta_boost

def apply_sugar_effect(eeg_signal, sugar_g):
    """
    Simulates a sugar rush/crash based on a continuous amount (0-100g).
    The effect scales with the amount of sugar.
    """
    if sugar_g <= 0:
        return eeg_signal
        
    # Normalize sugar amount to a 0-1 scale for effect strength
    scale_factor = min(sugar_g / 100.0, 1.0)
    
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    # Add slow-wave theta/delta activity proportional to sugar intake
    delta_freq = 3
    delta_amp = 12 * scale_factor
    delta_wave = delta_amp * np.sin(2 * np.pi * delta_freq * time)
    
    # Add high-frequency noise for 'instability' proportional to sugar intake
    instability_noise = np.random.normal(0, 3 * scale_factor, len(time))
    
    return eeg_signal + delta_wave + instability_noise

def apply_brain_rot_effect(eeg_signal, degree=0.5):
    """
    Simulates neurodegeneration ('brain rot').
    - Reduces overall signal amplitude/complexity.
    - Introduces more dominant slow (delta) waves.
    - 'degree' is a float from 0 (mild) to 1 (severe).
    """
    # Reduce overall amplitude based on degree
    dampened_signal = eeg_signal * (1 - 0.8 * degree)
    
    # Add prominent slow delta waves, scaled by degree
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    delta_freq = 1.5
    delta_amp = 30 * degree  # More rot = more delta
    delta_wave = delta_amp * np.sin(2 * np.pi * delta_freq * time)
    
    return dampened_signal + delta_wave

def apply_brain_tumor_effect(eeg_signal):
    """
    Simulates a brain tumor by adding a localized, high-amplitude,
    very slow delta wave, disrupting the normal rhythm.
    """
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    # Tumors often associated with focal, high-power delta activity
    tumor_delta_freq = 1.0
    tumor_delta_amp = 40 # Very high amplitude
    tumor_wave = tumor_delta_amp * np.sin(2 * np.pi * tumor_delta_freq * time)
    
    return eeg_signal * 0.7 + tumor_wave # Dampen normal activity and add tumor signal

def apply_sleep_disorder_effect(eeg_signal):
    """
    Simulates severe sleep deprivation.
    - Increases theta waves (drowsiness).
    - Reduces alpha power (less relaxed wakefulness).
    """
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    # Boost theta waves (drowsiness)
    theta_boost_freq = 6
    theta_boost_amp = 25
    theta_boost = theta_boost_amp * np.sin(2 * np.pi * theta_boost_freq * time)
    
    # Dampen the original signal to represent reduced alpha/beta power
    return eeg_signal * 0.6 + theta_boost

# --- 3. Main Data Generation Script ---

def generate_eeg_dataset(n_subjects_per_class=N_SUBJECTS_PER_CLASS):
    """
    Generates the full dataset with all classes and habit variations.
    """
    all_subjects_data = []
    subject_id_counter = 0
    
    classes = ["Normal", "Brain Rot", "Brain Tumor", "Sleep Disorder"]
    
    for class_name in classes:
        print(f"Generating data for class: {class_name}...")
        for i in range(n_subjects_per_class):
            # Assign continuous values for caffeine and sugar intake
            caffeine_mg = np.random.uniform(0, 400)
            sugar_g = np.random.uniform(0, 100)
            
            # --- Generate and modify signal based on class and habits ---
            eeg_signal = generate_baseline_eeg()
            
            # Apply class-specific effects
            rot_degree = 0.0
            if class_name == "Brain Rot":
                rot_degree = np.random.uniform(0.2, 0.9) # Assign random severity
                eeg_signal = apply_brain_rot_effect(eeg_signal, degree=rot_degree)
            elif class_name == "Brain Tumor":
                eeg_signal = apply_brain_tumor_effect(eeg_signal)
            elif class_name == "Sleep Disorder":
                eeg_signal = apply_sleep_disorder_effect(eeg_signal)
            
            # Apply habit-specific effects on top of the class signal
            eeg_signal = apply_caffeine_effect(eeg_signal, caffeine_mg)
            eeg_signal = apply_sugar_effect(eeg_signal, sugar_g)
                
            # --- Structure the data for this subject ---
            # Create the multivariate time series structure
            subject_df = pd.DataFrame({
                'time_step': np.arange(N_TIME_STEPS),
                'eeg_signal': eeg_signal
            })
            
            # Add the ancillary features, repeated for each time step
            subject_df['subject_id'] = subject_id_counter
            subject_df['class'] = class_name
            subject_df['caffeine_mg'] = round(caffeine_mg, 2)
            subject_df['sugar_g'] = round(sugar_g, 2)
            subject_df['rot_degree'] = round(rot_degree, 2) # Include the severity metric
            
            all_subjects_data.append(subject_df)
            subject_id_counter += 1
            
    # Combine all individual subject dataframes into one large dataframe
    final_dataset = pd.concat(all_subjects_data, ignore_index=True)
    return final_dataset


# --- Main execution ---
if __name__ == "__main__":
    # Generate the dataset
    simulated_eeg_data = generate_eeg_dataset()
    
    # Display the first few rows and info about the dataset
    print("\n--- Dataset Generation Complete ---")
    print("\nDataset Head:")
    print(simulated_eeg_data.head(10))
    
    print("\nDataset Tail (to show transition to next subject):")
    print(simulated_eeg_data.tail(10))
    
    print("\nDataset Info:")
    simulated_eeg_data.info()
    
    print("\nValue Counts for Classes:")
    print(simulated_eeg_data.drop_duplicates('subject_id')['class'].value_counts())
    
    print("\nDescriptive Statistics for Continuous Features:")
    print(simulated_eeg_data.drop_duplicates('subject_id')[['caffeine_mg', 'sugar_g', 'rot_degree']].describe())
    
    # Optional: Save the dataset to a CSV file
    simulated_eeg_data.to_csv("outputs/simulated_eeg_multivariate_dataset.csv", index=False)
    print("\nDataset saved to 'simulated_eeg_multivariate_dataset.csv'")

