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

# --- 2. Functions to Simulate Effects on EEG ---

def apply_caffeine_effect(eeg_signal, caffeine_mg, fs=SAMPLING_RATE):
    """
    Simulates the effect of caffeine based on a continuous dosage (0-400mg).
    The effect scales with the amount of caffeine.
    """
    if caffeine_mg <= 0:
        return eeg_signal
        
    scale_factor = min(caffeine_mg / 400.0, 1.0)
    time = np.arange(len(eeg_signal)) / fs
    beta_boost_freq = 22
    beta_boost_amp = 15 * scale_factor
    beta_boost = beta_boost_amp * np.sin(2 * np.pi * beta_boost_freq * time)
    suppression = 0.15 * scale_factor
    return eeg_signal * (1 - suppression) + beta_boost

def apply_sugar_effect(eeg_signal, sugar_g):
    """
    Simulates a sugar rush/crash based on a continuous amount (0-100g).
    The effect scales with the amount of sugar.
    """
    if sugar_g <= 0:
        return eeg_signal
        
    scale_factor = min(sugar_g / 100.0, 1.0)
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    delta_wave = (12 * scale_factor) * np.sin(2 * np.pi * 3 * time)
    instability_noise = np.random.normal(0, 3 * scale_factor, len(time))
    return eeg_signal + delta_wave + instability_noise

def apply_brain_rot_effect(eeg_signal, degree=0.5):
    """
    Simulates neurodegeneration ('brain rot'). Severe and chronic.
    - Reduces overall signal amplitude/complexity.
    - Introduces very dominant, very slow (delta) waves.
    """
    dampened_signal = eeg_signal * (1 - 0.8 * degree)
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    delta_freq = 1.5 # Very slow wave
    delta_amp = 30 * degree
    delta_wave = delta_amp * np.sin(2 * np.pi * delta_freq * time)
    return dampened_signal + delta_wave

def apply_brain_tumor_effect(eeg_signal):
    """
    Simulates a brain tumor by adding a localized, high-amplitude,
    very slow delta wave, disrupting the normal rhythm.
    """
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    tumor_delta_freq = 1.0
    tumor_delta_amp = 40
    tumor_wave = tumor_delta_amp * np.sin(2 * np.pi * tumor_delta_freq * time)
    return eeg_signal * 0.7 + tumor_wave

def apply_sleep_disorder_effect(eeg_signal):
    """
    Simulates severe sleep deprivation.
    - Increases theta waves (drowsiness).
    - Reduces alpha power (less relaxed wakefulness).
    """
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    theta_boost_freq = 6
    theta_boost_amp = 25
    theta_boost = theta_boost_amp * np.sin(2 * np.pi * theta_boost_freq * time)
    return eeg_signal * 0.6 + theta_boost

def apply_encephalopathy_effect(eeg_signal):
    """
    **NEW CONFUSER CLASS**
    Simulates metabolic encephalopathy. Often reversible.
    - Adds prominent theta/delta waves, similar to Brain Rot but slightly faster.
    - Slightly dampens overall signal.
    """
    dampened_signal = eeg_signal * 0.8 # Moderate dampening
    time = np.arange(len(eeg_signal)) / SAMPLING_RATE
    # Slower waves, but not as slow as the severe 'Brain Rot'
    slow_wave_freq = 2.5 
    slow_wave_amp = 25
    slow_wave = slow_wave_amp * np.sin(2 * np.pi * slow_wave_freq * time)
    return dampened_signal + slow_wave


# --- 3. Main Data Generation Script ---

def generate_eeg_dataset(n_subjects_per_class=N_SUBJECTS_PER_CLASS):
    """
    Generates the full dataset with all classes and habit variations.
    """
    all_subjects_data = []
    subject_id_counter = 0
    
    # Added the new confuser class
    classes = ["Normal", "Brain Rot", "Brain Tumor", "Sleep Disorder", "Metabolic Encephalopathy"]
    
    for class_name in classes:
        print(f"Generating data for class: {class_name}...")
        for i in range(n_subjects_per_class):
            caffeine_mg = np.random.uniform(0, 400)
            sugar_g = np.random.uniform(0, 100)
            
            eeg_signal = generate_baseline_eeg()
            
            rot_degree = 0.0
            if class_name == "Brain Rot":
                rot_degree = np.random.uniform(0.2, 0.9)
                eeg_signal = apply_brain_rot_effect(eeg_signal, degree=rot_degree)
            elif class_name == "Brain Tumor":
                eeg_signal = apply_brain_tumor_effect(eeg_signal)
            elif class_name == "Sleep Disorder":
                eeg_signal = apply_sleep_disorder_effect(eeg_signal)
            elif class_name == "Metabolic Encephalopathy":
                eeg_signal = apply_encephalopathy_effect(eeg_signal)

            eeg_signal = apply_caffeine_effect(eeg_signal, caffeine_mg)
            eeg_signal = apply_sugar_effect(eeg_signal, sugar_g)
                
            subject_df = pd.DataFrame({'time_step': np.arange(N_TIME_STEPS), 'eeg_signal': eeg_signal})
            subject_df['subject_id'] = subject_id_counter
            subject_df['class'] = class_name
            subject_df['caffeine_mg'] = round(caffeine_mg, 2)
            subject_df['sugar_g'] = round(sugar_g, 2)
            subject_df['rot_degree'] = round(rot_degree, 2)
            
            all_subjects_data.append(subject_df)
            subject_id_counter += 1
            
    final_dataset = pd.concat(all_subjects_data, ignore_index=True)
    return final_dataset


# --- Main execution ---
if __name__ == "__main__":
    simulated_eeg_data = generate_eeg_dataset()
    
    print("\n--- Dataset Generation Complete ---")
    print("\nDataset Info:")
    simulated_eeg_data.info()
    
    print("\nValue Counts for Classes:")
    print(simulated_eeg_data.drop_duplicates('subject_id')['class'].value_counts())
    
    output_filename = "outputs/simulated_eeg_multivariate_dataset.csv"
    simulated_eeg_data.to_csv(output_filename, index=False)
    print(f"\nDataset saved to '{output_filename}'")
