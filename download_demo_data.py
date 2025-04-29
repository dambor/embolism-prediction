#!/usr/bin/env python
"""
Download Demo Data for Embolism Detection Model

This script generates synthetic demo data for the embolism detection model.
The data mimics the structure of MIMIC-IV but does not contain real patient data.

Usage:
    python download_demo_data.py [--output_dir DATA_DIR]
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download demo data for embolism detection')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save the demo data')
    return parser.parse_args()


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


def generate_synthetic_data(output_dir):
    """Generate synthetic data that mimics MIMIC-IV structure"""
    np.random.seed(42)  # For reproducibility
    
    # Generate patient cohort data
    generate_cohort_data(output_dir)
    
    # Generate preprocessed multimodal data
    generate_preprocessed_data(output_dir)
    
    print("Synthetic data generation complete")


def generate_cohort_data(output_dir):
    """Generate synthetic cohort data"""
    # Number of patients
    n_patients = 1000
    
    # Generate subject IDs
    subject_ids = np.arange(10000, 10000 + n_patients)
    
    # Generate hospital admission IDs
    hadm_ids = np.arange(20000, 20000 + n_patients)
    
    # Generate ICU stay IDs
    stay_ids = np.arange(30000, 30000 + n_patients)
    
    # Generate demographics
    genders = np.random.choice(['M', 'F'], size=n_patients)
    ages = np.random.normal(65, 15, size=n_patients).astype(int)
    ages = np.clip(ages, 18, 100)  # Clip to realistic age range
    
    # Set random embolism status (10% prevalence)
    # 0: no embolism, 1: PE, 2: DVT
    embolism_status = np.zeros(n_patients, dtype=int)
    embolism_idx = np.random.choice(n_patients, size=int(n_patients * 0.1), replace=False)
    embolism_status[embolism_idx] = np.random.choice([1, 2], size=len(embolism_idx))
    
    # Create cohort dataframe
    cohort = pd.DataFrame({
        'subject_id': subject_ids,
        'hadm_id': hadm_ids,
        'stay_id': stay_ids,
        'gender': genders,
        'age': ages,
        'embolism_type': embolism_status,
        'anchor_year_group': np.random.choice([2020, 2021, 2022, 2023], size=n_patients)
    })
    
    # Save to CSV
    cohort_path = os.path.join(output_dir, 'cohort.csv')
    cohort.to_csv(cohort_path, index=False)
    print(f"Created cohort data with {n_patients} patients, saved to {cohort_path}")


def generate_preprocessed_data(output_dir):
    """Generate synthetic preprocessed data"""
    # Load cohort data
    cohort_path = os.path.join(output_dir, 'cohort.csv')
    cohort = pd.read_csv(cohort_path)
    
    # Create a list to store all patient data
    all_patient_data = []
    
    # Process each patient
    for _, row in cohort.iterrows():
        subject_id = row['subject_id']
        hadm_id = row['hadm_id']
        stay_id = row['stay_id']
        embolism_type = row['embolism_type']
        age = row['age']
        gender = row['gender']
        
        # Determine number of days of data
        n_days = np.random.randint(3, 10)  # 3-9 days of data
        
        # Create timestamps for each day
        start_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # For each day, create multiple measurements
        for day_idx, date in enumerate(dates):
            # Create 4-8 measurements per day
            n_measurements = np.random.randint(4, 9)
            times = sorted([date + timedelta(hours=np.random.randint(0, 24)) for _ in range(n_measurements)])
            
            for time_idx, timestamp in enumerate(times):
                # Generate vital signs data
                is_embolism_patient = embolism_type > 0
                heart_rate = generate_vital_sign('heart_rate', is_embolism_patient, day_idx, n_days)
                sbp = generate_vital_sign('sbp', is_embolism_patient, day_idx, n_days)
                dbp = generate_vital_sign('dbp', is_embolism_patient, day_idx, n_days)
                mbp = (sbp + 2*dbp) / 3
                resp_rate = generate_vital_sign('resp_rate', is_embolism_patient, day_idx, n_days)
                temp = generate_vital_sign('temp', is_embolism_patient, day_idx, n_days)
                spo2 = generate_vital_sign('spo2', is_embolism_patient, day_idx, n_days)
                
                # Generate lab values
                # Some lab values aren't measured at every time point
                include_labs = np.random.random() < 0.7  # 70% chance of including labs
                
                troponin_i = generate_lab_value('troponin_i', is_embolism_patient, day_idx, n_days) if include_labs else None
                ddimer = generate_lab_value('ddimer', is_embolism_patient, day_idx, n_days) if include_labs else None
                inr = generate_lab_value('inr', is_embolism_patient, day_idx, n_days) if include_labs else None
                wbc = generate_lab_value('wbc', is_embolism_patient, day_idx, n_days) if include_labs else None
                platelets = generate_lab_value('platelets', is_embolism_patient, day_idx, n_days) if include_labs else None
                
                # Generate clinical note features
                include_note = np.random.random() < 0.3  # 30% chance of having a note at this time point
                
                if include_note:
                    note_features = generate_note_features(is_embolism_patient, day_idx, n_days)
                else:
                    note_features = {
                        'contains_shortness_of_breath': 0,
                        'contains_chest_pain': 0,
                        'contains_leg_pain': 0,
                        'contains_leg_swelling': 0
                    }
                
                # Generate medication features
                include_meds = np.random.random() < 0.5  # 50% chance of having medication data
                
                if include_meds:
                    med_features = generate_medication_features(is_embolism_patient, day_idx, n_days)
                else:
                    med_features = {
                        'med_heparin': 0,
                        'med_enoxaparin': 0,
                        'med_warfarin': 0
                    }
                
                # Create a row of data
                patient_data = {
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'stay_id': stay_id,
                    'charttime': timestamp,
                    'chartdate': timestamp.date(),
                    'heart_rate': heart_rate,
                    'sbp': sbp,
                    'dbp': dbp,
                    'mbp': mbp,
                    'resp_rate': resp_rate,
                    'temp': temp,
                    'spo2': spo2,
                    'troponin_i': troponin_i,
                    'ddimer': ddimer,
                    'inr': inr,
                    'wbc': wbc,
                    'platelets': platelets,
                    **note_features,
                    **med_features,
                    'age': age,
                    'gender_numeric': 1 if gender == 'M' else 0
                }
                
                all_patient_data.append(patient_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_patient_data)
    
    # Sort by patient and time
    df = df.sort_values(['subject_id', 'hadm_id', 'charttime'])
    
    # Save to CSV
    preprocessed_path = os.path.join(output_dir, 'preprocessed_data.csv')
    df.to_csv(preprocessed_path, index=False)
    print(f"Created preprocessed data with {len(df)} rows, saved to {preprocessed_path}")


def generate_vital_sign(vital_name, is_embolism_patient, day_idx, n_days):
    """Generate synthetic vital signs based on condition"""
    # Base distributions for vitals
    vital_ranges = {
        'heart_rate': (60, 100, 10),  # mean, upper bound for normal, std
        'sbp': (120, 140, 15),
        'dbp': (80, 90, 10),
        'resp_rate': (14, 20, 3),
        'temp': (36.6, 37.2, 0.3),  # Celsius
        'spo2': (97, 100, 1)  # Percentage
    }
    
    mean, upper_normal, std = vital_ranges[vital_name]
    
    if is_embolism_patient:
        # For embolism patients, modify vitals to show deterioration as we approach diagnosis
        progression = min(1.0, day_idx / (n_days - 2)) if n_days > 2 else 0.5
        
        if vital_name == 'heart_rate':
            # Increased heart rate
            mean += 20 * progression
            std += 5 * progression
        elif vital_name == 'resp_rate':
            # Increased respiratory rate
            mean += 6 * progression
            std += 2 * progression
        elif vital_name == 'spo2':
            # Decreased oxygen saturation
            mean -= 3 * progression
            std += 2 * progression
    
    # Add some random variation
    value = np.random.normal(mean, std)
    
    # Ensure realistic ranges
    if vital_name == 'heart_rate':
        value = np.clip(value, 40, 180)
    elif vital_name == 'sbp':
        value = np.clip(value, 80, 200)
    elif vital_name == 'dbp':
        value = np.clip(value, 40, 120)
    elif vital_name == 'resp_rate':
        value = np.clip(value, 8, 40)
    elif vital_name == 'temp':
        value = np.clip(value, 35, 40)
    elif vital_name == 'spo2':
        value = np.clip(value, 70, 100)
    
    return value


def generate_lab_value(lab_name, is_embolism_patient, day_idx, n_days):
    """Generate synthetic lab values based on condition"""
    # Base distributions for labs
    lab_ranges = {
        'troponin_i': (0.01, 0.04, 0.01),  # mean, upper normal, std
        'ddimer': (200, 500, 100),  # ng/mL
        'inr': (1.1, 1.2, 0.1),
        'wbc': (7, 10, 2),  # x10^9/L
        'platelets': (250, 400, 50)  # x10^9/L
    }
    
    mean, upper_normal, std = lab_ranges[lab_name]
    
    if is_embolism_patient:
        # For embolism patients, modify labs to show deterioration as we approach diagnosis
        progression = min(1.0, day_idx / (n_days - 2)) if n_days > 2 else 0.5
        
        if lab_name == 'ddimer':
            # Significantly elevated D-dimer in embolism
            mean += 700 * progression
            std += 300 * progression
        elif lab_name == 'troponin_i':
            # Slightly elevated troponin in PE
            mean += upper_normal * 0.5 * progression
            std += upper_normal * 0.2 * progression
        elif lab_name == 'platelets':
            # Potentially decreased platelets
            mean -= 50 * progression
    
    # Add some random variation
    value = np.random.normal(mean, std)
    
    # Ensure realistic ranges (non-negative for all labs)
    value = max(0, value)
    
    # Additional specific constraints
    if lab_name == 'inr':
        value = max(0.8, value)
    
    return value


def generate_note_features(is_embolism_patient, day_idx, n_days):
    """Generate synthetic clinical note features"""
    features = {}
    
    # Base probabilities for symptoms
    base_probs = {
        'contains_shortness_of_breath': 0.05,
        'contains_chest_pain': 0.05,
        'contains_leg_pain': 0.02,
        'contains_leg_swelling': 0.02
    }
    
    if is_embolism_patient:
        # Increase probabilities as we approach diagnosis
        progression = min(1.0, day_idx / (n_days - 2)) if n_days > 2 else 0.5
        
        # Modify probabilities based on progression
        prob_modifiers = {
            'contains_shortness_of_breath': 0.5 * progression,
            'contains_chest_pain': 0.4 * progression,
            'contains_leg_pain': 0.3 * progression,
            'contains_leg_swelling': 0.3 * progression
        }
        
        # Apply modifiers
        for feature, base_prob in base_probs.items():
            modified_prob = base_prob + prob_modifiers.get(feature, 0)
            features[feature] = 1 if np.random.random() < modified_prob else 0
    else:
        # For non-embolism patients, use base probabilities
        for feature, base_prob in base_probs.items():
            features[feature] = 1 if np.random.random() < base_prob else 0
    
    return features


def generate_medication_features(is_embolism_patient, day_idx, n_days):
    """Generate synthetic medication features"""
    features = {}
    
    # Base probabilities for medications
    base_probs = {
        'med_heparin': 0.05,
        'med_enoxaparin': 0.05,
        'med_warfarin': 0.03
    }
    
    if is_embolism_patient:
        # Increase probabilities as we approach diagnosis
        progression = min(1.0, day_idx / (n_days - 2)) if n_days > 2 else 0.5
        
        # Only increase medication probabilities close to diagnosis
        if progression > 0.7:
            # Modify probabilities based on progression
            prob_modifiers = {
                'med_heparin': 0.3,
                'med_enoxaparin': 0.3,
                'med_warfarin': 0.1
            }
            
            # Apply modifiers
            for feature, base_prob in base_probs.items():
                modified_prob = base_prob + prob_modifiers.get(feature, 0)
                features[feature] = 1 if np.random.random() < modified_prob else 0
        else:
            for feature, base_prob in base_probs.items():
                features[feature] = 1 if np.random.random() < base_prob else 0
    else:
        # For non-embolism patients, use base probabilities
        for feature, base_prob in base_probs.items():
            features[feature] = 1 if np.random.random() < base_prob else 0
    
    return features


def main():
    """Main function"""
    args = parse_arguments()
    create_output_directory(args.output_dir)
    generate_synthetic_data(args.output_dir)


if __name__ == "__main__":
    main()