#!/usr/bin/env python3
import os
import glob
import argparse
import json
import multiprocessing

import numpy as np
import pandas as pd
import laspy
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  # <--- NEW IMPORT
import joblib

def read_las_to_df(path):
    print(f"[READ] Reading LAS file: {os.path.basename(path)}")
    las = laspy.read(path)
    coords = las.xyz
    df = pd.DataFrame(coords, columns=['x','y','z'])
    df['intensity'] = las.intensity.astype(float)
    print(f"[READ]   Loaded {len(df)} points")
    return df

def train_rf_for_location(location,
                          base_dir="/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs/utilities/beach_removal/training_data",
                          output_dir="/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs/utilities/beach_removal/joblib_files",
                          subsample_frac=.6,
                          test_size=0.2):
    print(f"\n[{location}] Starting training pipeline")

    # Determine CPU usage (use 75% of cores)
    total_cpus = multiprocessing.cpu_count()
    n_jobs = max(1, total_cpus // 3)
    print(f"[CONFIG] Using {n_jobs} parallel jobs out of {total_cpus} CPUs")

    # 1) Gather files
    print("[STEP 1] Locating LAS files...")
    beach_files = glob.glob(os.path.join(base_dir, location, "*beach.las"))
    cliff_files = glob.glob(os.path.join(base_dir, location, "*cliff.las"))
    print(f"[STEP 1]   Found {len(beach_files)} beach, {len(cliff_files)} cliff files")
    if not beach_files or not cliff_files:
        raise FileNotFoundError(f"No LAS files found for '{location}' under {base_dir}")

    # 2) Read & label
    print("[STEP 2] Reading and labeling points...")
    dfs = []
    for f in tqdm(beach_files, desc="  beach"):
        df = read_las_to_df(f)
        df['label'] = 1
        dfs.append(df)
    for f in tqdm(cliff_files, desc="  cliff"):
        df = read_las_to_df(f)
        df['label'] = 0
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    print(f"[STEP 2]   Total points before cleaning: {len(data)}")

    # 3) Clean & subsample
    print("[STEP 3] Cleaning and subsampling...")
    data.dropna(subset=['x','y','z','intensity'], inplace=True)
    data = data.sample(frac=subsample_frac, random_state=None)
    print(f"[STEP 3]   After subsample ({int(subsample_frac*100)}%): {len(data)} points")

    # 4) Normalization & Feature Assembly
    print("[STEP 4] Normalizing Intensity and Assembling feature matrix...")
    
    # --- NEW: NORMALIZE INTENSITY ---
    scaler = StandardScaler()
    # Reshape is required for sklearn single-feature scaling
    intensity_reshaped = data['intensity'].values.reshape(-1, 1)
    # Create normalized column
    data['intensity_norm'] = scaler.fit_transform(intensity_reshaped).flatten()
    
    # Use the NORMALIZED intensity for training
    feature_cols = ['x', 'y', 'z', 'intensity_norm']
    
    X = data[feature_cols].values
    y = data['label'].values
    print(f"[STEP 4]   Feature matrix shape: {X.shape}")

    # 5) Split into train/test
    print(f"[STEP 5] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=None
    )
    print(f"[STEP 5]   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 6) Train Random Forest
    print("[STEP 6] Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced',
        oob_score=True,
        n_jobs=n_jobs,
        random_state=None
    )
    clf.fit(X_train, y_train)
    print(f"[STEP 6]   OOB score: {clf.oob_score_:.4f}")

    # 6.5) Save the trained model AND the Scaler
    print("[STEP 6.5] Saving trained model and scaler...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Model
    model_path = os.path.join(output_dir, f"{location}RF.joblib")
    joblib.dump(clf, model_path)
    
    # --- NEW: Save Scaler ---
    scaler_path = os.path.join(output_dir, f"{location}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    print(f"[STEP 6.5]   Model saved to {model_path}")
    print(f"[STEP 6.5]   Scaler saved to {scaler_path}")

    # 7 & 8) Evaluate & report
    try:
        print("[STEP 7] Evaluating on test set...")
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report = classification_report(
            y_test, y_pred,
            labels=[0, 1],
            target_names=['cliff', 'beach'],
            zero_division=0
        )
        print(f"[STEP 7]   Test accuracy: {acc:.4f}")
        print(report)

        print("[STEP 8] Running 3-fold CV (F1-weighted)...")
        cv_scores = cross_val_score(
            clf, X_train, y_train,
            cv=3,
            scoring='f1_weighted',
            n_jobs=n_jobs
        )
        print(f"[STEP 8]   CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        print("[STEP 9] Saving train/test report...")
        report_path = os.path.join(output_dir, f"{location}_training_report.json")
        results = {
            'test_accuracy':   float(acc),
            'oob_score':       float(clf.oob_score_),
            'cv_f1_mean':      float(cv_scores.mean()),
            'cv_f1_std':       float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[STEP 9]   Report saved to {report_path}")

    except Exception as e:
        print(f"[ERROR] Evaluation/reporting failed: {e}")
        print(f"[INFO] The model is safely saved at: {model_path}")

    print(f"[{location}] Training pipeline complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RF on beach vs cliff LiDAR points with intensity normalization"
    )
    parser.add_argument("location", help="Folder name under training_data (e.g. SanElijo)")
    args = parser.parse_args()
    train_rf_for_location(args.location)