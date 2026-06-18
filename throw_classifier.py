import os
import glob
import math
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def calculate_angle(a, b, c):
    """Calculate angle between points A, B, C with vertex at B."""
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = math.degrees(radians)
    angle = angle % 360
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_features(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    
    # Identify the release frame
    release_rows = df[df['released'] == True]
    if len(release_rows) > 0:
        release_idx = release_rows.index[0]
    else:
        # Fallback: if no release frame is marked, use the frame where the shooting wrist is highest 
        # Note: in image coordinates, Y goes down, so highest wrist = minimum Y
        hand = df['handedness'].iloc[-1]
        if hand == 'Right':
            release_idx = df['right_wrist_y'].idxmin()
        else:
            release_idx = df['left_wrist_y'].idxmin()

    if pd.isna(release_idx):
        return None

    release_row = df.loc[release_idx]
    handedness = release_row['handedness']
    if pd.isna(handedness) or handedness == 'Unknown':
        handedness = 'Right' # default fallback
        
    prefix = 'right_' if handedness == 'Right' else 'left_'
    
    # Helper to extract a tuple of (x,y) for a joint
    def get_pt(row, joint):
        return (row[f'{prefix}{joint}_x'], row[f'{prefix}{joint}_y'])
        
    # Feature 1: Angles at the exact moment of release
    try:
        shoulder_pt = get_pt(release_row, 'shoulder')
        elbow_pt = get_pt(release_row, 'elbow')
        wrist_pt = get_pt(release_row, 'wrist')
        hip_pt = get_pt(release_row, 'hip')
        knee_pt = get_pt(release_row, 'knee')
        ankle_pt = get_pt(release_row, 'ankle')
        
        elbow_angle = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)
        shoulder_angle = calculate_angle(hip_pt, shoulder_pt, elbow_pt)
        knee_angle = calculate_angle(hip_pt, knee_pt, ankle_pt)
    except Exception:
        # If landmarks are missing
        elbow_angle, shoulder_angle, knee_angle = 0, 0, 0
        
    # Feature 2: Wrist velocity right before release
    # We look back up to 5 frames before the release
    lookback = max(0, release_idx - 5)
    past_row = df.loc[lookback]
    
    dt = release_row['time_sec'] - past_row['time_sec']
    if dt > 0:
        wrist_vel_x = (release_row[f'{prefix}wrist_x'] - past_row[f'{prefix}wrist_x']) / dt
        wrist_vel_y = (release_row[f'{prefix}wrist_y'] - past_row[f'{prefix}wrist_y']) / dt
    else:
        wrist_vel_x, wrist_vel_y = 0, 0
        
    # Compile features into a dictionary
    features = {
        'elbow_angle_release': elbow_angle,
        'shoulder_angle_release': shoulder_angle,
        'knee_angle_release': knee_angle,
        'wrist_vel_x': wrist_vel_x,
        'wrist_vel_y': wrist_vel_y,
    }
    return features

def build_dataset(data_dir):
    X = []
    y = []
    # Recursively find all CSVs in the data directory
    files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    
    for f in files:
        basename = os.path.basename(f)
        label = None
        if basename.endswith('-h.csv'):
            label = 1 # Hit
        elif basename.endswith('-m.csv'):
            label = 0 # Miss
            
        if label is not None:
            feats = extract_features(f)
            if feats is not None:
                X.append(list(feats.values()))
                y.append(label)
                
    feat_names = list(feats.keys()) if len(X) > 0 else []
    return np.array(X), np.array(y), feat_names

def train(data_dir, model_save_path):
    print(f"Scanning '{data_dir}' for annotated CSVs (ending in '-h.csv' or '-m.csv')...")
    X, y, feat_names = build_dataset(data_dir)
    
    if len(X) == 0:
        print("No annotated data found! Please rename your CSVs to end with '-h.csv' for hits or '-m.csv' for misses.")
        return
        
    print(f"Found {len(X)} annotated throws ({sum(y)} hits, {len(y)-sum(y)} misses).")
    
    # Using Random Forest as it's robust and prevents overfitting on small datasets
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # If we have enough data, perform cross-validation to give the user a performance estimate
    if len(X) >= 5:
        cv_folds = min(5, len(X))
        # Ensure we have at least 1 of each class in folds if possible
        if len(set(y)) > 1:
            scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy')
            print(f"Cross-Validation Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")
        
    clf.fit(X, y)
    
    # Display Feature Importances
    importances = clf.feature_importances_
    print("\nFeature Importances (what matters most for hit/miss):")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
        
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': clf, 'feature_names': feat_names}, f)
        
    print(f"\nModel successfully saved to {model_save_path}")

def predict(csv_path, model_path):
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Please run training first.")
        return
        
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        clf = data['model']
        feat_names = data['feature_names']
        
    feats = extract_features(csv_path)
    if feats is None:
        print(f"Failed to extract features from {csv_path}.")
        return
        
    X = np.array([list(feats.values())])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    
    result = "HIT" if pred == 1 else "MISS"
    confidence = prob[pred] * 100
    
    print(f"\nPrediction for {os.path.basename(csv_path)}: {result} ({confidence:.1f}% confidence)")
    
    print("\nExtracted Biomechanical Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Basketball Throw Classifier (Hit/Miss)")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode to run: 'train' on annotated data, or 'predict' on a single CSV.")
    parser.add_argument('--data-dir', type=str, default='data', help="Directory containing CSVs (for training). Default is 'data'.")
    parser.add_argument('--csv', type=str, help="Path to single CSV (for predicting).")
    parser.add_argument('--model', type=str, default='models/throw_classifier.pkl', help="Path to save/load model. Default is 'models/throw_classifier.pkl'.")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.data_dir, args.model)
    elif args.mode == 'predict':
        if not args.csv:
            print("Error: Please provide a path using --csv for prediction.")
        else:
            predict(args.csv, args.model)
