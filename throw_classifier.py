import os
import glob
import math
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

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
        
    try:
        shoulder_pt = get_pt(release_row, 'shoulder')
        wrist_pt = get_pt(release_row, 'wrist')
        hip_pt = get_pt(release_row, 'hip')
        knee_pt = get_pt(release_row, 'knee')
    except Exception:
        return None # Discard throw if primary landmarks are missing
    # --- VIEW-INVARIANT FEATURES ---
    # Since cameras can be at any horizontal angle (e.g., Left 45 or Right 45), 
    # horizontal X-coordinates and absolute 2D angles are highly distorted. 
    # However, vertical Y-coordinates (heights) are much more consistent across any level camera!
    
    # 1. Vertical Extension at Release (How high is the wrist above the shoulder?)
    # Since Y goes down in images, a higher wrist means a smaller Y. 
    # Distance = shoulder_y - wrist_y
    vertical_extension = shoulder_pt[1] - wrist_pt[1]
    
    # 2. Leg Vertical Extension (How straight are the legs? hip_y - knee_y)
    leg_extension = knee_pt[1] - hip_pt[1]
    
    # 3. Vertical Dip Depth (Lowest point of the wrist vs. Release point)
    # The lowest physical point means the highest Y value
    max_wrist_y = df[f'{prefix}wrist_y'].max()
    dip_depth = max_wrist_y - release_row[f'{prefix}wrist_y']
    
    # 4. Vertical Velocities leading up to release (We ignore X velocity as it's view-dependent)
    velocities_y = []
    start_idx = max(0, release_idx - 5)
    for i in range(start_idx, release_idx):
        if i not in df.index or (i+1) not in df.index:
            continue
        row1 = df.loc[i]
        row2 = df.loc[i+1]
        dt = row2['time_sec'] - row1['time_sec']
        if dt > 0:
            vy = (row2[f'{prefix}wrist_y'] - row1[f'{prefix}wrist_y']) / dt
            velocities_y.append(vy)

    if len(velocities_y) > 0:
        max_upward_vel_y = min(velocities_y) if min(velocities_y) < 0 else max(velocities_y) 
        mean_vel_y = np.mean(velocities_y)
        
        if len(velocities_y) >= 2:
            dt_total = release_row['time_sec'] - df.loc[start_idx]['time_sec']
            accel_y = (velocities_y[-1] - velocities_y[0]) / dt_total if dt_total > 0 else 0
        else:
            accel_y = 0
    else:
        max_upward_vel_y, mean_vel_y, accel_y = 0, 0, 0
    
    # Compile features into a dictionary (dropping view-dependent angles like elbow_angle)
    features = {
        'vertical_extension_release': vertical_extension,
        'leg_extension_release': leg_extension,
        'dip_depth': dip_depth,
        'max_upward_vel_y': max_upward_vel_y,
        'mean_vel_y': mean_vel_y,
        'accel_y': accel_y
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
    
    # Set up the Random Forest with balanced class weights
    base_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Define the Hyperparameter Grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],      # Number of trees in the forest
        'max_depth': [3, 5, 10, None],       # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],     # Min samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]        # Min samples required to be at a leaf node
    }
    
    print("\nRunning Hyperparameter Tuning (Grid Search)...")
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=5,               # 5-fold cross-validation
        scoring='accuracy', # We are optimizing for overall accuracy
        n_jobs=-1,          # Use all CPU cores to speed up the search
        verbose=1           # Print progress
    )
    
    # Fit the grid search to the data
    grid_search.fit(X, y)
    
    # Extract the absolute best model it found
    clf = grid_search.best_estimator_
    
    print(f"\nBest Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_*100:.2f}%")
    
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
