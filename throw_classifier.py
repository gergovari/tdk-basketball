import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Number of timesteps to resample every throw to
N_TIMESTEPS = 30

# Joints whose Y-trajectory we track
TRAJECTORY_JOINTS = ['wrist', 'elbow', 'shoulder', 'hip', 'knee']
MP_TRAJECTORY_JOINTS = ['wrist', 'elbow', 'shoulder', 'hip', 'knee', 'index', 'thumb', 'heel', 'foot_index']


def _resample(values, n=N_TIMESTEPS):
    """Resample a 1D array to exactly n points via linear interpolation."""
    x_orig = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_orig, values)


def _clean_series(series):
    """Convert to float, interpolate NaNs. Returns None if all NaN."""
    vals = pd.to_numeric(series, errors='coerce').values
    if np.all(np.isnan(vals)):
        return None
    mask = np.isnan(vals)
    if mask.any():
        valid = np.where(~mask)[0]
        if len(valid) == 0:
            return None
        vals = np.interp(np.arange(len(vals)), valid, vals[valid])
    return vals


def _extract_group_id(filepath):
    """Extract a group ID so the same physical throw from different cameras
    is always kept together during cross-validation.

    Filename pattern: {player}-angle{N}-{throw}-{label}.csv
    Path pattern:     data/{session}/{run}/throw/{filename}

    Group ID:         {session}/{run}/{player}-{throw}
    """
    parts = filepath.replace('\\', '/').split('/')
    basename = parts[-1]

    # Try to find session/run from directory structure
    # Expected: .../data/{session}/{run}/throw/{file}
    try:
        throw_idx = parts.index('throw')
        session = parts[throw_idx - 2]
        run = parts[throw_idx - 1]
    except (ValueError, IndexError):
        session = 'unknown'
        run = '0'

    m = re.match(r'(\d+)-angle\d+-(\d+)(?:-mp)?-[hm]\.csv', basename)
    if m:
        player, throw_num = m.group(1), m.group(2)
        return f'{session}/{run}/{player}-{throw_num}'

    # Fallback: use filename as unique group (no grouping)
    return basename


def extract_features(csv_path, is_mediapipe=False):
    """Extract view-invariant time-series features from a throw CSV.

    This is the simple feature set that achieved our best accuracy (68%):
      - Resampled Y-position trajectories for 5 joints (5 × 30 = 150)
      - Targeted summary statistics (8)
    """
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        return None

    # Determine shooting hand
    hand = df['handedness'].iloc[-1]
    if pd.isna(hand) or hand == 'Unknown':
        hand = 'Right'
    prefix = 'right_' if hand == 'Right' else 'left_'

    features = {}

    joints_to_track = MP_TRAJECTORY_JOINTS if is_mediapipe else TRAJECTORY_JOINTS

    # --- Pre-extract all Y series ---
    joint_series = {}
    for joint in joints_to_track:
        col = f'{prefix}{joint}_y'
        if col not in df.columns:
            return None
        cleaned = _clean_series(df[col])
        if cleaned is None:
            return None
        joint_series[joint] = cleaned

    # --- 1. Resampled Y-position trajectories (150 features) ---
    for joint, vals in joint_series.items():
        resampled = _resample(vals)
        for t in range(N_TIMESTEPS):
            features[f'{joint}_y_t{t:02d}'] = resampled[t]

    # --- 2. Summary statistics ---
    wrist_y = joint_series['wrist']

    # Range of motion
    features['wrist_y_range'] = float(np.ptp(wrist_y))

    # Peak position (fraction of throw where wrist is highest)
    peak_idx = np.argmin(wrist_y)  # most negative Y = highest in image coords
    features['wrist_peak_position'] = peak_idx / len(wrist_y)

    # Smoothness (mean absolute second derivative)
    if len(wrist_y) >= 3:
        d2 = np.diff(wrist_y, n=2)
        features['wrist_smoothness'] = float(np.mean(np.abs(d2)))
    else:
        features['wrist_smoothness'] = 0.0

    # Velocity at peak
    if 0 < peak_idx < len(wrist_y) - 1:
        features['wrist_vel_at_peak'] = float(wrist_y[peak_idx + 1] - wrist_y[peak_idx - 1]) / 2.0
    else:
        features['wrist_vel_at_peak'] = 0.0

    # Phase asymmetry (upswing vs downswing duration)
    if 2 < peak_idx < len(wrist_y) - 2:
        features['phase_asymmetry'] = peak_idx / len(wrist_y)
    else:
        features['phase_asymmetry'] = 0.5

    if is_mediapipe:
        index_y = joint_series.get('index')
        heel_y = joint_series.get('heel')
        foot_index_y = joint_series.get('foot_index')
        
        if index_y is not None and heel_y is not None and foot_index_y is not None:
            # Flick: vertical difference between index finger and wrist at peak wrist height
            features['index_wrist_diff_at_peak'] = float(index_y[peak_idx] - wrist_y[peak_idx])
            
            # Index finger range of motion
            features['index_y_range'] = float(np.ptp(index_y))
            
            # Calf raise: difference between foot_index (toes) and heel.
            # A larger positive difference means the heel is significantly higher (Y is smaller) than the toes.
            features['calf_raise_max'] = float(np.max(foot_index_y - heel_y))

    return features


def build_dataset(data_dir, is_mediapipe=False):
    """Scan data_dir for annotated CSVs and build feature matrix + labels + groups."""
    X = []
    y = []
    groups = []
    skipped = 0
    files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)

    feat_names = None
    for f in files:
        basename = os.path.basename(f)
        label = None
        
        if is_mediapipe:
            if basename.endswith('-mp-h.csv'):
                label = 1
            elif basename.endswith('-mp-m.csv'):
                label = 0
        else:
            if basename.endswith('-h.csv') and not basename.endswith('-mp-h.csv'):
                label = 1
            elif basename.endswith('-m.csv') and not basename.endswith('-mp-m.csv'):
                label = 0

        if label is not None:
            feats = extract_features(f, is_mediapipe)
            if feats is not None:
                if feat_names is None:
                    feat_names = list(feats.keys())
                X.append(list(feats.values()))
                y.append(label)
                groups.append(_extract_group_id(f))
            else:
                skipped += 1

    if skipped > 0:
        print(f"  (Skipped {skipped} files due to missing/bad data)")

    feat_names = feat_names if feat_names else []
    return np.array(X), np.array(y), groups, feat_names


def train(data_dir, model_save_path, is_mediapipe=False):
    mp_str = "'-mp-h.csv' or '-mp-m.csv'" if is_mediapipe else "'-h.csv' or '-m.csv'"
    print(f"Scanning '{data_dir}' for annotated CSVs (ending in {mp_str})...")
    X, y, groups, feat_names = build_dataset(data_dir, is_mediapipe)

    if len(X) == 0:
        print("No annotated data found! Rename CSVs to end with '-h.csv' (hit) or '-m.csv' (miss).")
        return

    n_hits = int(sum(y))
    n_misses = len(y) - n_hits
    unique_groups = set(groups)
    print(f"Found {len(X)} samples from {len(unique_groups)} unique physical throws "
          f"({n_hits} hits, {n_misses} misses), {len(feat_names)} features each.")

    # Encode group strings as integers for sklearn
    group_to_id = {g: i for i, g in enumerate(sorted(unique_groups))}
    group_ids = np.array([group_to_id[g] for g in groups])

    # --- Grouped CV ---
    # StratifiedGroupKFold ensures:
    #   1. Both camera angles of the same throw are ALWAYS in the same fold
    #      (prevents data leakage from seeing the same throw in train+test)
    #   2. Hit/miss ratio is preserved in each fold (stratified)
    cv_grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nUsing StratifiedGroupKFold (5 folds, grouped by physical throw)")
    print(f"  → Both camera angles of the same throw always stay together")
    print(f"  → No data leakage between train and test\n")

    models = {
        'RandomForest': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
            ]),
            'params': {
                'clf__n_estimators': [100, 200, 400],
                'clf__max_depth': [3, 5, 10, None],
                'clf__min_samples_leaf': [1, 2, 4],
            }
        },
        'GradientBoosting': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', HistGradientBoostingClassifier(
                    class_weight='balanced', random_state=42,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=10
                ))
            ]),
            'params': {
                'clf__max_depth': [3, 5, 7],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_iter': [100, 200, 400],
            }
        },
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, spec in models.items():
        print(f"{'='*50}")
        print(f"Tuning {name}...")
        print(f"{'='*50}")

        grid = GridSearchCV(
            estimator=spec['pipeline'],
            param_grid=spec['params'],
            cv=cv_grouped,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X, y, groups=group_ids)

        print(f"  Best CV Accuracy: {grid.best_score_ * 100:.2f}%")
        print(f"  Best Params: {grid.best_params_}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    print(f"\n{'='*50}")
    print(f"WINNER: {best_name} — {best_score * 100:.2f}% accuracy (grouped CV, no leakage)")
    print(f"{'='*50}")

    # Feature importances
    clf_step = best_model.named_steps['clf']
    print("\nCalculating Feature Importances...")
    if hasattr(clf_step, 'feature_importances_'):
        importances = clf_step.feature_importances_
    else:
        # For HistGradientBoosting, calculate permutation importance
        # We calculate it on the training set to see what the model relied on
        # (Though calculating on a held-out test set is theoretically better, 
        # doing it on train is fine to just peek into the model's logic here)
        result = permutation_importance(best_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    # Ensure importances are normalized (0 to 1) for consistent display
    if np.sum(importances) > 0:
        importances = importances / np.sum(importances)

    print("\nTop 15 Feature Importances:")
    ranked = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    for name_f, imp in ranked[:15]:
        bar = '█' * int(imp * 200)
        print(f"  {name_f:30s} {imp:.4f} {bar}")

    # Fit the winner on ALL data for deployment
    best_model.fit(X, y)

    # Save
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': best_model, 'feature_names': feat_names}, f)
    print(f"\nModel saved to {model_save_path}")


def predict(csv_path, model_path, is_mediapipe=False):
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Run training first.")
        return

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        clf = data['model']

    feats = extract_features(csv_path, is_mediapipe)
    if feats is None:
        print(f"Failed to extract features from {csv_path}.")
        return

    X = np.array([list(feats.values())])
    pred = clf.predict(X)[0]

    if hasattr(clf, 'predict_proba'):
        prob = clf.predict_proba(X)[0]
        confidence = prob[pred] * 100
        conf_str = f" ({confidence:.1f}% confidence)"
    else:
        conf_str = ""

    result = "HIT" if pred == 1 else "MISS"
    print(f"\nPrediction for {os.path.basename(csv_path)}: {result}{conf_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Basketball Throw Classifier (Hit/Miss)")
    parser.add_argument('mode', choices=['train', 'predict'],
                        help="'train' on annotated data, or 'predict' on a single CSV.")
    parser.add_argument('--data-dir', type=str, default='data',
                        help="Directory containing CSVs (for training).")
    parser.add_argument('--csv', type=str,
                        help="Path to a single CSV (for prediction).")
    parser.add_argument('--model', type=str, default='models/throw_classifier.pkl',
                        help="Path to save/load the model.")
    parser.add_argument('--mediapipe', action='store_true',
                        help="Use MediaPipe landmarks and look for -mp-[hm].csv files.")

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.data_dir, args.model, args.mediapipe)
    elif args.mode == 'predict':
        if not args.csv:
            print("Error: provide --csv for prediction mode.")
        else:
            predict(args.csv, args.model, args.mediapipe)
