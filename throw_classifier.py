import os
import glob
import math
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Number of timesteps to resample every throw to (fixed-length representation)
N_TIMESTEPS = 30

# Joints whose Y-trajectory we feed into the model.
# These are the ones most relevant to shooting mechanics.
TRAJECTORY_JOINTS = ['wrist', 'elbow', 'shoulder', 'hip', 'knee']


def extract_features(csv_path):
    """Extract view-invariant time-series features from a single throw CSV.

    Instead of summarizing a throw as 5-6 numbers at the release frame, we
    resample the full vertical (Y) trajectory of key joints into a fixed-length
    vector. This preserves the *shape* of the motion — the wind-up, the peak,
    and the release timing — which single-frame snapshots lose entirely.

    We intentionally ignore X-coordinates because they change drastically
    depending on the camera's horizontal angle.
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

    # --- 1. Resampled Y-trajectories (the bulk of the feature vector) ---
    # Resample each joint's vertical position to N_TIMESTEPS evenly-spaced points.
    # This makes throws of different durations directly comparable.
    x_orig = np.linspace(0, 1, len(df))
    x_new = np.linspace(0, 1, N_TIMESTEPS)

    for joint in TRAJECTORY_JOINTS:
        col = f'{prefix}{joint}_y'
        if col not in df.columns:
            return None
        vals = pd.to_numeric(df[col], errors='coerce').values
        if np.all(np.isnan(vals)):
            return None
        # Forward-fill then back-fill NaNs (from missing detections)
        mask = np.isnan(vals)
        if mask.any():
            valid = np.where(~mask)[0]
            if len(valid) == 0:
                return None
            vals = np.interp(np.arange(len(vals)), valid, vals[valid])

        resampled = np.interp(x_new, x_orig, vals)
        for t in range(N_TIMESTEPS):
            features[f'{joint}_y_t{t:02d}'] = resampled[t]

    # --- 2. Summary statistics over the trajectory ---
    wrist_y = pd.to_numeric(df[f'{prefix}wrist_y'], errors='coerce').values
    mask = ~np.isnan(wrist_y)
    if mask.sum() < 3:
        return None
    wrist_y_clean = wrist_y[mask]

    # Range of motion (total vertical travel of the wrist)
    features['wrist_y_range'] = float(np.ptp(wrist_y_clean))

    # Position of the peak (what fraction of the throw is the wrist at its highest?)
    # In image coords, highest = most negative Y
    peak_idx = np.argmin(wrist_y_clean)
    features['wrist_peak_position'] = peak_idx / len(wrist_y_clean)

    # Smoothness: mean absolute second derivative (jerk proxy)
    if len(wrist_y_clean) >= 3:
        d2 = np.diff(wrist_y_clean, n=2)
        features['wrist_smoothness'] = float(np.mean(np.abs(d2)))
    else:
        features['wrist_smoothness'] = 0.0

    # Velocity at the peak (how fast is the wrist moving at its highest point?)
    if peak_idx > 0 and peak_idx < len(wrist_y_clean) - 1:
        features['wrist_vel_at_peak'] = float(wrist_y_clean[peak_idx + 1] - wrist_y_clean[peak_idx - 1]) / 2.0
    else:
        features['wrist_vel_at_peak'] = 0.0

    # Asymmetry: how different is the upward phase from the downward phase
    if peak_idx > 2 and peak_idx < len(wrist_y_clean) - 2:
        up_len = peak_idx
        down_len = len(wrist_y_clean) - peak_idx - 1
        features['phase_asymmetry'] = up_len / (up_len + down_len)
    else:
        features['phase_asymmetry'] = 0.5

    return features


def build_dataset(data_dir):
    """Scan data_dir for annotated CSVs and build feature matrix + labels."""
    X = []
    y = []
    skipped = 0
    files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)

    feat_names = None
    for f in files:
        basename = os.path.basename(f)
        label = None
        if basename.endswith('-h.csv'):
            label = 1  # Hit
        elif basename.endswith('-m.csv'):
            label = 0  # Miss

        if label is not None:
            feats = extract_features(f)
            if feats is not None:
                if feat_names is None:
                    feat_names = list(feats.keys())
                X.append(list(feats.values()))
                y.append(label)
            else:
                skipped += 1

    if skipped > 0:
        print(f"  (Skipped {skipped} files due to missing/bad data)")

    feat_names = feat_names if feat_names else []
    return np.array(X), np.array(y), feat_names


def train(data_dir, model_save_path):
    print(f"Scanning '{data_dir}' for annotated CSVs (ending in '-h.csv' or '-m.csv')...")
    X, y, feat_names = build_dataset(data_dir)

    if len(X) == 0:
        print("No annotated data found! Rename CSVs to end with '-h.csv' (hit) or '-m.csv' (miss).")
        return

    n_hits = int(sum(y))
    n_misses = len(y) - n_hits
    print(f"Found {len(X)} usable throws ({n_hits} hits, {n_misses} misses), {len(feat_names)} features each.")

    # --- Model candidates ---
    # We try two fundamentally different tree ensemble strategies:
    #
    # 1. Random Forest: builds many independent trees and averages their votes.
    #    Good baseline, resistant to overfitting.
    #
    # 2. Histogram Gradient Boosting: builds trees *sequentially*, where each
    #    new tree specifically targets the mistakes of the previous ones.
    #    Better at extracting weak signals, which is exactly our situation.

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'RandomForest': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
            ]),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [5, 10, None],
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
                'clf__min_samples_leaf': [1, 5, 10],
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, spec in models.items():
        print(f"\n{'='*50}")
        print(f"Tuning {name}...")
        print(f"{'='*50}")

        grid = GridSearchCV(
            estimator=spec['pipeline'],
            param_grid=spec['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X, y)

        print(f"  Best CV Accuracy: {grid.best_score_ * 100:.2f}%")
        print(f"  Best Params: {grid.best_params_}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    print(f"\n{'='*50}")
    print(f"WINNER: {best_name} — {best_score * 100:.2f}% accuracy")
    print(f"{'='*50}")

    # Feature importances (only available for tree-based models)
    clf_step = best_model.named_steps['clf']
    if hasattr(clf_step, 'feature_importances_'):
        importances = clf_step.feature_importances_
        print("\nTop 15 Feature Importances:")
        ranked = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        for name_f, imp in ranked[:15]:
            bar = '█' * int(imp * 200)
            print(f"  {name_f:30s} {imp:.4f} {bar}")

    # Save
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': best_model, 'feature_names': feat_names}, f)
    print(f"\nModel saved to {model_save_path}")


def predict(csv_path, model_path):
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Run training first.")
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

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.data_dir, args.model)
    elif args.mode == 'predict':
        if not args.csv:
            print("Error: provide --csv for prediction mode.")
        else:
            predict(args.csv, args.model)
