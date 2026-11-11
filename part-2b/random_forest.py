"""
Usage:
    python random_forest.py

Input:
    make_windows.py (functions + windows.npz file)
    model_utils.py (loading, scoring and saving that is shared with other models)

Outputs:
    models/random_forest.joblib - trained random forest regressor model
    results/rf_predictions.csv - actual vs predicted flow for each validation and test window
    results/rf_metrics.csv - RMSE, MAE, MSE and R2 and a 'previous reading' baseline
    results/rf_site_metrics.csv - test errors of each site
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model_utils import PERIODS, load_data, evaluate, save_results

MODEL_FILE = Path("models/random_forest.joblib")

# Random forest settings
N_TREES = 100
MAX_DEPTH = None # Lets the tree grow until it cannot split anymore, a limit is still set by min_samples_leaf
MIN_SAMPLES_LEAF = 5 # Each leaf needs to hold at least 5 training windows
MAX_FEATURES = 1.0 # Every split considers 100% of the features
RANDOM_STATE = 42 # Fix the seed so its not random on every run

# Optional features
USE_TIME_FEATURES = True # Adds time of day (sin & cos) and day of week as inputs
USE_SITE = True # Tracks site the window is from (one column per site)

# Building the input table for windows, will return table and name of each column
# X - scaled past readings with shape, site_idx - site the window is from, sites - the list of all sites
def features_from_file(X, timestamps, site_idx, sites):
    names = [f"t-{X.shape[1] - i}" for i in range(X.shape[1])] # Labels the past readings as t-12 to t-1 (last reading before current prediction)
    columns = [X]
    if USE_TIME_FEATURES:
        ts = pd.to_datetime(timestamps) # Convert timestamp to datetime format (so that day of week can be obtained)
        fraction = (ts.hour * 60 + ts.minute) / 1440 # How far through the day the time is
        # Sine and cosine used for time so it is a 'circle' - 23:45 and 0:00 are close together (day of week is 0 for Monday, 6 for Sunday)
        columns.append(np.column_stack([np.sin(2 * np.pi * fraction), np.cos(2 * np.pi * fraction), ts.day_of_week]))
        names += ["time_sin", "time_cos", "day_of_week"]
    if USE_SITE: # One-hot encoding to keep track of site the window is from
        columns.append(np.eye(len(sites), dtype=np.float32)[site_idx]) # n*n identity matrix indexed by site_idx
        names += [f"site_{s}" for s in sites]
    return np.hstack(columns), names # Joins everything into an array with column names

# Builds the features for one period (train, val or test). [:, :, 0] turns 3D shape into 2D shape (LSTM and GRU train on 3D while RF trains on 2D)
def build_features(d, period):
    return features_from_file(d[f"X_{period}"][:, :, 0], d["timeline"][d[f"t_{period}"]], d[f"site_{period}"], d["sites"])

def main():
    if len(sys.argv) > 1: # Takes no arguments
        print("Usage: python random_forest.py")
        print("No arguments are needed. Any values should be set inside of the file.")
        sys.exit(1)

    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib # Used to save the trained model
    except ImportError:
        print("Error: sklearn is not installed!")
        print("Run 'pip install scikit-learn' first")
        sys.exit(1)

    # Loads the windows file (prints the problem and stops if it can not be read)
    d = load_data()
    features = {p: build_features(d, p) for p in PERIODS} # Builds (feature_table, column_names) pair for all periods
    names = features["train"][1] # Keeps column names

    # Trains the forest on the train period only
    print(f"Training {N_TREES} trees on {len(d['y_train']):,} windows with {len(names)} features...")
    start = time.time()
    model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(features["train"][0], d["y_train"])
    print(f"Done in {time.time() - start:.3f} seconds")

    # Predicts, turns everything back into vehicles per 15 minutes and scores it
    results, predictions, site_metrics = evaluate(d, lambda period: model.predict(features[period][0]), "random_forest")

    # Saves everything (model and results .csv)
    try:
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_FILE, compress=3) # Saves the trained model
    except OSError as exc:
        print(f"Error while writing the model: {exc}")
        print("Close the files in the 'models' folder if they are open in another program, then try again")
        sys.exit(1)
    predictions_file, metrics_file, site_metrics_file = save_results("rf", results, predictions, site_metrics)

    # Prints summary to terminal
    print(results.round(2).to_string(index=False))

    # Importance here means how much the model relies on this feature for predictions
    importance = pd.Series(model.feature_importances_, index=names)
    importance = pd.concat([importance[~importance.index.str.startswith("site_")], pd.Series({"site (all columns)": importance[importance.index.str.startswith("site_")].sum()})])
    print("Most important features: " + ", ".join(f"{n} ({v:.2f})" for n, v in importance.sort_values(ascending=False).head(5).items()))
    print(f"Wrote {MODEL_FILE}, {predictions_file}, {metrics_file} and {site_metrics_file}")

if __name__ == "__main__":
    main()