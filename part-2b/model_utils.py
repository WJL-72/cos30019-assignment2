"""
Shared code for all model scripts. Ensures all models are loaded, scored and saved in the same way

Usage for the models:
    from model_utils import PERIODS, load_data, evaluate, save_results # Importing this file's functions

    # Loading the data and then train it
    d = load_data()
    (train the model on d["X_train"] and d["y_train"])

    # Evaluation and saving results
    results, predictions, site_metrics = evaluate(d, lambda period: model.predict(d[f"X_{period}"]), <model>)
    save_results(<model>, results, predictions, site_metrics)

The function given to evaluate() is called with "val" and then "test". 
It must return the SCALED predicted flow (the same 0-1 scale as y_val and y_test), one number for each window of that period. 
evaluate() turns the numbers back into vehicles per 15 minutes before scoring.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from make_windows import load_windows, to_vehicles, OUTPUT_FILE as WINDOWS_FILE

RESULTS_DIR = Path("results")
PERIODS = ["train", "val", "test"]

# Loads the windows file as a dictionary, prints the problem and stops the script if it can not be read
def load_data():
    try:
        return load_windows(WINDOWS_FILE)
    except FileNotFoundError:
        print(f"Error: file '{WINDOWS_FILE}' not found!")
        print("Run make_windows.py first to create it")
        sys.exit(1)
    except Exception as exc:
        print(f"Error while reading '{WINDOWS_FILE}': {exc}")
        sys.exit(1)

# Error metrics of the predictions (actual and predicted are both in vehicles per 15 minutes)
def metrics(actual, predicted):
    error = predicted - actual # Differences for each window
    mse = (error ** 2).mean()
    return {"RMSE": np.sqrt(mse),
            "MAE": np.abs(error).mean(),
            "MSE": mse,
            "R2": 1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum()}

# Scores a model on the validation and test periods. predict(period) must return the scaled predicted flow of every window of that period
def evaluate(d, predict, model_name, periods=("val", "test")):
    results, rows = [], []
    for period in periods:
        site = d[f"site_{period}"]

        actual = to_vehicles(d[f"y_{period}"].astype(float), site, d) # Actual flow from the dataset
        scaled = np.asarray(predict(period), dtype=float).ravel() # ravel() so a Keras output of shape (windows, 1) still works
        if len(scaled) != len(actual):
            raise ValueError(f"the model gave {len(scaled)} predictions for the {period} period, but it has {len(actual)} windows")
        predicted = to_vehicles(scaled, site, d) # Predicted flow

        previous = to_vehicles(d[f"X_{period}"][:, -1, 0].astype(float), site, d) # Baseline - predicts using the last reading it was given (model SHOULD ALWAYS beat this)
        results.append({"model": model_name, "period": period, **metrics(actual, predicted)})
        results.append({"model": "previous_reading", "period": period, **metrics(actual, previous)})
        rows.append(pd.DataFrame({"period": period, "scats_num": d["sites"][site], "timestamp": d["timeline"][d[f"t_{period}"]],"actual": actual.round(1), "predicted": predicted.round(1)}))
    results = pd.DataFrame(results) # Converts into a metrics table
    predictions = pd.concat(rows, ignore_index=True) # Stack the two periods tables together into one

    # Test errors for each site - checks which intersection the model works well/badly on
    test = predictions[predictions["period"] == "test"].assign(error=lambda t: t["predicted"] - t["actual"])
    site_metrics = test.groupby("scats_num")["error"].agg(windows="size", RMSE=lambda e: np.sqrt((e ** 2).mean()), MAE=lambda e: e.abs().mean(), MSE=lambda e: (e ** 2).mean())
    return results, predictions, site_metrics

# Saves the three tables as <model>_predictions.csv, <model>_metrics.csv and <model>_site_metrics.csv
def save_results(model, results, predictions, site_metrics, results_dir=RESULTS_DIR):
    files = [results_dir / f"{model}_predictions.csv", results_dir / f"{model}_metrics.csv", results_dir / f"{model}_site_metrics.csv"]
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(files[0], index=False)
        results.to_csv(files[1], index=False)
        site_metrics.to_csv(files[2])
    except OSError as exc:
        print(f"Error while writing the results: {exc}")
        print(f"Close the files in the '{results_dir}' folder if they are open in another program, then try again")
        sys.exit(1)
    return tuple(files)