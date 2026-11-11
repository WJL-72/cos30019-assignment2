"""
Usage:
    python lstm.py

Input:
    data_splitting.py (functions + windows.npz file)

Outputs:
    models/lstm.keras - trained LSTM model (travel_time.py loads this file)
    results/lstm_predictions.csv - actual vs predicted flow for each validation and test window
    results/lstm_metrics.csv - RMSE, MAE, MSE and R2 and a 'previous reading' baseline (same format as rf_metrics.csv)
    results/lstm_site_metrics.csv - test errors of each site
    results/lstm_history.csv - training and validation loss of each epoch
    results/lstm_loss.png - graph of the training and validation loss
    results/lstm_training.csv - training time, epochs, parameters and prediction speed (for comparing models)

The model takes (N, LOOKBACK, 1) as input and gives one scaled value (0-1) as output, which is what travel_time.py expects
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data_splitting import load_windows, to_vehicles, OUTPUT_FILE as WINDOWS_FILE
from random_forest import metrics # Same metrics function as the random forest so the results can be compared

MODEL_FILE = Path("models/lstm.keras")
PREDICTIONS_FILE = Path("results/lstm_predictions.csv")
METRICS_FILE = Path("results/lstm_metrics.csv")
SITE_METRICS_FILE = Path("results/lstm_site_metrics.csv")
HISTORY_FILE = Path("results/lstm_history.csv")
LOSS_PLOT_FILE = Path("results/lstm_loss.png")
TRAINING_FILE = Path("results/lstm_training.csv")

# LSTM settings
UNITS = (64, 64) # Units in the first and second LSTM layer
DROPOUT = 0.2 # Randomly turns off 20% of units before the output layer to reduce overfitting
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 100 # Early stopping will usually stop training before this
PATIENCE = 10 # Stops training if validation loss has not improved for this many epochs
LR_PATIENCE = 4 # Halves the learning rate if validation loss has not improved for this many epochs
RANDOM_STATE = 42 # Fix the seed so its not random on every run

# Builds the LSTM network, input is (lookback, 1) which is 12 past readings with 1 value each
def build_model(lookback):
    from keras import layers, models, optimizers # Imported here so the rest of the file still loads if keras is missing
    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(UNITS[0], return_sequences=True), # return_sequences=True gives every time step to the next LSTM layer
        layers.LSTM(UNITS[1]), # Only gives the last time step
        layers.Dropout(DROPOUT),
        layers.Dense(1), # No activation on the output since scaled val/test flow can go a bit over 1, which sigmoid cannot output
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
    return model

# Saves the loss curve as a picture, skipped if matplotlib is not installed (the csv is still saved)
def plot_history(history):
    try:
        import matplotlib
        matplotlib.use("Agg") # Only saves a file, does not open a window
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.index + 1, history["loss"], label="Training loss")
    ax.plot(history.index + 1, history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled flow)")
    ax.set_title("LSTM training and validation loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_FILE, dpi=150)
    plt.close(fig)
    return True

# Main code block
def main():
    if len(sys.argv) > 1: # Takes no arguments
        print("Usage: python lstm.py")
        print("No arguments are needed. Any values should be set inside of the file.")
        sys.exit(1)

    try:
        import keras
    except ImportError:
        print("Error: keras (with a backend such as tensorflow) is not installed!")
        print("Run 'pip install tensorflow' first")
        sys.exit(1)

    # Loads the windows file
    try:
        d = load_windows(WINDOWS_FILE)
    except FileNotFoundError:
        print(f"Error: file '{WINDOWS_FILE}' not found!")
        print("Run data_splitting.py first to create it")
        sys.exit(1)
    except Exception as exc:
        print(f"Error while reading '{WINDOWS_FILE}': {exc}")
        sys.exit(1)

    keras.utils.set_random_seed(RANDOM_STATE) # Sets the seed for python, numpy and the backend
    lookback = int(d["lookback"])
    model = build_model(lookback)
    model.summary()

    # Trains on the train period only, the validation period decides when to stop (test period is not used here)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True), # Goes back to the best epoch's weights
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-5),
    ]
    print(f"Training LSTM on {len(d['y_train']):,} windows, validating on {len(d['y_val']):,} windows...")
    start = time.time()
    fit = model.fit(d["X_train"], d["y_train"], validation_data=(d["X_val"], d["y_val"]), batch_size=BATCH_SIZE,
                    epochs=MAX_EPOCHS, shuffle=True, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - start
    history = pd.DataFrame(fit.history) # Loss of each epoch
    best_epoch = int(history["val_loss"].idxmin()) + 1 # idxmin starts from 0 but epochs start from 1
    print(f"Done in {train_seconds:.1f} seconds, {len(history)} epochs, best epoch {best_epoch}")

    # Predicts and turns everything back into vehicles per 15 minutes
    results, rows = [], []
    predict_seconds, predict_count = 0.0, 0
    for period in ("val", "test"):
        site = d[f"site_{period}"]
        actual = to_vehicles(d[f"y_{period}"].astype(float), site, d) # Actual flow from the dataset
        t0 = time.time()
        scaled = np.ravel(model.predict(d[f"X_{period}"], batch_size=1024, verbose=0))
        predict_seconds += time.time() - t0
        predict_count += len(scaled)
        lstm = to_vehicles(scaled.astype(float), site, d) # Predicted flow
        previous = to_vehicles(d[f"X_{period}"][:, -1, 0].astype(float), site, d) # Baseline - predicts the last reading it was given (model SHOULD ALWAYS beat this)
        results.append({"model": "lstm", "period": period, **metrics(actual, lstm)})
        results.append({"model": "previous_reading", "period": period, **metrics(actual, previous)})
        rows.append(pd.DataFrame({"period": period, "scats_num": d["sites"][site], "timestamp": d["timeline"][d[f"t_{period}"]],
                                  "actual": actual.round(1), "predicted": lstm.round(1)}))
    results = pd.DataFrame(results) # Converts into a metric table
    predictions = pd.concat(rows, ignore_index=True) # Converts into a stacked predictions table

    # Test errors for each site - checks which intersection the model works well/badly on
    test = predictions[predictions["period"] == "test"].assign(error=lambda t: t["predicted"] - t["actual"])
    site_metrics = test.groupby("scats_num")["error"].agg(windows="size", RMSE=lambda e: np.sqrt((e ** 2).mean()), MAE=lambda e: e.abs().mean(), MSE=lambda e: (e ** 2).mean())

    # Training info used when comparing the models in the report
    training = pd.DataFrame([{"model": "lstm", "parameters": model.count_params(), "epochs_run": len(history), "best_epoch": best_epoch,
                              "train_seconds": round(train_seconds, 1), "predict_ms_per_1000_windows": round(predict_seconds / predict_count * 1e6, 2)}])

    # Saves everything (results .csv and model)
    try:
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_FILE)
        predictions.to_csv(PREDICTIONS_FILE, index=False)
        results.to_csv(METRICS_FILE, index=False)
        site_metrics.to_csv(SITE_METRICS_FILE)
        history.to_csv(HISTORY_FILE, index_label="epoch")
        training.to_csv(TRAINING_FILE, index=False)
        plotted = plot_history(history)
    except OSError as exc:
        print(f"Error while writing the results: {exc}")
        print("Close the files in the 'models' and 'results' folders if they are open in another program, then try again")
        sys.exit(1)

    # Prints summary to terminal
    print(results.round(2).to_string(index=False))
    print(f"Wrote {MODEL_FILE}, {PREDICTIONS_FILE}, {METRICS_FILE}, {SITE_METRICS_FILE}, {HISTORY_FILE}, {TRAINING_FILE}" + (f" and {LOSS_PLOT_FILE}" if plotted else ""))

if __name__ == "__main__":
    main()
