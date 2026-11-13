"""
Usage:
    python gru.py

Input:
    make_windows.py (functions + windows.npz file)
    model_utils.py (loading, scoring and saving that is shared with the other models)

Outputs:
    models/gru.keras - trained GRU model (calculate_time.py can load this file)
    results/gru_predictions.csv - actual vs predicted flow for each validation and test window
    results/gru_metrics.csv - RMSE, MAE, MSE and R2 and a 'previous reading' baseline (same format as rf_metrics.csv)
    results/gru_site_metrics.csv - test errors of each site
    results/gru_history.csv - training and validation loss of each epoch
    results/gru_loss.png - graph of the training and validation loss
    results/gru_training.csv - training time, epochs, parameters and prediction speed (for comparing models)

The model takes (N, LOOKBACK, 1) as input and gives one scaled value (0-1) as output, which is what calculate_time.py expects
"""
import sys
import time
from pathlib import Path
import pandas as pd
from model_utils import load_data, evaluate, save_results

MODEL_FILE = Path("models/gru.keras")
HISTORY_FILE = Path("results/gru_history.csv")
LOSS_PLOT_FILE = Path("results/gru_loss.png")
TRAINING_FILE = Path("results/gru_training.csv")

# GRU settings (kept the same as the LSTM and Randpm Forest so the two models can be compared fairly, Thanks Evan)
UNITS = (64, 64) # Units in the first and second GRU layer
DROPOUT = 0.2 # Randomly turns off 20% of units before the output layer to reduce overfitting
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 100 # Early stopping will usually stop training before this
PATIENCE = 10 # Stops training if validation loss has not improved for this many epochs
LR_PATIENCE = 4 # Halves the learning rate if validation loss has not improved for this many epochs
RANDOM_STATE = 42 # Fix the seed so its not random on every run

# Builds the GRU network, input is (lookback, 1) which is 12 past readings with 1 value each
def build_model(lookback):
    from keras import layers, models, optimizers # Imported here so the rest of the file still loads if keras is missing
    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.GRU(UNITS[0], return_sequences=True), # return_sequences=True gives every time step to the next GRU layer
        layers.GRU(UNITS[1]), # Only gives the last time step
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
    ax.set_title("GRU training and validation loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_FILE, dpi=150)
    plt.close(fig)
    return True

# Main code block
def main():
    if len(sys.argv) > 1: # Takes no arguments
        print("Usage: python gru.py")
        print("No arguments are needed. Any values should be set inside of the file.")
        sys.exit(1)

    try:
        import keras
    except ImportError:
        print("Error: keras (with a backend such as tensorflow) is not installed!")
        print("Run 'pip install tensorflow' first")
        sys.exit(1)

    # Loads the windows file
    d = load_data()

    keras.utils.set_random_seed(RANDOM_STATE) # Sets the seed for python, numpy and the backend
    lookback = int(d["lookback"])
    model = build_model(lookback)
    model.summary()

    # Trains on the train period only, the validation period decides when to stop (test period is not used here)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True), # Goes back to the best epoch's weights
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-5),
    ]
    print(f"Training GRU on {len(d['y_train']):,} windows, validating on {len(d['y_val']):,} windows...")
    start = time.time()
    fit = model.fit(d["X_train"], d["y_train"], validation_data=(d["X_val"], d["y_val"]), batch_size=BATCH_SIZE,
                    epochs=MAX_EPOCHS, shuffle=True, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - start
    history = pd.DataFrame(fit.history) # Loss of each epoch
    best_epoch = int(history["val_loss"].idxmin()) + 1 # idxmin starts from 0 but epochs start from 1
    print(f"Done in {train_seconds:.1f} seconds, {len(history)} epochs, best epoch {best_epoch}")

    # Predicts for one period and keeps count of how long it takes (used for the training info below)
    timing = {"seconds": 0.0, "windows": 0}
    def predict(period):
        t0 = time.time()
        scaled = model.predict(d[f"X_{period}"], batch_size=1024, verbose=0)
        timing["seconds"] += time.time() - t0
        timing["windows"] += len(scaled)
        return scaled

    # Turns everything back into vehicles per 15 minutes and scores it
    results, predictions, site_metrics = evaluate(d, predict, "gru")

    # Training info used when comparing the models in the report
    training = pd.DataFrame([{"model": "gru", "parameters": model.count_params(), "epochs_run": len(history), "best_epoch": best_epoch,
                              "train_seconds": round(train_seconds, 1), "predict_ms_per_1000_windows": round(timing["seconds"] / timing["windows"] * 1e6, 2)}])

    # Saves everything (results .csv and model)
    predictions_file, metrics_file, site_metrics_file = save_results("gru", results, predictions, site_metrics)
    try:
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_FILE)
        history.to_csv(HISTORY_FILE, index_label="epoch")
        training.to_csv(TRAINING_FILE, index=False)
        plotted = plot_history(history)
    except OSError as exc:
        print(f"Error while writing the results: {exc}")
        print("Close the files in the 'models' and 'results' folders if they are open in another program, then try again")
        sys.exit(1)

    # Prints summary to terminal
    print(results.round(2).to_string(index=False))
    print(f"Wrote {MODEL_FILE}, {predictions_file}, {metrics_file}, {site_metrics_file}, {HISTORY_FILE}, {TRAINING_FILE}" + (f" and {LOSS_PLOT_FILE}" if plotted else ""))

if __name__ == "__main__":
    main()
