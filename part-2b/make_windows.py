"""
Usage:
    python make_windows.py (reads data/processed/site_flows.csv and writes data/processed/windows.npz and window-checking.csv)

Input:
    site_flows.csv (calculate_flow.py)

Output:
    windows.npz (better for training ML algorithms than using CSV)
    window-checking.csv (shows how many window for each site - mainly for documentation and checking)
    
Using the result in a training script:
    d = load_windows()
    X_train, y_train = d["X_train"], d["y_train"] # shapes (N, LOOKBACK, 1) and (N,)
    vehicles = to_vehicles(predictions, d["site_test"], d) # turns scaled values back into vehicles per 15 min
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

INPUT_FILE = Path("data/processed/site_flows.csv") # Output of calculate_flow.py
OUTPUT_FILE = Path("data/processed/windows.npz") # Windows, scaler and site list
COUNTS_FILE = Path("data/processed/window-checking.csv") # Number of windows per site and period

LOOKBACK = 12 # Reads back 3 hours of reading. (15 min x 12 = 3 hours)
HORIZON = 1 # Predicts 15 minutes ahead
VAL_START = "2006-10-23" # First day of the validation set
TEST_START = "2006-10-27" # First day of the test set - everything before test and val set is in training set
INTERVAL_MIN = 15
DAILY_INTERVALS = 96

# Reads site_flows.csv into a table with one row for each time and one column for each site
def load_flows(path):
    df = pd.read_csv(path)
    missing = {"scats_num", "date", "interval", "flow"} - set(df.columns) # Finds any column names not in the file
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}") # Raises error if there is any missing column

    # Builds reading time in minutes from midnight (e.g: V08 -> 8, 8 * 15 min -> 160 min), to_timedelta turns it into a time span in minutes
    df["timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["interval"].str[1:].astype(int) * INTERVAL_MIN, unit="min")
    if df.duplicated(["scats_num", "timestamp"]).any():
        raise ValueError("Some site and time pairs appear more than once") # Raises error since pivot only takes unique data
    flows = df.pivot(index="timestamp", columns="scats_num", values="flow").sort_index() # Convert long table into wide table

    # Every site must have a reading or empty cell every 15 minutes with no gap in the time
    gaps = flows.index.to_series().diff().dropna().unique() # diff gets time between each row and unique lists the unique times
    if len(flows) != df["date"].nunique() * DAILY_INTERVALS or len(gaps) != 1 or gaps[0] != pd.Timedelta(minutes=INTERVAL_MIN): # Checks for correct num of rows, same gap size and exactly 15 minute gaps
        raise ValueError("The time line is not a complete 15-minute grid") # Missing times (gap in timeline)
    if flows.isna().all().any(): # Checks for sites with no readings across all days
        raise ValueError(f"Site(s) with no readings at all: {', '.join(map(str, flows.columns[flows.isna().all()]))}")
    return flows

# Finds where each period starts and ends in the time line
def split_bounds(timeline):
    val_start, test_start = pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)
    if not timeline[0] < val_start < test_start <= timeline[-1]: # Checks if the timeline is in order
        raise ValueError(f"Validation set and test set must be in order and inside {timeline[0].date()} to {timeline[-1].date()}")
    a, b = int(timeline.searchsorted(val_start)), int(timeline.searchsorted(test_start)) # a is first row of validation (23 oct) and b is first row of test (27 oct)
    return {"train": (0, a), "val": (a, b), "test": (b, len(timeline))} # Gets starting and ending row of each set

# Makes the windows for one site
def data_splitting(series, start, stop):
    w = sliding_window_view(series, LOOKBACK) # Lists every 12 consecutive readings (i to i + 11)
    n = len(w) - HORIZON # Last horizon window have no target, n is the number of windows that actually have a target
    X = w[:n] # Keep windows with a target
    target = np.arange(n) + LOOKBACK - 1 + HORIZON # Position of each target in the time line
    y = series[target]

    # Keeps the window if target is between the set period, input is not empty (has flow) and target is not empty 
    keep = (target >= start) & (target < stop) & ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[keep], y[keep], target[keep]

# Turns scaled values (0-1) back into vehicles per 15 minutes, site_idx says which site it belongs to
def to_vehicles(scaled, site_idx, d):
    return scaled * d["scaler_span"][site_idx] + d["scaler_min"][site_idx]

# Loads the file written by this script, used for training ML algorithms
def load_windows(path=OUTPUT_FILE):
    return dict(np.load(path))

# Main code block
def main():
    if len(sys.argv) > 1: # Takes no arguments
        print("Usage: python make_windows.py")
        print("No arguments are needed. Any values should be set inside of the file.")
        sys.exit(1)
    if LOOKBACK < 1 or HORIZON < 1: # Checking if the lookback and horizon settings are correct (not 0 or negative)
        print("Error: LOOKBACK and HORIZON must be at least 1") 
        sys.exit(1)

    # Error handling for loading the flows
    try:
        flows = load_flows(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: file '{INPUT_FILE}' not found!")
        print("Run calculate_flow.py first to create it")
        sys.exit(1)
    except Exception as exc:
        print(f"Error while reading '{INPUT_FILE}': {exc}")
        sys.exit(1)
    try:
        bounds = split_bounds(flows.index) # Any wrong date will trigger an error
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    sites = list(flows.columns)
    values = flows.to_numpy(dtype=float) # Rows are times and columns are sites, empty reading stored as NaN

    # Scales each site to 0-1 using the train days only
    train_values = values[:bounds["train"][1]] # Takes until end of training only, prevent validation or test set messing up scaling here
    lo, hi = np.nanmin(train_values, axis=0), np.nanmax(train_values, axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo) # Avoids dividing by zero for a site with flow that never changes
    scaled = (values - lo) / span # NaN stays NaN

    # Makes the windows for each period and each site
    data = {}
    counts = pd.DataFrame(index=pd.Index(sites, name="scats_num"))
    for period, (start, stop) in bounds.items(): # Loops through training, validation and test time frames
        parts = [data_splitting(scaled[:, j], start, stop) for j in range(len(sites))] # Gets the window of the entire site series

        # The 'shape' of data taken by LSTM and GRU is (N, 12, 1) where N = number of windows, 12 = lookback/batch_size and 1 is the dimension size (one dimension since there's only 1 input)
        data[f"X_{period}"] = np.concatenate([p[0] for p in parts])[..., np.newaxis].astype(np.float32) # np.concantenance makes (N, 12) while new axis adds the 1
        data[f"y_{period}"] = np.concatenate([p[1] for p in parts]).astype(np.float32)
        data[f"t_{period}"] = np.concatenate([p[2] for p in parts]).astype(np.int32) # Target position in timeline
        data[f"site_{period}"] = np.concatenate([np.full(len(p[1]), j) for j, p in enumerate(parts)]).astype(np.int16) # Site index
        counts[period] = [len(p[1]) for p in parts]
        counts[f"{period}_possible"] = stop - start # (train/val/test)_possible is the 'ideal' number of windows if the data were to be perfect (no empty)

    # Saves all output files
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # '**data' unpacks dictionary into arguments (X_train, Y_train,..), other items are site listing, timeline, scaling and settings used (lookback, horizon)
        np.savez_compressed(OUTPUT_FILE, **data, sites=np.array(sites), timeline=flows.index.strftime("%Y-%m-%d %H:%M").to_numpy(str),scaler_min=lo, scaler_span=span, lookback=LOOKBACK, horizon=HORIZON)
        counts.to_csv(COUNTS_FILE) # Window counts (for checking)
    except OSError as exc:
        print(f"Error while writing to '{OUTPUT_FILE.parent}': {exc}")
        print("Close the files in that folder if they are open in another program, then try again")
        sys.exit(1)

    print(f"{len(sites)} sites, look-back {LOOKBACK} readings, horizon {HORIZON}") # Prints out overall summary
    for period, (start, stop) in bounds.items():
        print(f"{period:5s} {flows.index[start].date()} to {flows.index[stop - 1].date()}: {len(data[f'y_{period}']):,} windows") # Print each period's first date and last date + window count
    # Sites where less than half of the targets could be used
    for period in ("val", "test"):
        low = counts[counts[period] < counts[f"{period}_possible"] / 2]
        if len(low):
            print(f"Sites with less than half of their {period} windows: " + ", ".join(f"{s} ({row[period]})" for s, row in low.iterrows()))
    print(f"Wrote {OUTPUT_FILE} and {COUNTS_FILE}")

if __name__ == "__main__":
    main()