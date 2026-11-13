"""
main.py - single entry point for the whole TBRGS pipeline

Usage of this file:
    python main.py <model> <date and time> <origin> <destination(s)> <search algorithm> (e.g python main.py rf "2026-11-11 12:00" 970 4032 4034 AS)

    model: lstm, gru, rf or all
    date/time: In "YYYY-MM-DD HH:MM" format (rounded down to the quarter hour)
    origin, destination: SCATS site numbers, if more than 1 destination is given the program will pass through them in the order given
    algorithm: DFS, BFS, GBFS, AS, CUS1 or CUS2 (Assignment 2A methods), defaults to AS

Anything the route needs that is missing (processed data, trained model) is built automatically first.
A stage is skipped when its output files are already newer than its input files.
To rebuild from scratch, delete the files in 'data/processed/' and 'models/'.

Extra command:
    python main.py compare # Side-by-side table of every trained model

The commands here will work as long as the raw .xls is in 'data/raw/'
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# File paths and folders
RESULTS = Path("results")
RAW, PROCESSED = Path("data/raw/Scats Data October 2006.xls"), Path("data/processed")
WINDOWS = PROCESSED / "windows.npz"
TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
USAGE = 'python main.py <lstm|gru|rf|all> "<YYYY-MM-DD HH:MM>" <origin> <destination(s)> [algorithm]' # Usage string shown when there is incorrect input

# Mapping model short name to its tuple
MODELS = {
    "lstm": ("lstm.py", "lstm", Path("models/lstm.keras")),
    "gru": ("gru.py", "gru", Path("models/gru.keras")),
    "rf": ("random_forest.py", "rf", Path("models/random_forest.joblib")),
}

# Data stages in order (label, script, input files, output files)
DATA_STAGES = [
    ("Wide -> long format", "wide_to_long.py", [RAW], [PROCESSED / "long_format.csv"]),
    ("Edges + distance graph", "calculate_distance.py", [RAW], [PROCESSED / f for f in ("edges.csv", "sites.csv", "calcdist_graph.txt")]),
    ("Clean site flows", "calculate_flow.py", [PROCESSED / "long_format.csv"], [PROCESSED / "site_flows.csv"]),
    ("Training windows", "make_windows.py", [PROCESSED / "site_flows.csv"], [WINDOWS]),
]

class PipelineError(Exception):
    """A stage has failed and needs to be reported"""

# Prints the heading for each stage
def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")

# Runs a script as a stage
def run_stage(script, inputs, outputs, label=None):
    for path in inputs: # Check if input file is there
        if not path.exists():
            raise PipelineError(f"input file '{path}' not found (the raw SCATS .xls goes in data/raw/)")
    if all(p.exists() for p in outputs) and min(p.stat().st_mtime for p in outputs) >= max(p.stat().st_mtime for p in inputs):
        print(f"[Skipped] {script}: outputs are up to date") # Skips the script if output file is newer than input
        return
    print(f"\n[Run] {label or script}")
    start = time.time()
    if subprocess.run([sys.executable, script]).returncode != 0: # Runs the script as separate process
        raise PipelineError(f"{script} failed, see the message above")
    print(f"[Done] {script} in {time.time() - start:.1f}s")

# Runs all of the scripts for data related tasks
def stage_data():
    banner("DATA PROCESSING")
    for label, script, inputs, outputs in DATA_STAGES:
        run_stage(script, inputs, outputs, f"{label} ({script})")

# Runs the script for model training
def stage_train(models):
    banner("MODEL TRAINING + EVALUATION")
    for name in models:
        script, prefix, model_file = MODELS[name]
        if Path(script).exists():
            run_stage(script, [WINDOWS], [model_file, RESULTS / f"{prefix}_metrics.csv"], f"training {name} ({script}), this can take a few minutes")
        else:
            print(f"[WARN] {script} does not exist, skipping the {name} model")

# Puts the validation and test results of every trained model in one table, only run when the user types 'compare'
def stage_compare():
    banner("MODEL COMPARISON")
    rows = []
    for name, (_script, prefix, _file) in MODELS.items():
        metrics_file, training_file = RESULTS / f"{prefix}_metrics.csv", RESULTS / f"{prefix}_training.csv"
        if not metrics_file.exists(): # If model metrics doesn't exist, the model was not trained yet
            continue
        metrics = pd.read_csv(metrics_file)
        info = pd.read_csv(training_file).iloc[0] if training_file.exists() else {} # Parameters and training time, if the script saved them
        for period in ("val", "test"): # Loops through all of the periods in the metric file
            part = metrics[metrics["period"] == period].set_index("model") # Index the row with the model name
            if part.empty:
                continue
            own = part.drop("previous_reading", errors="ignore").iloc[0] # Drop the 'baseline' row, we want to compare the models only
            row = {"model": name, "period": period, "RMSE": own["RMSE"], "MAE": own["MAE"], "R2": own["R2"]}
            if "previous_reading" in part.index: # Error of baseline RMSE, checks if the model actually helps
                row["baseline_RMSE"] = part.loc["previous_reading", "RMSE"]
            if period == "test" and len(info): # Adds training information for test period
                row.update(parameters=info.get("parameters"), train_seconds=info.get("train_seconds"))
            rows.append(row)
    if not rows: # No models were trained
        print("No model results found yet, route with a model first so it gets trained")
        return
    table = pd.DataFrame(rows).sort_values(["period", "RMSE"]).reset_index(drop=True) # Build the table sorting by period and RMSE
    RESULTS.mkdir(exist_ok=True)
    print(table.round(3).to_string(index=False))
    test = table[table["period"] == "test"]
    if len(test):
        print(f"\nBest test RMSE: {test.iloc[0]['model']} ({test.iloc[0]['RMSE']:.2f} vehicles per 15 min)") # test.iloc[0] is the model with lowest RMSE

# Prepares whatever a route needs but is missing (processed data, trained model), nothing is rebuilt if all files exist
def ensure_ready(model):
    if not all(p.exists() for _label, _script, _inputs, outputs in DATA_STAGES for p in outputs):
        print("Processed data not found, creating it first...")
        stage_data() # Runs stage_data if any is missing
    for name in (MODELS if model == "all" else [model]):
        if not MODELS[name][2].exists():
            print(f"The {name} model has not been trained yet, training it first...")
            stage_train([name]) # Trains model if the required model isn't trained yet

# Predicts the traffic, builds the travel-time graph and finds up to 5 routes through the stops in order
def stage_route(model, departure, origin, destinations, method):
    from calculate_time import build_graph, write_graph_files, TravelTimeError
    from k_routes import k_shortest_paths_via, print_routes

    banner("ROUTE GUIDANCE")
    stops = [origin, *destinations]
    label = " -> ".join(map(str, stops))
    for name in (MODELS if model == "all" else [model]): # Uses the chosen model to do prediction
        try:
            graph = build_graph(name, departure, origin, destinations)
            write_graph_files(graph)
            routes = k_shortest_paths_via(graph["nodes"], graph["edges"], stops, method=method)
        except TravelTimeError as exc:
            message = f"{exc}\n  {exc.hint}" if exc.hint else str(exc)
            if model != "all":
                raise PipelineError(message)
            print(f"[WARN] skipping {name}: {message}") # One model that can not run should not cause all to fail
            continue
        except ValueError as exc: # Bad method name or unknown site
            raise PipelineError(str(exc))

        # Prints out any note if exists and the header line
        for note in graph["notes"]:
            print(f"Note: {note}")
        print(f"\n{name.upper()} | {label} | depart {graph['time']:%a %Y-%m-%d %H:%M} ({graph['mode']} traffic) | {method.upper()} | up to 5 routes")
        print_routes(routes)
        if not routes and len(stops) > 2: # Alerts if a stop is unreachable
            print("A stop can not be reached from the stop before it")

# Reads the command line input and returns (model, time, origin, destinations, method)
def parse_command(argv):
    model, *tokens = argv # Takes ML model and splits everything else as 'token'
    model = model.lower()
    if model not in (*MODELS, "all"):
        sys.exit(f"Error: unknown model '{model}'\nUsage: {USAGE}")
    when = tokens.pop(0) if tokens else ""
    if tokens and TIME_PATTERN.match(tokens[0]): # The date and time were typed as two arguments instead of in quotation marks
        when += " " + tokens.pop(0)
    sites = []
    while tokens and tokens[0].isdigit(): # Origin first, then the destinations
        sites.append(int(tokens.pop(0)))
    if ":" not in when: # Hour and minute in time should be separated with ':'
        sys.exit(f'Error: no date and time, use "YYYY-MM-DD HH:MM"\nUsage: {USAGE}')
    if len(sites) < 2:
        sys.exit(f"Error: needs an origin and at least one destination (SCATS site numbers)\nUsage: {USAGE}")
    if len(tokens) > 1 or any(t.startswith("-") for t in tokens): # Only the search algorithm may follow the stops
        sys.exit(f"Error: unexpected argument(s) after the stops: {' '.join(tokens)}\nUsage: {USAGE}")
    return model, when, sites[0], sites[1:], (tokens[0] if tokens else "AS")

def main():
    os.chdir(Path(__file__).resolve().parent) # All scripts use relative paths
    argv = sys.argv[1:]
    if not argv:
        sys.exit(f"Usage: {USAGE}\n       python main.py compare")
    if argv[0].lower() == "compare": # Only optional command available
        stage_compare()
        return
    model, when, origin, destinations, method = parse_command(argv)

    start = time.time()
    try:
        ensure_ready(model)
        stage_route(model, when, origin, destinations, method)
    except PipelineError as exc:
        sys.exit(f"\nError: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nStopped by user")
    print(f"\nFinished in {time.time() - start:.1f}s\n")

if __name__ == "__main__":
    main()