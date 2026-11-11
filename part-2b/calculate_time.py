"""
Usage (command line):
    python calculate_time.py <model> "<departure time>" [origin destination ...] (e.g. : python calculate_time.py rf "2006-10-30 08:00" 970 4032)
 
    model: lstm, gru or rf (random forest)
    departure time: "YYYY-MM-DD HH:MM" on a quarter hour, inside the data (2006-10-01 03:00 to 2006-10-31 23:45)
    origin, destinations: SCATS site numbers, they are written into the graph file (default 970 and 4032)

Usage (from other code, e.g. the GUI or the main program):
    from travel_time import build_graph, TravelTimeError
    try:
        graph = build_graph("rf", "2006-10-30 08:00", 970, [4032])
    except TravelTimeError as exc:
        ...
    graph["nodes"], graph["edges"]  # Same layout that parse_file() gives for the Assignment 2A search methods, edge costs are minutes

NOTE: MODELS MUST BE TRAINED BEFORE RUNNING THIS FILE. ALSO I ASSUMED THAT LSTM AND GRU WILL BE BUILT USING KERAS
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from make_windows import load_windows, load_flows , INTERVAL_MIN, INPUT_FILE, OUTPUT_FILE as WINDOWS_FILE
from random_forest import features_from_file

# File paths
MODEL_FILES = {"lstm": Path("models/lstm.keras"), "gru": Path("models/gru.keras"), "rf": Path("models/random_forest.joblib"), }
EDGES_FILE = Path("data/processed/edges.csv")
SITES_FILE = Path("data/processed/sites.csv")
GRAPH_OUTPUT = Path("data/processed/graph_travel_time.txt") # The graph file written, the cost of each edge is its travel time in minutes
TIMES_OUTPUT = Path("data/processed/edge_times.csv") # Flow, speed and travel time of every edge

# Variables to set
DEFAULT_ORIGIN = 970
DEFAULT_DESTINATIONS = [4032]
INCLUDE_GUESSED_EDGES = False # False leaves out the edges marked 'guess', same logic as '--edges' in calculate_distance.py
FLOW_PER_APPROACH = False # False uses the summed flow of the whole site. True divides it by the number of approaches (roads) at the site

# Flow to speed formula from the conversion sheet: flow (veh/h) = -A x speed^2 + B x speed
A, B = 1.4648375, 93.75
SPEED_LIMIT_KMH = 60 # Speed limit with low traffic volume
LIGHT_TRAFFIC_FLOW = 351 # At or below this flow, the speed is the speed limit
INTERSECTION_DELAY_MIN = 0.5 # 30 seconds for each intersection

# Error encountered when running the script.
class TravelTimeError(Exception):
    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint # Optional second line telling the user what to do

# Assuming the road is below capacity, it takes the larger root of the formula
def speed_kmh(flow_per_hour):
    flow = np.asarray(flow_per_hour, dtype=float) # Turns it into a float array
    discriminant = np.maximum(B * B - 4 * A * flow, 0) # Rearraged formula. The formula has no solution above about 1500 veh/h, so it stops at the peak
    speed = np.minimum((B + np.sqrt(discriminant)) / (2 * A), SPEED_LIMIT_KMH) # Takes the larger root
    return np.where(flow <= LIGHT_TRAFFIC_FLOW, SPEED_LIMIT_KMH, speed) # Returns 60 km/h if flow is below 351, otherwise return calculated speed

# Uses the model to predict the site flow (scaled to 0-1), inputs has shape (sites, lookback)
def predict_scaled(model_name, inputs, time, d):
    path = MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(path)
    if model_name == "rf": # Random forest is built using SciKit
        import joblib
        features, _ = features_from_file(inputs, [time] * len(inputs), np.arange(len(inputs)), d["sites"])
        return joblib.load(path).predict(features)
    import keras # LSTM and GRU are ASSUMED to be built using Keras
    return np.ravel(keras.models.load_model(path).predict(inputs[..., np.newaxis], verbose=0))

# Reads the departure time and checks that it can be used. Raises ValueError if not
def read_time(text, timeline, lookback, horizon):
    try:
        time = pd.Timestamp(text) # Turns input into timestamp
    except ValueError:
        raise ValueError(f"'{text}' is not a date and time, use \"YYYY-MM-DD HH:MM\"")
    if time.minute % INTERVAL_MIN != 0 or time.second != 0: # Minute must be a multiple of quater-hour (15 minutes)
        raise ValueError(f"'{text}' is not on a quarter hour (for example 08:00, 08:15)")
    if time not in timeline: # Time is outside of the time set
        raise ValueError(f"'{text}' is outside the data ({timeline[0]:%Y-%m-%d %H:%M} to {timeline[-1]:%Y-%m-%d %H:%M})")
    if timeline.get_loc(time) < lookback + horizon - 1: # Must have at least 12 slots before it
        raise ValueError(f"'{text}' is too early, it needs {lookback} earlier readings (first possible time is {timeline[lookback + horizon - 1]:%Y-%m-%d %H:%M})")
    return time

# Loads the windows file (settings and scaling), the flows, the sites and the edges. Raises TravelTimeError if any is missing or wrong
def load_inputs():
    try:
        d = load_windows(WINDOWS_FILE)
        flows = load_flows(INPUT_FILE)
        sites = list(d["sites"])
        if list(flows.columns) != sites: # Checks if the flow table and window file have the sites in the same order
            raise ValueError("the sites in site_flows.csv are different from the ones in windows.npz, run make_windows.py again")
        edges = pd.read_csv(EDGES_FILE)
        site_table = pd.read_csv(SITES_FILE, index_col=0)
        missing = {"site_a", "site_b", "distance_km"} - set(edges.columns) # Look for missing columns
        if missing:
            raise ValueError(f"{EDGES_FILE} is missing column(s): {', '.join(sorted(missing))}")
    except FileNotFoundError as exc:
        raise TravelTimeError(f"file '{exc.filename}' not found!", "Run the scripts that create it first (wide_to_long.py -> calculate_distance.py -> calculate_flow.py -> make_windows.py)")
    except Exception as exc:
        raise TravelTimeError(f"could not read the input files: {exc}")
    return d, flows, sites, edges, site_table

# Predicts the flow (vehicles per hour) of every site at the departure time, sites that can not be predicted get NaN
def predict_site_flows(model_name, flows, time, d):
    lookback, horizon = int(d["lookback"]), int(d["horizon"])

    # Takes the readings that come just before the departure time, flow at actual departure time is NOT checked
    values = (flows.to_numpy(dtype=float) - d["scaler_min"]) / d["scaler_span"] # Same scaling as in windows.npz
    end = flows.index.get_loc(time) - (horizon - 1) # Finds departure time and then step back by (horizon - 1) so end is exactly the departure row when horizon = 1
    inputs = values[end - lookback:end].T # Takes 12 rows before end and gets shape (sites, 12)
    no_input = np.isnan(inputs).any(axis=1) # Marks sites with a gap in their last 12 readings (hard to predict)
    inputs = np.nan_to_num(inputs) # Replace NaN with 0 (will be discarded later)

    # Uses model to predict then turns scaled numbers to vehicles per hour
    try:
        scaled = predict_scaled(model_name, inputs, time, d) # Get scaled predicted flow
    except FileNotFoundError:
        raise TravelTimeError(f"file '{MODEL_FILES[model_name]}' not found!", f"Train the {model_name} model first")
    except ImportError as exc:
        raise TravelTimeError(f"a library for the {model_name} model is not installed ({exc})")
    except Exception as exc:
        raise TravelTimeError(f"prediction with the {model_name} model failed: {exc}")
    
    flow_per_hour = np.maximum(scaled * d["scaler_span"] + d["scaler_min"], 0) * (60 // INTERVAL_MIN) # Undo the scaling then changes to vehicles per hour
    if FLOW_PER_APPROACH: # Only used when set as TRUE - divide the site flow with number of approaches
        n_approaches = pd.read_csv(INPUT_FILE, usecols=["scats_num", "n_approaches"]).drop_duplicates("scats_num").set_index("scats_num")["n_approaches"]
        flow_per_hour = flow_per_hour / n_approaches.reindex(list(d["sites"])).to_numpy()

    flow_per_hour[no_input] = np.nan # No prediction for these sites
    return flow_per_hour, no_input

# Travel time of each edge (assumed same on both directions, uses the flow and speed from starting site). Returns the table and the number of guessed edges left out
def edge_travel_times(edges, site_info):
    if "source" in edges.columns and not INCLUDE_GUESSED_EDGES:
        left_out = int((edges["source"] == "guess").sum()) # Leave out 'guessed' edges (from straight line distance)
        edges = edges[edges["source"] != "guess"]
    else:
        left_out = 0

    forward = edges.rename(columns={"site_a": "from_site", "site_b": "to_site"}) # Forward treats 'A' as start and 'B' as end
    backward = edges.rename(columns={"site_b": "from_site", "site_a": "to_site"}) # Opposite of 'forward'
    both = pd.concat([forward, backward], ignore_index=True)[["from_site", "to_site", "distance_km"]] # Makes sure each road apprears in both directions
    both = both.join(site_info, on="from_site") # Attaches the site information based on starting site
    both["travel_time_min"] = (both["distance_km"] / both["speed_kmh"] * 60 + INTERSECTION_DELAY_MIN).round(3) # Calculating the travel time from speed and distance (accounts for 30 sec delay)
    return both.sort_values(["from_site", "to_site"]).reset_index(drop=True), left_out

# Predicts the traffic at the departure time and builds the graph with travel times as the edge costs. Raises TravelTimeError for any problem the user can fix
# Returns a dictionary with these keys: {model, time, origin, destinations, nodes, edges, edge_table, site_info, notes}
def build_graph(model_name, departure_time, origin=DEFAULT_ORIGIN, destinations=None):
    destinations = list(DEFAULT_DESTINATIONS if destinations is None else destinations)
    model_name = str(model_name).lower()
    if model_name not in MODEL_FILES: # Checking if the model exists (must be rf, lstm or gru)
        raise TravelTimeError(f"'{model_name}' does not exist!", f"Models available: {', '.join(MODEL_FILES)}")
    d, flows, sites, edges, site_table = load_inputs()

    # Checks the departure time, origin and destinations
    try:
        time = read_time(str(departure_time), flows.index, int(d["lookback"]), int(d["horizon"])) # Validates the departure time
    except ValueError as exc:
        raise TravelTimeError(str(exc))
    unknown = [s for s in [origin] + destinations if s not in sites] # List of origin and destination sites not in the data
    if unknown:
        raise TravelTimeError(f"site(s) {', '.join(map(str, unknown))} not found in the data!")

    flow_per_hour, no_input = predict_site_flows(model_name, flows, time, d)
    speed = np.where(no_input, SPEED_LIMIT_KMH, speed_kmh(np.nan_to_num(flow_per_hour))) # Calculate speed from flow, if no prediction set at speed limit
    site_info = pd.DataFrame({"flow_veh_per_h": flow_per_hour, "speed_kmh": speed, "no_prediction": no_input}, index=sites)
    both, left_out = edge_travel_times(edges, site_info)

    # Same layout as parse_file() in Assignment 2A (coordinates rounded to 4 decimals, same as the graph file)
    nodes = {int(s): (round(float(site_table.at[s, "x"]), 4), round(float(site_table.at[s, "y"]), 4)) for s in sites}
    graph_edges = {int(s): [] for s in sites}
    for e in both.itertuples():
        graph_edges[int(e.from_site)].append((int(e.to_site), float(e.travel_time_min)))

    notes = []
    if no_input.any(): # Checks if any site has no input, will use speed limit for them
        notes.append(f"No prediction for site(s): {', '.join(str(s) for s, bad in zip(sites, no_input) if bad)} (empty readings just before this time, used 60 km/h)")
    if left_out: # States how many guessed edges are skipped
        notes.append(f"Skipped {left_out} guessed edge(s)")
    return {"model": model_name, "time": time, "origin": origin, "destinations": destinations, "nodes": nodes, "edges": graph_edges,
            "edge_table": both, "site_info": site_info, "notes": notes}

# Writes the graph in the Assignment 2A format and a .csv file of edge times
def write_graph_files(graph, graph_path=GRAPH_OUTPUT, times_path=TIMES_OUTPUT):
    try:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "w") as f:
            f.write("Nodes:\n")
            for site, (x, y) in graph["nodes"].items():
                f.write(f"{site}: ({x:.4f},{y:.4f})\n")
            f.write("Edges:\n")
            for e in graph["edge_table"].itertuples():
                f.write(f"({e.from_site},{e.to_site}): {e.travel_time_min}\n")
            f.write(f"Origin:\n{graph['origin']}\nDestinations:\n{'; '.join(map(str, graph['destinations']))}\n")
        graph["edge_table"].round({"flow_veh_per_h": 1, "speed_kmh": 1}).to_csv(times_path, index=False) # Saving the .csv file
    except OSError as exc:
        raise TravelTimeError(f"could not write to '{graph_path.parent}': {exc}", "Close the files in that folder if they are open in another program, then try again")

# Main code block (only used when executed through command line)
def main():
    if len(sys.argv) < 3: # Needs at least the model and the departure time
        print('Usage: python calculate_time.py <model> "<departure time>" [origin destination ...]')
        print('Example: python calculate_time.py rf "2006-10-30 08:00" 970 4032')
        print(f"Models available: {', '.join(MODEL_FILES)}")
        sys.exit(1)
    try:
        numbers = [int(a) for a in sys.argv[3:]]
    except ValueError:
        print("Error: origin and destinations must be SCATS site numbers!")
        sys.exit(1)
    if len(numbers) == 1:
        print("Error: give an origin and at least one destination or none at all (default - origin: 970, destination: 4032)")
        sys.exit(1)
    origin, destinations = (numbers[0], numbers[1:]) if numbers else (DEFAULT_ORIGIN, DEFAULT_DESTINATIONS)

    # Builds and writes the graph files
    try:
        graph = build_graph(sys.argv[1], sys.argv[2], origin, destinations)
        write_graph_files(graph)
    except TravelTimeError as exc:
        print(f"Error: {exc}")
        if exc.hint:
            print(exc.hint)
        sys.exit(1)

    # Prints summary to terminal
    info, times = graph["site_info"], graph["edge_table"]["travel_time_min"]
    print(f"Model {graph['model']}, departure {graph['time']:%Y-%m-%d %H:%M}, {len(graph['edge_table'])} directed edges")
    print(f"Predicted flow: {info['flow_veh_per_h'].min():.0f} - {info['flow_veh_per_h'].max():.0f} veh/h | Speed: {info['speed_kmh'].min():.1f} - {info['speed_kmh'].max():.1f} km/h, "
          f"Edge times: {times.min():.2f} - {times.max():.2f} min")
    for note in graph["notes"]:
        print(note)
    print(f"Wrote {GRAPH_OUTPUT} and {TIMES_OUTPUT}")

if __name__ == "__main__":
    main()