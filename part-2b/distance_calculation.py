"""
Arguments and usage of this file:
    python build_graph_simple.py "Scats Data October 2006.xls" # auto edges -> edges.csv, graph.png, graph_testcase.txt
    python build_graph_simple.py "Scats Data October 2006.xls" --edges edges.csv # rebuild from your hand-edited edges.csv
    python build_graph_simple.py "Scats Data October 2006.xls" --isolated drop # leave sites with no road link out of the graph
"""
import argparse
import math
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg") # Saves it to a file

ORIGIN = 970 # Placeholder origin for the testcase file
DESTINATIONS = [2200] # Placeholder destination point
COST_MODE = "freeflow" # Either "distance" for edge distance or "freeflow" for placeholder travel time calculated
OUTPUT = "graph_testcase.txt" # Name of the graph file written
MAX_EDGE_KM = 2.5 # Drops candidate edges longer than this 
FREE_FLOW_KMH = 60 # Speed limit
INTERSECTION_DELAY_MIN = 0.5 # 30 secs per intersections

# Ambiguous names of the same road in the data
ROAD_ALIASES = {
    "HARP_ST": "HARP_RD",
    "EASTERN_FWY_W_BD_RAMPS": "EASTERN_FWY",
    "OFFRAMP_EASTERN_FWY": "EASTERN_FWY",
}

# Matches "<road> <direction> of <cross road>", e.g. "HIGH_ST NE OF PRINCESS_ST" and then removes the direction since its irrelevant
LOCATION_PATTERN = re.compile(r"^(.+?)\s+(?:N|S|E|W|NE|NW|SE|SW)\s+OF\s+(.+)$", re.IGNORECASE)

# Loading coordinates and road names
def load_sites(xls_path: str) -> pd.DataFrame:

    # Same code as data_processing.py
    raw = pd.read_excel(xls_path, sheet_name="Data", header=1, dtype={"SCATS Number": str})
    raw["SCATS Number"] = raw["SCATS Number"].str.strip()
    raw["SCATS Number"] = raw["SCATS Number"].astype(int) # Changes nodes like '0970' to '970'

    # Each detector row has its own slightly different coordinates so average them
    good = raw[(raw["NB_LATITUDE"] != 0) & (raw["NB_LONGITUDE"] != 0)] # Drops nodes with no coordinates (4266)
    sites = good.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].mean() # Averages out the site coordinates
    sites.columns = ["lat", "lon"]

    # Splits every location into its two road names and tidies it up, then add it to a set pairing the site number to its location name
    roads = {}
    for scats, location in zip(raw["SCATS Number"], raw["Location"]):
        match = LOCATION_PATTERN.match(location.strip()) # If match, the group is the two road names. If not, the whole location is treated as one name
        for name in (match.groups() if match else (location,)):
            name = re.sub(r"\s+", " ", name.strip().upper())
            roads.setdefault(scats, set()).add(ROAD_ALIASES.get(name, name))
    sites["roads"] = pd.Series({k: sorted(v) for k, v in roads.items()})

    # Converts the latitude and longitude into kilometers where x = east and y = north.
    # 1 degree of latitude is about 111.195 km [6371 x (π / 180) = 111.195], a degree of longitude is shorter by cos(latitude)
    sites["x"] = (sites["lon"] - sites["lon"].mean()) * 111.195 * math.cos(math.radians(sites["lat"].mean()))
    sites["y"] = (sites["lat"] - sites["lat"].mean()) * 111.195
    return sites

# Straight-line distance in km between two sites, using the same x/y coordinates that are written into the graph file
def distance_km(sites, a, b) -> float:
    return math.hypot(sites.at[a, "x"] - sites.at[b, "x"], sites.at[a, "y"] - sites.at[b, "y"])

# Getting the candidate edges and returns [site_a, site_b, road and distance in km]
def auto_edges(sites: pd.DataFrame) -> pd.DataFrame:
    # For each road name, get its sites
    by_road = {}
    for scats, row in sites.iterrows():
        for road in row["roads"]:
            by_road.setdefault(road, []).append(scats)

    rows = []
    for road, members in by_road.items():
        pts = sites.loc[members] # pts is part of the site tables matching those sites
        # Puts the sites in order along the road - by x if the road runs mostly east-west, otherwise by y
        along = "x" if pts["x"].max() - pts["x"].min() >= pts["y"].max() - pts["y"].min() else "y"
        ordered = pts.sort_values(along).index

        for a, b in zip(ordered, ordered[1:]): # Pairs each site with the next - (1,2), (2,3) .....
            a, b = sorted((a, b)) # Smaller site number first so the same pair is never listed twice
            d = distance_km(sites, a, b)
            if d <= MAX_EDGE_KM: # Edge must be less than 2.5km, if more it is likely wrong
                rows.append({"site_a": a, "site_b": b, "road": road, "distance_km": round(d, 3), "source": "road"})
    edges = pd.DataFrame(rows).drop_duplicates(subset=["site_a", "site_b"])  # Two roads that can join the same sites, so drop it
    return edges.sort_values(["site_a", "site_b"]).reset_index(drop=True)

# Site with no road name gets a 'guessed' edge based on its nearest neighbour. MAY BE INACCURATE, THIS IS AN EARLY VERSION
def connect_isolated(sites: pd.DataFrame, edges: pd.DataFrame, k: int = 2, max_km: float = 3.0) -> pd.DataFrame:
    linked = set(edges["site_a"]) | set(edges["site_b"]) # Every site that appears in an edge gets skipped, only keep the ones without edges
    rows = []
    for s in sites.index:
        if s in linked:
            continue
        nearest = sorted((distance_km(sites, s, o), o) for o in linked)[:k] # Gets the nearest neighbour that has an edge (not isolated)
        rows += [{"site_a": s, "site_b": o, "road": "NEAREST_GUESS", "distance_km": round(d, 3), "source": "guess"}
                 for d, o in nearest if d <= max_km]
    return pd.concat([edges, pd.DataFrame(rows)], ignore_index=True)

# Output for the test case
# Writes the output into a testcase format like assignment 2A's. Made in case the algorithms search in the same way
def write_testcase(path, sites, edges):
    with open(path, "w") as f:
        f.write("Nodes:\n")
        for scats, row in sites.iterrows():
            f.write(f"{scats}: ({row['x']:.4f},{row['y']:.4f})\n")
        f.write("Edges:\n")
        for _, e in edges.iterrows():
            d = e["distance_km"]
            # Edge cost is either distance between two node or free-flow time
            # Free-flow travel time including intersection delay
            cost = round(d if COST_MODE == "distance" else d / FREE_FLOW_KMH * 60 + INTERSECTION_DELAY_MIN, 3)
            f.write(f"({e['site_a']},{e['site_b']}): {cost}\n")
            f.write(f"({e['site_b']},{e['site_a']}): {cost}\n")
        f.write(f"Origin:\n{ORIGIN}\nDestinations:\n{'; '.join(map(str, DESTINATIONS))}\n")

# Draws out the graph using matplotlib, used to visualise the map and see if everything gets mapped properly
def plot_graph(path, sites, edges):
    connected = set(edges["site_a"]) | set(edges["site_b"])
    fig, ax = plt.subplots(figsize=(12, 10))
    for _, e in edges.iterrows():
        a, b = sites.loc[e["site_a"]], sites.loc[e["site_b"]]
        guess = e["source"] == "guess" # 'Guessed' edges are the nearest-neighbour edges
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#ff9d00" if guess else "#888888",
        ls="--" if guess else "-", lw=1.2, zorder=1)
    for scats, row in sites.iterrows():
        colour = "#2b7bba" if scats in connected else "#ff0000" # Red is used for site without edges
        ax.scatter(row["x"], row["y"], s=60, color=colour, zorder=2)
        ax.annotate(str(scats), (row["x"], row["y"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_aspect("equal"); ax.set_xlabel("x (km east)"); ax.set_ylabel("y (km north)")
    ax.set_title("SCATS sites and candidate edges (dashed orange = guessed, red = no edges)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

# Main code block
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the SCATS route graph in Assignment 2A format")
    ap.add_argument("xls")
    ap.add_argument("--edges", help="use edges CSV instead of the automatic guess (columns: site_a, site_b, road, distance_km)")
    ap.add_argument("--isolated", choices=["connect", "drop", "leave"], default="connect",
                    help="sites with no edges: connect = join to 2 nearest sites (guess), drop = leave them out of the graph, leave = keep as is")
    args = ap.parse_args()

    # Loads the sites and gets edges from either the original file or the '--edges' file
    sites = load_sites(args.xls)
    if args.edges:
        edges = pd.read_csv(args.edges)
        if "source" not in edges.columns:
            edges["source"] = "manual"
    else:
        edges = auto_edges(sites)
        print("Wrote edges.csv (rerun with --edges edges.csv after editing to manually set edges)")

    # Changes how to save the edges if user wants guessed edges or not
    if args.isolated == "connect":
        edges = connect_isolated(sites, edges)
    elif args.isolated == "drop":
        sites = sites.loc[sites.index.isin(edges["site_a"]) | sites.index.isin(edges["site_b"])]
    edges.to_csv("edges.csv", index=False)

    # Writing the test case and plotting the graph
    sites.assign(roads=sites["roads"].map(" | ".join)).to_csv("sites.csv")
    plot_graph("graph.png", sites, edges)
    write_testcase(OUTPUT, sites, edges)

    # Final output text
    isolated = sorted(set(sites.index) - set(edges["site_a"]) - set(edges["site_b"]))
    print(f"{len(sites)} sites, {len(edges)} edges (each written in both directions)")
    if isolated:
        print(f"Sites with no edges: {isolated}")
    print(f"Wrote sites.csv, graph.png, {OUTPUT}")