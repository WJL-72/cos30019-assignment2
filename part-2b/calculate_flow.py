"""
calculate_flow.py - turn the long-format SCATS file into one clean flow series per SCATS site

Input: long_format.csv   (output of wide_to_long.py; one row per location / day / 15-min slot)
Output: site_flows.csv      (one row per site/day/slot, 40 sites x 31 days x 96 slots)

Use:
    python calculate_flow.py                                  # uses the default paths after the import lines
    python calculate_flow.py <input.csv> --out <output.csv>

The 'flow' column are vehicles per 15 minutes, summed over all approaches of a site
'n_approaches' is how many approaches were summed

Note: the Canvas formula (capacity 1500 veh/h) has no valid speed above 1500 veh/h.
Summed site flow x 4 exceeds that in the readings to some part
Question is whether 'flow' (sum) or 'flow' / 'n_approaches' is the quantity fed to the formula in Canvas
"""
import argparse
import pandas as pd

DEFAULT_IN = "data/processed/long_format.csv"
DEFAULT_OUT = "data/processed/site_flows.csv"

MIN_DAYS = 20        # approaches with fewer days of data than this is dropped (e.g. site 3001 has one with 2 days)
ZERO_RUN_SLOTS = 4   # this much all-zero slots in a row (= 1 hour) will be treated as a sensor/clock gap
SITE_DAY = ["scats_num", "date"]


def load(path: str) -> pd.DataFrame:
    # Site numbers are read as text, then made integers
    # Done to avoid any potential mismatches
    df = pd.read_csv(path, dtype={"scats_num": str})
    df["scats_num"] = df["scats_num"].astype(int)
    return df


def merge_detectors(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["scats_num", "location_name", "date", "time", "interval"], as_index=False)["volume"].sum()


def drop_sparse_locations(df: pd.DataFrame) -> pd.DataFrame:
    # Count how many different days each approach has, and keep only the significant ones (majority)
    days = df.groupby(["scats_num", "location_name"])["date"].transform("nunique")
    return df[days >= MIN_DAYS]


def site_totals(df: pd.DataFrame) -> pd.DataFrame:
    # Sum the approaches of each site into one flow per slot, and mark which site-days are complete
    # A site-day is complete only if every approach of that site reported that day
    # otherwise the sum would be too low and look like a quiet road
    expected = df.groupby("scats_num")["location_name"].nunique()
    present = df.groupby(SITE_DAY)["location_name"].nunique().rename("present").reset_index()
    present["complete"] = present["present"].values == expected.reindex(present["scats_num"]).values

    totals = df.groupby(["scats_num", "date", "interval", "time"], as_index=False).agg(
        flow=("volume", "sum"))
    return totals.merge(present[SITE_DAY + ["complete"]], on=SITE_DAY)


def full_grid(totals: pd.DataFrame) -> pd.DataFrame:
    # Build every possible site/day/slot combination so a missing day shows up as an empty row
    slots = totals[["interval", "time"]].drop_duplicates()
    index = pd.MultiIndex.from_product(
        [sorted(totals["scats_num"].unique()), sorted(totals["date"].unique()), sorted(slots["interval"])],
        names=["scats_num", "date", "interval"])
    grid = index.to_frame(index=False).merge(slots, on="interval")
    grid = grid.merge(totals, on=["scats_num", "date", "interval", "time"], how="left")
    grid["complete"] = grid["complete"].fillna(False).astype(bool)
    return grid.sort_values(["scats_num", "date", "interval"], ignore_index=True)


def zero_run_length(flow: pd.Series) -> pd.Series:
    # Run-length: a new run starts whenever "is zero" changes, so cumsum() of the changes
    # gives every run its own label, and summing the zero flags per label gives the run's length
    # (Non-zero runs sum to 0) NaN is not zero, so empty readings never join a zero run
    is_zero = flow.eq(0)
    run_id = (is_zero != is_zero.shift()).cumsum()
    return is_zero.groupby(run_id).transform("sum")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_sparse_locations(merge_detectors(df))
    grid = full_grid(site_totals(df))

    # Rule 1: incomplete site-days are unusable
    grid.loc[~grid["complete"], "flow"] = float("nan")
    # Rule 2: long all-zero runs inside a complete day are considered outages. Runs are measured across midnight per site
    runs = grid.groupby("scats_num")["flow"].transform(zero_run_length)
    grid["zero_gap"] = runs >= ZERO_RUN_SLOTS
    grid.loc[grid["zero_gap"], "flow"] = float("nan")
    # Number of approaches summed into each site's flow. Divide flow by this to get the average flow per
    # approach, which is the scale the Canvas flow-to-speed formula was designed for (see note at the top).
    grid["n_approaches"] = grid["scats_num"].map(df.groupby("scats_num")["location_name"].nunique())
    return grid[["scats_num", "date", "time", "interval", "flow", "n_approaches"]].assign(
        complete=grid["complete"], zero_gap=grid["zero_gap"])


def summarise(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    per_day = out.groupby(SITE_DAY).agg(complete=("complete", "first"))
    print(f"input rows            : {len(raw):,}")
    print(f"rows merged           : {len(raw) - len(merge_detectors(raw)):,}")
    print(f"output rows           : {len(out):,}  ({out['scats_num'].nunique()} sites x "
          f"{out['date'].nunique()} days x {out['interval'].nunique()} slots)")
    print(f"complete site-days    : {int(per_day['complete'].sum()):,} of {len(per_day):,}")
    print(f"zero-gap slots blanked: {int((out['zero_gap'] & out['complete']).sum()):,}")
    print(f"empty flow readings   : {int(out['flow'].isna().sum()):,} of {len(out):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean SCATS data into one flow series per site")
    ap.add_argument("input", nargs="?", default=DEFAULT_IN, help="long-format CSV from wide_to_long.py")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write site_flows.csv")
    args = ap.parse_args()

    raw = load(args.input)
    result = clean(raw)
    result.to_csv(args.out, index=False)
    summarise(raw, result)
    print(f"Wrote {args.out}")





# NOTES TO SELF
# ======================================================================================
# ok so. the raw file is: one row per APPROACH (like "WARRIGAL_RD N of HIGH ST"),
# per day, per 15 min. but the route graph only cares about SITES (intersections)
# so this script's whole job = take the approach-level stuff and spit out ONE clean
# flow number per site per 15 min, with the garbage left EMPTY (NaN) instead of guessed
#
# what goes in: long_format.csv (leader's long-format file)
# what comes out: site_flows.csv -> 40 sites x 31 days x 96 slots = 119,040 rows
#
# the steps, in order, and why:
#
# 1. load(): site numbers become integers. "0970" -> 970. sites.csv and edges.csv use ints
#    and joining text to ints just returns NOTHING with no error
#
# 2. drop_duplicates(): site 4335 has 31 location-days that appear twice (identical copies)
#    keep them the sum doubles for that site? gone.
#
# 3. drop_sparse_locations(): one approach (site 3001, CHURCH_ST SW of BARKERS_RD) only has
#    2 days of data. keep it and site 3001 looks "incomplete" almost every day. so anything
#    under 20 days gets dropped. judgement call, could be changed
#
# 4. site_totals(): add up all the approaches of a site per slot. BUT only trust a site-day if
#    EVERY approach of that site reported that day. otherwise summing 3 of 4 approaches looks
#    like a quiet road when it's really just a missing sensor
#
# 5. full_grid(): build every site/day/slot combo, then paste the data onto it. why? because
#    if a day is missing the rows just vanish and the time series just silently jumps
#    dont think that would be great for the training
#
# 6. clean() rule 1: incomplete site-day -> flow = NaN
#    clean() rule 2: 4+ zeros in a row (1 hour) at site level = sensor/clock gap, not traffic -> NaN
#    I checked the data: the ONLY runs like that are 23 sites on 29 Oct, 01:45-02:30, same hour everywhere
#    it COULD be daylight saving clock change (idk, im only guessing)
#    short zeros (1-3 slots) stay, those could be real idk
#
# the run-length trick (zero_run_length), because ts just magic:
#    is_zero      : F T T F T T T T F
#    change marker: a new run starts wherever is_zero flips
#    cumsum of it : 1 2 2 3 4 4 4 4 5   <- every run gets its own id
#    sum zeros per id -> run length: 0 2 2 0 4 4 4 4 0  -> the 4-run gets flagged
#    NaN counts as "not zero" so it never joins a run
#
# 7. n_approaches: how many approaches got summed into each site's flow (1 to 6).
#
# ======================================================================================
# DONT FORGET THESE
# ======================================================================================
# - flow here = cars per 15 MIN. the Canvas formula wants cars per HOUR. so x4 before using it.
#   (that happens in the travel-time step, not here.)
# - Canvas formula: flow = -1.4648375*speed^2 + 93.75*speed
#   top of the parabola = capacity:
#   1500 veh/h at 32 km/h. at or below ~351 veh/h speed is capped at 60 km/h.
#   above 1500 there is NO valid speed (negative under a square root)
# - PROBLEM: summed site flow x4 goes over 1500 in ~43% of readings. per-approach average
#   (flow / n_approaches) does it in 0.08%. so the formula was clearly built for per-approach
#   scale. which one we use = ask dr joel ig
# - empty (NaN) readings are NOT filled in. whoever makes the training windows must skip any
#   window that touches a NaN. 5,272 of 119,040 readings are empty
# - NO scaling in this file on purpose. scale using train days only or test info leaks in
# - the test split (27-31 Oct) contains 29 Oct, the clock-gap day. no idea if readings after
#   that hour are shifted by an hour too. should mention that

# if i could put a reaction gif in here i wpuld