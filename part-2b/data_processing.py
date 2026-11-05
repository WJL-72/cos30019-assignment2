import argparse
import pandas as pd

OUTPUT = 'processed_data.csv'
VAL_DAYS = 4
TEST_DAYS = 5
DAILY_INTERVALS = 96
INTERVAL_MIN = 15
INTERVAL_COLS = [f"V{i:02d}" for i in range(DAILY_INTERVALS)]

def convert(path: str) -> pd.DataFrame:
    # 'header=1' lets the program know the header is on 2nd line, SCATS number saved as string so leading 0 remain
    raw = pd.read_excel(path, sheet_name="Data", header=1, dtype={"SCATS Number": str})
    raw["SCATS Number"] = raw["SCATS Number"].str.strip()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize() # Keeps the date only

    keys = ["SCATS Number", "Location", "Date"] # Columns to keep
    long = raw.melt(id_vars=keys, value_vars=INTERVAL_COLS, var_name="interval", value_name="volume") # Converts from wide to long format
    long["slot"] = long["interval"].str[1:].astype(int) # Changes the interval from 'V01' to '1', possibly easier to work with

    minutes = long["slot"] * INTERVAL_MIN # Start of the interval in minutes, begins at midnight
    long["time"] = (minutes // 60).astype(str).str.zfill(2) + ":" + (minutes % 60).astype(str).str.zfill(2) # Creating the time column

    # Order: location -> day -> interval
    long = long.sort_values(["SCATS Number", "Location", "Date", "slot"], kind="stable")
    long["Date"] = long["Date"].dt.strftime("%Y-%m-%d") # Converts date into YYYY-MM-DD

    # Provides the 
    out = long[["SCATS Number", "Location", "Date", "time", "interval", "volume"]]
    out.columns = ["scats_num", "location_name", "date", "time", "interval", "volume"]
    return out.reset_index(drop=True) # Renumbers the rows after sorting

def write_splits(df: pd.DataFrame, prefix: str):
    # Used for optional argument to split the dataset
    days = sorted(df["date"].unique())
    test_set = set(days[-TEST_DAYS:]) # Days in test set
    val_set = set(days[-(TEST_DAYS + VAL_DAYS):-TEST_DAYS]) # Days in validation set

    # Picks the rows corresponding to each part
    parts = {"train": df[~df["date"].isin(test_set | val_set)],
             "val": df[df["date"].isin(val_set)],
             "test": df[df["date"].isin(test_set)]}

    # Prints out output to console according to the dataset type and the date range
    for name, part in parts.items():
        part.to_csv(f"{prefix}_{name}.csv", index=False)
        print(f"  {prefix}_{name}.csv: {len(part):,} rows ({part['date'].min()} to {part['date'].max()})")


if __name__ == "__main__":
    # Handling the arguments passed through the command prompt (file name and '--split' to split into different sets)
    ap = argparse.ArgumentParser(description="Convert SCATS October 2006 spreadsheet to a long-format CSV")
    ap.add_argument("xls", help="path to Scats Data October 2006.xls")
    ap.add_argument("--split", action="store_true", help="also write scats_long_train/_val/_test.csv by date")
    args = ap.parse_args()

    # Runs the conversion and saves to CSV while printing summary to terminal
    df = convert(args.xls)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}: {len(df):,} rows, {df['scats_num'].nunique()} sites, "
          f"{df['location_name'].nunique()} locations, {df['date'].nunique()} days")
    if args.split:
        write_splits(df, OUTPUT.removesuffix(".csv"))