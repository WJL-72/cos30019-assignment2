import sys
from pathlib import Path
import pandas as pd
 
RAW_FILE = Path("data/raw/Scats Data October 2006.xls") # Input data file
OUTPUT = Path("data/processed/processed_data.csv") # Output file
DAILY_INTERVALS = 96
INTERVAL_MIN = 15
INTERVAL_COLS = [f"V{i:02d}" for i in range(DAILY_INTERVALS)]
 
# Converts the spreadsheet into a long table. Raises ValueError if anything goes wrong
def convert(path) -> pd.DataFrame:
    # 'header=1' lets the program know the header is on 2nd line, SCATS number saved as string so leading 0 remain
    raw = pd.read_excel(path, sheet_name="Data", header=1, dtype={"SCATS Number": str})
    keys = ["SCATS Number", "Location", "Date"] # Columns to keep
    missing = set(keys + INTERVAL_COLS) - set(raw.columns) # Finds any column names not in the file

    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    raw["SCATS Number"] = raw["SCATS Number"].str.strip()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize() # Keeps the date only
 
    long = raw.melt(id_vars=keys, value_vars=INTERVAL_COLS, var_name="interval", value_name="volume") # Converts from wide to long format
    long["slot"] = long["interval"].str[1:].astype(int) # Changes the interval from 'V01' to '1', possibly easier to work with
 
    minutes = long["slot"] * INTERVAL_MIN # Start of the interval in minutes, begins at midnight
    long["time"] = (minutes // 60).astype(str).str.zfill(2) + ":" + (minutes % 60).astype(str).str.zfill(2) # Creating the time column
 
    # Order: location -> day -> interval
    long = long.sort_values(["SCATS Number", "Location", "Date", "slot"], kind="stable")
    long["Date"] = long["Date"].dt.strftime("%Y-%m-%d") # Converts date into YYYY-MM-DD
 
    # Keeps only the columns needed and renames them
    out = long[["SCATS Number", "Location", "Date", "time", "interval", "volume"]]
    out.columns = ["scats_num", "location_name", "date", "time", "interval", "volume"]
    return out.reset_index(drop=True) # Renumbers the rows after sorting
 
# Main code block
def main():
    if len(sys.argv) > 1: # Takes no arguments
        print("Usage: python data_processing.py")
        print(f" Reads '{RAW_FILE}' and writes '{OUTPUT}', no argument is needed")
        sys.exit(1)
 
    # Error handling for loading the data file
    try:
        df = convert(RAW_FILE)
    except FileNotFoundError:
        print(f"Error: file '{RAW_FILE}' not found!")
        print(f"Place 'Scats Data October 2006.xls' in the folder '{RAW_FILE.parent}' before running")
        sys.exit(1)
    except Exception as exc:
        print(f"Error while reading '{RAW_FILE}': {exc}")
        sys.exit(1)
 
    # Saves to CSV, creating the folder if it does not exist
    try:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT, index=False)
    except OSError as exc:
        print(f"Error while writing '{OUTPUT}': {exc}")
        print("Close the file if it is open in another program, then try again")
        sys.exit(1)
 
    # Prints summary to terminal
    print(f"Wrote {OUTPUT}: {len(df):,} rows, {df['scats_num'].nunique()} sites, "
          f"{df['location_name'].nunique()} locations, {df['date'].nunique()} days")
 
if __name__ == "__main__":
    main()