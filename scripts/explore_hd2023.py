import pandas as pd

# Load the main directory file with proper encoding
try:
    hd = pd.read_csv("raw_data/hd2023.csv", encoding="utf-8")
except UnicodeDecodeError:
    # Try with latin-1 encoding if utf-8 fails
    hd = pd.read_csv("raw_data/hd2023.csv", encoding="latin-1")

print("HD2023 Basic Info:")
print(f"Total institutions: {len(hd)}")
print(f"Total columns: {len(hd.columns)}")
print("\nFirst 5 institutions:")
print(hd[["UNITID", "INSTNM", "CITY", "STABBR"]].head())

print("\nColumn names:")
print(list(hd.columns))
