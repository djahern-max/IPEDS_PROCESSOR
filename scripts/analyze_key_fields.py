# scripts/analyze_key_fields.py
import pandas as pd

try:
    hd = pd.read_csv("raw_data/hd2023.csv", encoding="utf-8")
except UnicodeDecodeError:
    hd = pd.read_csv("raw_data/hd2023.csv", encoding="latin-1")

print("Key Field Analysis:")
print("\nCONTROL (Institution Control):")
print(hd["CONTROL"].value_counts().sort_index())

print("\nICLEVEL (Institutional Level):")
print(hd["ICLEVEL"].value_counts().sort_index())

print("\nSECTOR (Combined classification):")
print(hd["SECTOR"].value_counts().sort_index())

print("\nSample institutions by type:")
for control in hd["CONTROL"].unique():
    sample = hd[hd["CONTROL"] == control][["INSTNM", "CITY", "STABBR"]].head(2)
    print(f"\nCONTROL {control}:")
    print(sample)
