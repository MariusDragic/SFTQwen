from datasets import Dataset
from pathlib import Path 

DATA_PATH = "./data/cnn_data.arrow"

ds = Dataset.from_file(DATA_PATH)

print("Dataset length:", len(ds))
print("\n Column names:", ds.column_names)
print("\n Example of article:", ds[1000])
print()
