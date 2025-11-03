import os

root = "data/processed/bzd"

for root, dirs, files in os.walk(root):
    print(f"cartella:{root}")
    print(f"sottocartella:{dirs}")