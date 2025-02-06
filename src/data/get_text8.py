import requests
import os

data_path = "src/data/text8.txt"

def download_text8():
    """Download the text8 dataset if it doesn't exist."""
    if not os.path.exists(data_path):
        print("Downloading text8 dataset...")
        r = requests.get("https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8")
        with open(data_path, "wb") as f:
            f.write(r.content)
        print("Download complete!")
    else:
        print("text8 file already exists")

    # Read and return the data
    with open(data_path) as f:
        return f.read()

