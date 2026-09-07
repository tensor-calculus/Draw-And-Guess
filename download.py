import urllib.request
import urllib.parse
import os

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt",
    "categories.txt"
)

os.makedirs("quickdraw_data", exist_ok=True)

with open("categories.txt", encoding="utf-8") as f:
    categories = [line.strip() for line in f]

base = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

for category in categories:
    url = base + urllib.parse.quote(category) + ".ndjson"
    filename = os.path.join("quickdraw_data", category + ".ndjson")
    print("Downloading", category)
    urllib.request.urlretrieve(url, filename)
