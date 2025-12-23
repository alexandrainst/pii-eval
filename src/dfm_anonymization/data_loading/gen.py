
from pathlib import Path
import zipfile

def unpack_generated_datafiles():
    destination = Path("data/ground_truth")
    with zipfile.ZipFile("gen.zip", 'r') as zip_ref:
        zip_ref.extractall(destination)


if __name__ == "__main__":
    unpack_generated_datafiles()
