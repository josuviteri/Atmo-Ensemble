import kagglehub
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment", output_dir="data")

print("Path to dataset files:", path)

shutil.rmtree("data/.complete")
