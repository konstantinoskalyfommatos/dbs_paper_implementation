import datasets
from pathlib import Path
import os

DATA_DIR = os.path.join(
    os.getcwd(),
    "data/downloaded_e2e_dataset"
)

os.makedirs(    
    DATA_DIR,
    exist_ok=True
)
datasets.config.DOWNLOADED_DATASETS_PATH = Path(DATA_DIR)

ds = datasets.load_dataset(
    path="GEM/e2e_nlg", 
    data_dir=DATA_DIR, 
    trust_remote_code=True
)