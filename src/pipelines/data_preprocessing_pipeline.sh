#!/bin/bash

root_dir=curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

cd ${root_dir}

mkdir ./data/

mkdir ./models/

python ./src/data_processing/download_data.py

python ./src/data_processing/create_synthetic_names.py

python ./src/data_processing/process_initial_dataset.py

python ./src/data_processing/create_dataset.py
