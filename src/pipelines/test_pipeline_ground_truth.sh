#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python ./src/augmentation_component/predict_llm.py

python ./src/evaluation/calculate_metrics.py --ground_truth
