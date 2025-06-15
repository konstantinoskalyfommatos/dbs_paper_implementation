#!/bin/bash


root_dir=curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

cd ${root_dir}

python ./src/retrieval_component/scripts/index_qdrant.py

python ./src/retrieval_component/scripts/retrieve_relevant_documents.py

python ./src/augmentation_component/predict_llm.py

python ./src/evaluation/calculate_metrics.py
