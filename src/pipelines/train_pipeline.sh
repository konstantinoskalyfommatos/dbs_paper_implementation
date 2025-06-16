#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python ./src/retrieval_component/scripts/finetune_dpr.py

python ./src/augmentation_component/bart_finetune.py
