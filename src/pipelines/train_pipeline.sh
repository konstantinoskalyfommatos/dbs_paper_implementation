#!/bin/bash

root_dir=curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

cd ${root_dir}

python ./src/retrieval_component/scripts/finetune_dpr.py

python ./src/retrieval_component/scripts/bart_finetune.py
