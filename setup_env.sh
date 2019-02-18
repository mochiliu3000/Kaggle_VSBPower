#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "Use $0 <KAGGLE_USERNAME> <KAGGLE_KEY>"
  exit 1
fi

export KAGGLE_USERNAME=$1
export KAGGLE_KEY=$2
pip3 install --upgrade numpy pandas numba tensorflow keras tqdm matplotlib seaborn kaggle 
kaggle competitions download -c vsb-power-line-fault-detection
