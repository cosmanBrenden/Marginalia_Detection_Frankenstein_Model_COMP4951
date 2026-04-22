#!/bin/bash

python3 train.py --data-path ./data/comb --proportion-of-labeled 0.61

./scripts/cleanup.sh