#!/bin/bash

python3 train.py --data-path ./data/indist --proportion-of-labeled 1

./scripts/cleanup.sh