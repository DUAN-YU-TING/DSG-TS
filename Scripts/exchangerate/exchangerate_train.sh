#!/bin/bash
LD_LIBRARY_PATH= python main.py --name exchangerate_24 --config_file Config/exchangerate.yaml --gpu 1 --train
# LD_LIBRARY_PATH= python main.py --name exchangerate_48 --config_file Config/exchangerate.yaml --gpu 1 --train
# LD_LIBRARY_PATH= python main.py --name exchangerate_96 --config_file Config/exchangerate.yaml --gpu 1 --train
