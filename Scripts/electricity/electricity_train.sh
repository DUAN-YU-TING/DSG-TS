#!/bin/bash
LD_LIBRARY_PATH= python main.py --name electricity_24 --config_file Config/electricity.yaml --gpu 1 --train
# LD_LIBRARY_PATH= python main.py --name electricity_48 --config_file Config/electricity.yaml --gpu 1 --train
# LD_LIBRARY_PATH= python main.py --name electricity_96 --config_file Config/electricity.yaml --gpu 1 --train
