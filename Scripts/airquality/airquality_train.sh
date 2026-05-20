#!/bin/bash
LD_LIBRARY_PATH= python main.py --name airquality_24 --config_file Config/airquality.yaml --gpu 1 --train
# LD_LIBRARY_PATH= python main.py --name airquality_48 --config_file Config/airquality.yaml --gpu 1 --train
# LD_LIBRARY_PATH= python main.py --name airquality_96 --config_file Config/airquality.yaml --gpu 1 --train
