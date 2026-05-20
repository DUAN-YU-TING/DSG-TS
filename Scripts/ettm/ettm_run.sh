#!/bin/bash
python main.py --name ettm_24 --config_file Config/ettm.yaml --gpu 1 --milestone 10
python main.py --name ettm_48 --config_file Config/ettm.yaml --gpu 1 --milestone 10
python main.py --name ettm_96 --config_file Config/ettm.yaml --gpu 1 --milestone 10
