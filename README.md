# DSG-TS

Official implementation of "Interpretable Multi-Granularity Time-Series Generation via Decomposed Semantic Guidance"

## Project Structure

- `main.py` - Main training and generation entry script
- `evaluation.py` - Evaluation script for generated outputs 
- `requirements.txt` - Python dependencies
- `Config/` - Dataset and model configuration files
- `Data/` - Data loader and dataset definitions
- `engine/` - Core engine modules for training, logging, and learning rate scheduling
- `Models/interpretable_diffusion/` - Diffusion model implementation
- `Scripts/` - Dataset-specific training/testing shell scripts
- `Semantic_Decomposition/` - Semantic decomposition analysis scripts
- `Utils/` - Utility modules and dataset helpers

## Requirements

Recommended Python environment: Python 3.10+ or compatible.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Download the dataset from https://huggingface.co/datasets/WinfredGe/TSFragment-600K and process it using the scripts under `Semantic_Decomposition` to generate the dataset required by this project.

Place the processed dataset under `Data/datasets/`, and make sure the 'data_root' paths in the configuration files point to this directory.

> Note: Dataset file names and formats should match the 'name' parameter specified in the corresponding 'Config/*.yaml' file.

## Usage

### Training

Run training with `main.py`:

```bash
python main.py --name <experiment_seqlen> --config_file Config/<dataset>.yaml --gpu <gpu_id> --train
```

Example:

```bash
python main.py --name electricity_24 --config_file Config/electricity.yaml --gpu 0 --train
```

### Generation / Inference

Without `--train`, `main.py` runs in generation mode:

```bash
python main.py --name <experiment_name> --config_file Config/<dataset>.yaml --gpu <gpu_id>
```

If you want to sample multiple times and save multiple outputs, enable `--run_multi`:

```bash
python main.py --name <experiment_name> --config_file Config/<dataset>.yaml --gpu 0 --run_multi True
```

Generated results are saved under `OUTPUT/<experiment_name>/run/run_<i>/` as `.npy` files.

## Configuration

Configurations are stored in `Config/`. Each dataset has a YAML file defining:

- `model` - model architecture and hyperparameters
- `solver` - optimizer, scheduler, training settings, and checkpoint saving
- `dataloader` - train/test dataset settings and batch sizes

For example, `Config/electricity.yaml` defines sequence length, model depth, learning rate, epochs, and other key settings for the electricity dataset.

## Evaluation

Use `evaluation.py` to evaluate generated outputs:

```bash
python evaluation.py --true_path <true_data.npy> --gen_path <generated_data.npy> --gen_root <run_multi_root> --gen_name <generated_filename> --save_dir ./evaluation_results
```

Example:

```bash
python evaluation.py --true_path ./Data/datasets/electricity_24_test.npy \
  --gen_path OUTPUT/electricity_24/ddpm_fake_electricity_24.npy \
  --gen_root OUTPUT/electricity_24/run \
  --gen_name ddpm_fake_electricity_24.npy \
  --save_dir ./evaluation_results
```

Results are saved as JSON files.

## Available Dataset Configurations

Current configuration files include:

- `Config/airquality.yaml`
- `Config/electricity.yaml`
- `Config/etth.yaml`
- `Config/ettm.yaml`
- `Config/exchangerate.yaml`
- `Config/traffic.yaml`


## Notes

- Use `--gpu` to specify the GPU device if available.
- Training checkpoints and outputs are saved to the directory set by `solver.results_folder`.
- In generation mode `main.py` loads the model and calls `Trainer.sample()` to produce outputs.

## Extension

To add new datasets or model variants:

1. Add a new YAML configuration file under `Config/`
2. Add the corresponding dataset files under `Data/datasets/`
3. Extend model or dataset classes in `Models/interpretable_diffusion/` or `Utils/Data_utils/real_datasets.py`

