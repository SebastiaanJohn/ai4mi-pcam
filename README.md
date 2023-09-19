# AI4MI PCAM Project

## Project Description

## Project Structure

The repository is organized as follows:

- **README.md**: Project description and other relevant details.
- **data**: Directory with raw and processed data.
  - **processed**: Processed data for the project.
  - **raw**: Raw dataset files, such as those related to the camelyonpatch level 2 split.
- **environment.yml**: Conda environment configuration.
- **experiments**: Directory for experiment scripts and their results.
- **models**: Storage for trained model files.
- **notebooks**: Jupyter notebooks for the project.
- **pyproject.toml**: Configuration for the ruff tool, a linting and code formatting utility.
- **requirements.txt**: Python dependencies for the project.
- **src**: Main source code directory, which encompasses:
  - **callbacks**: Callback functions used during training (e.g., `wandb_callback.py`).
  - **data**: Data processing scripts like `make_dataset.py`.
  - **datasets**: Handling of specific datasets, including:
    - **pcam**: Scripts for the PCAM (Patch Camelyon) dataset, including `datamodule.py` and `dataset.py`.
  - **main.py**: Main entry point for the project.
  - **modules**: Diverse modules utilized in the project:
    - **interface.py**: Defines common interfaces.
    - **metrics**: Metric calculation scripts (e.g., `metrics.py`).
    - **models**: Model definitions, such as `simple_cnn.py`.
  - **utils**: Utility scripts and functions used throughout the project.