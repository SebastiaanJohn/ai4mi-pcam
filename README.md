# AI4MI PCAM Project
TODO Project description here.


# Project Structure
The repository is organized as follows:

<!-- Created with https://tree.nathanfriend.io/ -->
<pre>
.
├── ℹ️ <b>README.md</b>: Project description and other relevant details.
├── ❗️ <b>environment.yml</b>: Conda environment configuration.
├── 📄 <b>requirements.txt</b>: Python dependencies for the project.
├── ⚙️ <b>pyproject.toml</b>: Configuration for the ruff tool, a linting and code formatting utility.
├── 📁 <b>data</b>: Directory with raw and processed data.
│   ├── 📁 <b>raw</b>: Raw dataset files, such as those related to the camelyonpatch level 2 split.
│   └── 📁 <b>processed</b>: Processed data for the project.
├── 📁 <b>src</b>: Main source code directory, which encompasses:
│   ├── 🐍 <b>main.py</b>: Main entry point for the project.
│   ├── 📁 <b>callbacks</b>: Callback functions used during training (e.g., `wandb_callback.py`).
│   ├── 📁 <b>data</b>: Data processing scripts like `make_dataset.py`.
│   ├── 📁 <b>datasets</b>: Handling of specific datasets, including:
│   │   └── 📁 <b>pcam</b>: Scripts for the PCAM (Patch Camelyon) dataset, including `datamodule.py` and `dataset.py`.
│   ├── 📁 <b>modules</b>: Diverse modules utilized in the project:
│   │   ├── 🐍 <b>interface.py</b>: Defines common interfaces.
│   │   ├── 📁 <b>metrics</b>: Metric calculation scripts (e.g., `metrics.py`).
│   │   └── 📁 <b>models</b>: Model definitions, such as `simple_cnn.py`.
│   └── 📁 <b>utils</b>: Utility scripts and functions used throughout the project.
├── 📁 <b>experiments</b>: Directory for experiment scripts and their results.
├── 📁 <b>models</b>: Storage for trained model files.
└── 📁 <b>notebooks</b>: Jupyter notebooks for the project.
</pre>


# Requirements
One of our goals is to make the project as easy to set up as possible. To this end, we took the greatest care in automating every step, from downloading and preprocessing the dataset to running the training and evaluation code. The following guide is written to be as concise and easy to follow as possible.

## Environment
To set up the correct environment, we recommend downloading the [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://github.com/mamba-org/mamba) (a drop-in fast replacement for Conda) package manager. Once installed, create a new environment with the following commands:
```bash
conda env create -f ai4mi-pcam
```

Be aware that this can take a while (depending on the hardware and network speed, around 10 to 30 minutes). Once the environment is created, activate it with:
```bash
conda activate ai4mi-pcam
```

## Dataset
We recommend obtaining the dataset using `setup_pcam.sh`, which will automatically download, extract, and pre-process the data, putting the (intermediate) results in the `data` folder. Alternatively, you can download the datasets manually via PCam's aforementioned GitHub website and/or pre-process the data yourself.

### PCam
The dataset used in this project is the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) benchmark. It contains around 327k small images extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue. To download PCam and preprocess the dataset, run:
```bash
sh ./data/setup_pcam.sh
```
Around 8 GB of data will be downloaded and processed.


## Getting started

### Usage
Train the model with:
```bash
python3 TODO
```

Calculate the metrics with:
```bash
python3 TODO
```

### Examples
For training TODO with TODO hyperparameters, run:
```bash
python3 TODO --TODO TODO etc.
```
Each epoch takes around TODO minutes on a single A100 GPU, and TODO minutes on a Titan RTX GPU. The trained model will be saved to the `models` folder.

To calculate the metrics for the trained model, run:
```bash
python3 TODO --TODO TODO etc.
```
This will calculate the metrics for the model saved in `models` and save the results in the `experiments` folder.
