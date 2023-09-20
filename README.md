# AI4MI PCAM Project

## Project Description

## Project Structure

The repository is organized as follows:

<!-- Created with https://tree.nathanfriend.io/ -->
<pre>
.
├── ℹ️ <b>README.md</b>: Project description and other relevant details.
├── ❗️ <b>environment.yml</b>: Conda environment configuration.
├── 📄 <b>requirements.txt</b>: Python dependencies for the project.
├── ⚙️ <b>pyproject.toml</b>: Configuration for the ruff tool, a linting and code formatting utility.
├── 📁 <b>data</b>: Directory with raw and processed data.
│   ├── 📁 <b>processed</b>: Processed data for the project.
│   └── 📁 <b>raw</b>: Raw dataset files, such as those related to the camelyonpatch level 2 split.
├── 📁 <b>experiments</b>: Directory for experiment scripts and their results.
├── 📁 <b>models</b>: Storage for trained model files.
├── 📁 <b>notebooks</b>: Jupyter notebooks for the project.
└── 📁 <b>src</b>: Main source code directory, which encompasses:
    ├── 🐍 <b>main.py</b>: Main entry point for the project.
    ├── 📁 <b>callbacks</b>: Callback functions used during training (e.g., `wandb_callback.py`).
    ├── 📁 <b>data</b>: Data processing scripts like `make_dataset.py`.
    ├── 📁 <b>datasets</b>: Handling of specific datasets, including:
    │   └── 📁 <b>pcam</b>: Scripts for the PCAM (Patch Camelyon) dataset, including `datamodule.py` and `dataset.py`.
    ├── 📁 <b>modules</b>: Diverse modules utilized in the project:
    │   ├── 🐍 <b>interface.py</b>: Defines common interfaces.
    │   ├── 📁 <b>metrics</b>: Metric calculation scripts (e.g., `metrics.py`).
    │   └── 📁 <b>models</b>: Model definitions, such as `simple_cnn.py`.
    └── 📁 <b>utils</b>: Utility scripts and functions used throughout the project.
</pre>
