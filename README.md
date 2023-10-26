# Performance and Interpretability of Deep Learning Models in Breast Tumor Identification

Project for the course AI for Medical Imaging at the University of Amsterdam for the MSc Artificial Intelligence.

## Abstract

This research delves into interpretability techniques to understand and analyze deep learning models trained on the PatchCamelyon (PCam) dataset for detecting metastatic tissue in breast cancer patients. Three interpretability methods, namely saliency mapping, integrated gradients, and LIME, were used to generate heatmaps representing what several models focused on. The main objective of our work is to investigate whether models with similar performance prioritize the same regions during their predictions. We use Intersection over Union (IoU) as the primary evaluation metric to compare the heat maps of different model pairs. We average the IoU score across multiple images to understand the models' performance comprehensively.


## Project Structure

The repository is organized as follows:

<pre>
.
├── ℹ️ <b>README.md</b>: Project description and other relevant details.
├── ❗️ <b>environment.yml</b>: Conda environment configuration.
├── 📄 <b>requirements.txt</b>: Python dependencies for the project.
├── ⚙️ <b>pyproject.toml</b>: Configuration for the ruff tool, a linting and code formatting utility.
├── 📁 <b>data</b>: Directory with raw and processed data.
│   ├── 📁 <b>raw</b>: Raw dataset files, such as those related to the camelyonpatch level 2 split.
│   │   └── 📄 <b>setup_pcam.sh</b>: Script to set up the PCAM dataset.
│   └── 📁 <b>processed</b>: Processed data for the project.
│       └── 📁 <b>pcam</b>: Directory for the PCAM dataset, including metadata and test data.
├── 📁 <b>src</b>: Main source code directory, which encompasses:
│   ├── 📁 <b>datasets</b>: Handling of specific datasets, including:
│   │   └── 📁 <b>pcam</b>: Scripts for the PCAM (Patch Camelyon) dataset, including `datamodule.py` and `dataset.py`.
│   ├── 📁 <b>engines</b>: Engine scripts for training and testing models.
│   ├── 📁 <b>models</b>: Model definitions, including various neural network architectures.
│   ├── 📁 <b>scripts</b>: Scripts for training and testing models.
│   ├── 📁 <b>utils</b>: Utility scripts and functions used throughout the project.
│   └── 📁 <b>xai</b>: Explainable AI (XAI) scripts for interpreting model predictions.
├── 📁 <b>experiments</b>: Directory for experiment scripts and their results.
├── 📁 <b>models</b>: Storage for trained model files.
└── 📁 <b>notebooks</b>: Jupyter notebooks for the project.
</pre>


## Requirements

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
python3 -m src.scripts.train --model resnet_34 --pretrained --freeze --batch_size 64 --num_workers 8
```

Run `python3 -m src.scripts.train --help` to see all possible arguments.

The models we trained can be downloaded from [here](https://drive.google.com/drive/folders/1lYijb7mw1lnOqntwWt3dNgyCyu_NzpEB?usp=share_link). The models should be placed in the `models` folder under the respective model name, e.g. `models/ResNet34/_.ckpt`.

Test the model with:

```bash
python3 -m src.scripts.test --model resnet_34 --version 0
```

Run `python3 -m src.scripts.test --help` to see all possible arguments.

Run the XAI methods with:

```bash
python3 -m src.xai.xai --logging_level INFO --models resnet34 resnet50 densenet121 vit_b_16 alexnet vgg11 efficientnet --explanation saliency_mapping --similarity_measure thresholded_iou --num_images 256 --batch_size 8 --num_workers 1
```

```bash
python3 -m src.xai.xai --logging_level INFO --models resnet34 resnet50 densenet121 vit_b_16 alexnet vgg11 efficientnet --explanation integrated_gradients --similarity_measure thresholded_iou --num_images 256 --batch_size 8 --num_workers 1
```

```bash
python3 -m src.xai.xai --logging_level INFO --models resnet34 resnet50 densenet121 vit_b_16 alexnet vgg11 efficientnet --explanation lime --similarity_measure thresholded_iou --num_images 256 --batch_size 8 --num_workers 1
```
