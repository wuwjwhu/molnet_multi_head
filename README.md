# Multi-Property Prediction of Drug Molecules Based on Graph Neural Networks

## Overview
This repository contains code for training Graph Neural Networks to predict multiple properties of drug molecules. It supports both classification and regression tasks, and provides functionality for pre-training, finetuning, and transfer learning.

## Prerequisites
Ensure you have the following software installed:
* Python == 3.8
* [PyTorch](https://pytorch.org/) == 2.0.1
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) == 2.3.1
* [RDKit](https://www.rdkit.org/) == 2023.3.3

You can also install all required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Data Preparation
### Data Preprocessing
To preprocess the dataset for training, use the following command:
```bash
python dataset_spliter_merger.py
```

## Model Training
### Single-Property Model Pre-training
Train models for individual properties using:
* Classification tasks:
```bash
python single_cls.py --config config/single/{task}.yaml  # Replace {task} with the specific task name
```
* Regression tasks:
```bash
python single_reg.py --config config/single/{task}.yaml
```

### Filling the Dataset
To augment the merged training set with predictions from the single-property model, run:
```bash
python dataset_filler_marker.py
```

## Multi-Property Model Training
### Pre-training
For pre-training the multi-property model:
```bash
python multi_pretrain.py --config config/multi/pretrain.yaml
```
Check the dataset performance from the single-property models specified in pretrain.yaml.

### Finetuning
Finetune the pre-trained multi-property model on specific tasks using:
* Classification tasks:
```bash
python multi_finetune_cls.py --config config/multi/{task}.yaml
```
* Regression tasks:
```bash
python multi_finetune_reg.py --config config/multi/{task}.yaml
```
Check the pretrained model path from saved models specified in pretrain.yaml.

### Transfer Learning
Transfer learned models to new tasks:
* Classification tasks:
```bash
python multi_transfer_cls.py --config config/multi/{task}.yaml
```
* Regression tasks:
```bash
python multi_transfer_reg.py --config config/multi/{task}.yaml
```
