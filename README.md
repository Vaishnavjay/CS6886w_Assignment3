# CS6886W – Assignment 3: MobileNetV2 Compression on CIFAR-10

This repository contains my implementation for **CS6886W – Systems Engineering for Deep Learning, Assignment 3**.  
The goal is to train a baseline MobileNetV2 model on CIFAR-10 and then apply a custom, configurable quantization-based compression pipeline to model **weights and activations**, and evaluate accuracy vs. compression trade-offs.

---

## 1. Repository Structure

├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data.py           # CIFAR-10 dataloaders + preprocessing
│   ├── model.py          # MobileNetV2CIFAR10 definition
│   ├── quantization.py   # QuantConfig, QuantAct, weight quantization utilities
│   ├── utils.py          # set_seed, checkpoint save/load, logging helpers
│   ├── train.py          # Baseline MobileNetV2 training script (Q1)
│   └── compress_eval.py  # Compression + evaluation + W&B logging (Q2, Q3, Q4)
└── notebooks
    └── CS6886W_Assignment3_Vaishnav.ipynb   original Colab notebook

## 2. Environment & Dependencies

This code was developed and tested on Google Colab (Python 3.12, CUDA 12) with:
	•	torch==2.9.0+cu126
	•	torchvision==0.24.0+cu126
	•	torchaudio==2.9.0+cu126
	•	numpy==2.0.2
	•	matplotlib==3.10.0
	•	wandb==0.23.1

All required packages (plus a few Colab defaults) are pinned in requirements.txt.

2.1. Install dependencies

From the repository root:


