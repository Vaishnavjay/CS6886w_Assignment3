# CS6886W – Assignment 3: MobileNetV2 Compression on CIFAR-10

This repository contains my implementation for **CS6886W – Systems Engineering for Deep Learning, Assignment 3**.  
The goal is to train a baseline MobileNetV2 model on CIFAR-10 and then apply a custom, configurable quantization-based compression pipeline to model **weights and activations**, and evaluate accuracy vs. compression trade-offs.

---

## 1. Repository Structure

```text
.
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
    └── CS6886W_Assignment3_Vaishnav.ipynb   # (optional) original Colab notebook
