# CS6886W – Assignment 3: MobileNetV2 Compression on CIFAR-10

This repository contains my implementation for **CS6886W – Systems Engineering for Deep Learning, Assignment 3**.  
The goal is to train a baseline MobileNetV2 model on CIFAR-10 and then apply a custom, configurable quantization-based compression pipeline to model **weights and activations**, and evaluate accuracy vs. compression trade-offs.

---
## 1. Repository Structure
```text
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
    └── CS6886W_Assignment3_Vaishnav.ipynb   # Original Colab notebook
```
---
## 2. Environment & Dependencies
This code was developed and tested on Google Colab (Python 3.12, CUDA 12) with:
```text
	•	torch==2.9.0+cu126
	•	torchvision==0.24.0+cu126
	•	torchaudio==2.9.0+cu126
	•	numpy==2.0.2
	•	matplotlib==3.10.0
	•	wandb==0.23.1
```
All required packages (plus a few Colab defaults) are pinned in requirements.txt.

### 2.1. Install dependencies
From the repository root:
```text
pip install -r requirements.txt
```
On Google Colab, this is usually enough to match the environment used for the assignment.

---
## 3. Reproducibility & Seed Configuration

All experiments use a fixed random seed for reproducibility:
	•	Seed: 42

The following are seeded:
	•	Python random
	•	NumPy
	•	PyTorch (CPU + CUDA)

Seeding is done via set_seed(42) implemented in src/utils.py, and exposed to the scripts via --seed argument.

---
## 4. Training the Baseline (Question 1)

The baseline is a MobileNetV2 adapted for CIFAR-10:
	•	First conv stride changed from 2 → 1
	•	Width multiplier α = 1.0
	•	Dropout p = 0.2 before classifier
	•	SGD optimizer with momentum and MultiStepLR schedule

### 4.1. Command (full baseline run)

From the repo root, run:
```text
python -m src.train \
    --data-root ./data \
    --save-dir ./runs/baseline_sgd \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.075 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --milestones 100 150 \
    --seed 42
```
This will:
	•	Download CIFAR-10 into ./data (if not present)
	•	Train MobileNetV2 for 200 epochs
	•	Save the best checkpoint (by test accuracy) into ./runs/baseline_sgd/, e.g.
mobilenetv2_cifar10_best.pth

---
## 5. Compression & Evaluation (Questions 2–4)

The compression pipeline:
	•	Quantizes all Conv2d and Linear weights (per-channel for Conv2d, per-tensor for Linear).
	•	Inserts QuantAct before the classifier to quantize the final activation vector.
	•	Supports configurable bitwidths for weights and activations via QuantConfig.

### 5.1. Single compression/evaluation run

Once you have a trained baseline checkpoint (e.g. from Section 4):
```text
python -m src.compress_eval \
    --data-root ./data \
    --checkpoint ./runs/baseline_sgd/mobilenetv2_cifar10_best.pth \
    --project cs6886_mobilenet_quant \
    --seed 42
```
This script will:
	•	Load the baseline model from the given checkpoint.
	•	Build compressed copies of the model using different (w_bits, a_bits, per_channel) settings.
	•	Evaluate each configuration on CIFAR-10 test set.
	•	Compute:
	•	Accuracy after compression
	•	Model size (MB)
	•	Compression ratios for model / weights / activations
	•	Optionally log all runs to Weights & Biases (see below).

---
## 6. Weights & Biases Logging (Question 3)

To reproduce the parallel coordinates plot for Question 3, create (or reuse) a W&B account and login:
```text
wandb login
```
Then run compress_eval.py as in Section 5.1. The script logs:
	•	weight_quant_bits
	•	activation_quant_bits
	•	per_channel flag
	•	compressed_size_mb
	•	compression_ratio
	•	quantized_acc
	•	baseline_size_mb

### 6.1. W&B project link

All my quantization sweep runs are logged under:
```text
	•	Project: cs6886_mobilenet_quant
	•	URL: https://api.wandb.ai/links/ee24d032-iitm-india/vpoiiiqr
```
The Parallel Coordinates chart shown in the report was generated from this project.

---
## 7. Google Colab Notebook

For convenience, the original Colab notebook with all intermediate experiments is included under:
```text
notebooks/CS6886W_Assignment3_Vaishnav.ipynb
```
You can open this directly in Google Colab:
	1.	Upload the .ipynb to Google Drive.
	2.	Right-click → “Open with” → “Colaboratory”.

This notebook mirrors the scripts and was used for exploratory work and plotting.

---
## 8. Full steps
```text
# 1. Clone repository
git clone https://github.com/Vaishnavjay/CS6886w_Assignment3.git
cd CS6886w_Assignment3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train baseline model (200 epochs)
python -m src.train \
    --data-root ./data \
    --save-dir ./runs/baseline_sgd \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.075 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --milestones 100 150 \
    --seed 42

# 4. Run compression sweep + evaluation
python -m src.compress_eval \
    --data-root ./data \
    --checkpoint ./runs/baseline_sgd/mobilenetv2_cifar10_best.pth \
    --project cs6886_mobilenet_quant \
    --seed 42
```






