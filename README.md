# CoSTaR
### Consensus Self-Taught Aggregation with Reliability for Weakly Supervised Medical Image Segmentation from Loose Bounding Boxes

Official example implementation of **CoSTaR**, a framework for learning medical image segmentation from loose bounding box annotations through multi-view consensus and self-taught pseudo-label refinement.

---

## Overview

Medical image segmentation typically requires dense pixel-level annotations, which are expensive and time-consuming to obtain. CoSTaR addresses this challenge by learning directly from **loose bounding boxes** without requiring precise segmentation masks during training.

The framework generates multiple augmented views of each training image, aggregates predictions into a common reference frame, estimates prediction reliability, and constructs high-quality pseudo labels for self-supervised optimization.

<div align="center">

<img src="Figures/Figure 1.png" width="100%">

**Figure 1.** Overview of the proposed CoSTaR framework.

</div>

---

## Features

- Weakly supervised medical image segmentation
- Training from loose bounding box annotations
- Multi-view prediction aggregation
- Soft Spatial Prior (SSP)
- Reliability-aware pseudo-label generation
- Self-taught iterative refinement
- Architecture independent
- Compatible with CNN and Transformer segmentation models

---

## Framework

The proposed pipeline consists of four major components:

1. **Random geometric augmentation**
2. **Multi-view prediction aggregation**
3. **Consensus and Soft Spatial Prior**
4. **Reliability-aware self-taught pseudo-label generation**

The final pseudo labels are used to supervise the segmentation network during training.

---

## Experimental Results

### Quantitative Comparison

CoSTaR is evaluated on three public medical image segmentation datasets:

- BRISC
- Kvasir-SEG
- HAM10000

using five representative segmentation backbones:

- U-Net
- TransUNet
- UNetT
- MedFormer
- DeepLabv3+

<div align="center">

<img src="Figures/Table 1.png" width="100%">

**Table 1.** Quantitative comparison across datasets and backbone architectures.

</div>

---

### Statistical Analysis

Paired statistical tests demonstrate that CoSTaR consistently improves over competing box-supervised methods across datasets and architectures.

<div align="center">

<img src="Figures/Table 2.png" width="70%">

**Table 2.** Statistical comparison between CoSTaR and competing methods.

</div>

---

### Performance Distribution

The figure below illustrates the distribution of segmentation performance and backbone rankings.

<div align="center">

<img src="Figures/Figure 2.png" width="100%">

**Figure 2.** Distribution of segmentation performance and corresponding backbone rankings.

</div>

---

### Qualitative Results

Representative segmentation examples together with the subjective evaluation.

<div align="center">

<img src="Figures/Figure 3.png" width="100%">

**Figure 3.** Subjective comparison and representative qualitative examples.

</div>

---

## Ablation Study

### Loss Function Ablation

<div align="center">

<img src="Figures/Table 3.png" width="95%">

**Table 3.** Ablation study of different CoSTaR components.

</div>

---

### Hyperparameter Sensitivity

<div align="center">

<img src="Figures/Table 4.png" width="95%">

**Table 4.** Hyperparameter sensitivity analysis.

</div>

---

## Supplementary Results

### Dataset-wise Subjective Evaluation

<div align="center">

<img src="Figures/Figure S1.png" width="70%">

**Figure S1.** Subjective evaluation for each dataset.

</div>

---

### Prediction Visualization

<div align="center">

<img src="Figures/Figure S2.png" width="95%">

**Figure S2.** Ground truth and prediction overlays.

</div>

---

### Training Hyperparameters

<div align="center">

<img src="Figures/Table S1.png" width="70%">

**Table S1.** Key hyperparameters.

</div>

---

## Repository Structure

```text
CoSTaR/
│
├── Model/                  # Segmentation network architectures
├── LossFunction/           # Loss functions used by CoSTaR
├── Evaluation/             # Evaluation metrics and scripts
│
├── train.py                # Main training script
├── Solver.py               # Training pipeline
├── model_loader.py         # Model construction
├── loss.py                 # Loss function wrapper
│
├── data_loader.py 
│
├── Evaluation_Matix.py     # Evaluation entry point
├── getmatrix.py            # Metric computation
├── get_overleaf_table.py   # Generate publication tables
├── printable.py            # Result formatting
│
├── utils.py                # Utility functions
│
├── Figures/                # Figures used in README and paper
│
└── README.md
```
---

## Training

```bash
python3 train.py --train True --resume True --network Deeplabv3 --batch_size 15
python3 train.py --train True --resume True --network Medformer --batch_size 15
python3 train.py --train True --resume True --network Transunet --batch_size 15
python3 train.py --train True --resume True --network unet  --batch_size 15
python3 train.py --train True --resume True --network UnetT --batch_size 15 
```
Evaluation is automatic after training/resuming training.

---

## Citation

If you find this repository useful, please consider citing our paper.

```bibtex
@inproceedings{costar,
}
```

---

## Acknowledgements

This repository builds upon publicly available medical image segmentation datasets and open-source segmentation frameworks. We thank the authors of these resources for making their work publicly available.
