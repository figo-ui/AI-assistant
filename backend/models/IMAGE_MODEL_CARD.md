# Image Model Card

## Overview

The image branch is currently dermatology-focused and supports:

- `efficientnet_b3`
- `resnet50`

Artifacts:

- `skin_cnn_torch.pt`
- `image_labels.json`
- `image_model_metadata.json`
- `image_training_metrics.json`

## Current Status

The checked-in metrics show the image branch is still prototype-grade:

- dataset: `dermamnist`
- test macro-F1: `0.307`
- test accuracy: `0.5022`
- classes: `7`

This should be treated as a supplementary dermatology signal, not a production-trustworthy medical image model.

## Intended Use

- skin-image ranking in a multimodal prototype
- fusion input when the image quality score is acceptable

## Not Intended Use

- radiology, pathology, ophthalmology, or general medical imaging
- sole basis for diagnosis or treatment

## Safety Notes

- runtime downweights poor-quality images
- runtime exposes the branch as `dermatology_skin_only` for legacy 7-class checkpoints
- text-only behavior must remain available when the image branch is missing or weak

## Recommended Path

Use Kaggle to retrain on a broader manifest built from:

- HAM10000
- PAD-UFES-20
- Fitzpatrick17k
- optional ISIC subset

Entry points:

- manifest build: `backend/scripts/download_medical_image_datasets.py`
- training: `backend/scripts/train_image_model.py`
- Kaggle workflow: `backend/notebooks/kaggle_image_training_workflow.md`
