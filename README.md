# Scroll-Ink-Detector

## Overview

This project utilizes deep learning techniques for the semantic segmentation of ink in burnt papyrus scrolls excavated from Vesuvius. The aim is to recover and visualize the text or patterns present on these scrolls, even in their damaged state. We employ a combination of UNet models with various backbones and an ensemble approach to achieve accurate and robust results.

This project achieves a score of `0.599965` in the Vesuvius Ink Detection Challenge hosted on Kaggle. This scores corresponds to a Silver Medal finish in the competition.

## Table of Contents

- [Background](#background)
- [Models](#models)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Composite Loss](#composite-loss)
- [Ensemble Strategy](#ensemble-strategy)

## Background

Deep learning models have shown great promise in the field of computer vision, including semantic segmentation tasks. In this project, we leverage state-of-the-art techniques to address the unique challenge of detecting ink in ancient, damaged scrolls. Our approach combines various models and ensembles their predictions to increase the overall accuracy and robustness of the segmentation.

## Models

This project includes UNet with different backbones for semantic segmentation:

- 'mit_b2'
- 'mit_b3'
- 'mit_b4'
- 'regnety_064'
- 'resnest50d_4s2x40d'
- 'resnet50'
- 'resnet34'

These models serve as the foundation for our segmentation task. Depending on your requirements, you can select the appropriate model for your task.

## Data

Please ensure that you have the relevant dataset prepared and organized as per your project requirements. The dataset should include both input images and corresponding ground truth masks for training and evaluation.

## Training

Our training pipeline involves the following steps:

- Modify the `train_config.py` file to set your desired configuration, including the backbone architecture and training hyperparameters, and then run `train.py`.

- We employ a composite loss function comprising Binary Cross-Entropy (BCE), Dice, and Focal loss as the criterion for training our models.

- The learning rate is managed using the GradualWarmupSchedulerV2, and optimization is performed using AdamW.

## Inference

For generating predictions from our trained models:

- Modify the `test_config.py` file to specify the list of backbones and the path to your data and then run `test.py`.

- We apply an ensemble strategy by averaging the confidence scores for each pixel across multiple models. This ensemble approach enhances the accuracy and reliability of our predictions.

## Configuration

You can fine-tune various parameters and configurations in the respective configuration files (`train_config.py` and `test_config.py`) to tailor the project to your needs.

## Composite Loss

Our composite loss function combines Binary Cross-Entropy (BCE), Dice, and Focal loss to effectively train the models. The combination of these loss functions helps capture fine-grained details and improves segmentation accuracy.

## Ensemble Strategy

To make predictions more robust, we utilize an ensemble strategy by running the model on all three consecutive layers for eight layers and then averaging the output. A threshold of 0.5 is applied to convert confidence scores into a binary mask.
