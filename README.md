# clmbot

![Linting](https://github.com/donaghhorgan/clmbot/workflows/Lint%20code%20base/badge.svg)
![Regression tests](https://github.com/donaghhorgan/clmbot/workflows/Regression%20tests/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A framework for training causal language models for bots.

## Table of contents

- [Train a model](#train-a-model)
  - [Train with Gradient](#train-with-gradient)
  - [Train with Python](#train-with-python)
- [Deploy a model](#deploy-a-model)

## Train a model

### Train with Gradient

Training with [Gradient](https://gradient.run/) is a simple and free way to get started. Just follow these steps:

1. Set up an account with Gradient.
2. Create a project on Gradient to manage your work. You can name the project anything you like.
3. Create a new workflow under your Gradient project. You can name the workflow anything you like.
4. Create the following datasets on Gradient:
   - `clmbot-config`
   - `clmbot-data`
   - `clmbot-models`
5. Upload a copy of `train.yml` to the `clmbot-config` dataset. Edit the file before uploading if you'd like to change the default training parameters.
6. Upload one or more `.txt` files to the `clmbot-data` dataset.
7. In a terminal, set the environment variable `GRADIENT_TRAIN_WORKFLOW_ID` to the ID of your Gradient workflow. Then, run `make train` to start training.
8. Wait until training has completed.
9. In a terminal, set the environment variable `GRADIENT_MODEL_DATASET_ID` to the ID of the `clmbot-models` dataset (run `gradient datasets list` to see the IDs of your datasets). Then, run `make fetch` to download the trained model to your local machine. 

### Train with Python

Training with Python is not complicated, but you will probably need a GPU to do it in a reasonable amount of time. Just follow these steps:

1. Edit `train.yml` if you'd like to change the default training parameters.
2. Copy one or more `.txt` files to the dataset path specified in `train.yml`.
3. Run `python -m clmbot train` to start training.

## Deploy a model

Deploying a model is straightforward:

1. Edit `deploy.yml` if you'd like to change the default deployment parameters.
2. Copy a trained model to the path specified in `deploy.yml`.
3. Run `python -m clmbot deploy` to deploy the model.
