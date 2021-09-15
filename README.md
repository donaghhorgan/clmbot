# clmbot

## Training a model

### Training with Gradient

Training with [Gradient](https://gradient.run/) is a simple and free way to get started. Just follow these steps:

1. Set up an account with Gradient.
2. Create a project on Gradient to manage your work. You can name the project anything you like.
3. Create a new workflow under your Gradient project. You can name the workflow anything you like.
4. Create the following datasets on Gradient:
   - `clmbot-data`
   - `clmbot-models`
5. Upload one or more `.txt` files to the `clmbot-data` dataset.
6. In a terminal, set the environment variable `GRADIENT_TRAIN_WORKFLOW_ID` to the ID of your Gradient workflow. Then, run `make train` to start training.
7. Monitor the progress of your training on Gradient.