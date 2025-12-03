# Session 2

Train your own model with a custom dataset ([Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist), [direct download](https://www.kaggle.com/api/v1/datasets/download/gpreda/chinese-mnist)), using hyperparameter tuning.



## Solution notes

For `main.py`, these are the results with this solution (depends on machine/seed, so not perfectly reproduceable):

```
Train Epoch 4 loss=0.14 acc=0.96
Eval Epoch 4 loss=0.15 acc=0.95
Test loss=0.14 acc=0.96
```

And with `main_hyperparam_optimize.py`, these are the set of hyperparameters that gave best results:

```python
{
    'lr': 0.0006913958136676372,
    'batch_size': 64,
    'epochs': 5,
    'h1': 36,
    'h2': 54,
    'h3': 137,
    'h4': 119
}
```


## Installation
### With Conda
Create a conda environment by running

```bash
conda create --name aidl-session2 python=3.8
```
Then, activate the environment
```bash
conda activate aidl-session2
```
and install the dependencies
```bash
pip install -r requirements.txt
```

**Note:** it is important to create a new conda environment, and not re-use previous ones from other sessions.

## Running the project

To run the project, run
```bash
python session-2/main.py
```
To run the project with hyperparameter tuning, run
```bash
python session-2/main_hyperparam_optimize.py
```
