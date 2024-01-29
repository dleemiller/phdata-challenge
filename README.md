# phdata-challenge

## Data Analysis

View the jupyter notebook for the data analysis portion of the project.
The notebook must be used to produce the deduplicated dataset for training.

## Training

Run `train.py` with the search options:
```
$ python train.py search --file artifacts/grid_search.json
```

This will run a grid search over the specified parameters and saves a file `artifacts/best_params.json`.
Because of the class imbalance, grid search is scored according to the f1-score.

Train your model with the train option:
```
$ python train.py train
```


## Code

### Preprocessing

A few custom transformations are defined in `preprocessing/`. The pipeline for preprocessing is define its init file.

### Data

Import the (deduplicated) dataframe directly for training.

