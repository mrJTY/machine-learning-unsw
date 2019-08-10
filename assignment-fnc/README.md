# Fake New Challenge

Author: z5232245 Justin Ty

My attempt at the Fake News Challenge(FNC) as a project for COMP9417.

Original utility functions and data were sourced from:
https://github.com/FakeNewsChallenge/fnc-1-baseline

# Installation

Use `make install` or `pip install -r requirements` to install the required packages.

Use Python3, recommended to run in a conda environment.

# Help

```
$ python main.py --help

usage: main.py [-h] [--skip_preprocess] [--model MODEL]
               [--train_prop TRAIN_PROP]

optional arguments:
  -h, --help            show this help message and exit
  --skip_preprocess     If flag is true, it will skip preprocessing the data
                        and load .pickle files in the data/ folder
  --model MODEL         Model name
  --train_prop TRAIN_PROP
                        The proportion of training dataset to read from.
                        Between 0 and 1
```

For example to train a the baseline model using only 25% of training set:

```
$ python main.py --train_prop 0.25 --model gbm > logs/gbm_0.25.log
```


# Wrapper scripts

There are wrapper scripts in `sbin/` that help with setting things up:

1) `sbin/1-download-datasets.sh` - Downloads the article csv's from the FNC challenge
2) `sbin/2-preprocess.sh` - Generate train / test matrices and save then as pickle files in `data/` folder
3) `sbin/3-train-and-test-models.sh` - Train models and get test results

# Preprocessing

See `preprocessing.py` to generate train / test matrices and save then as pickle files in the `data/` folder.

Some feature engineering utility functions were borrowed from:
https://github.com/FakeNewsChallenge/fnc-1-baseline

# Models

Choose and tune the models in `modelling.py`

# Logs

The logs are located in the logs/ folder. The scores were reported here.

# Datasets
The datasets can be originally be found in https://github.com/FakeNewsChallenge/fnc-1/tree/29d473af2d15278f0464d5e41e4cbe7eb58231f2

It is assumed that the `data/` folder will contain these files:
* competition_test_bodies.csv
* competition_test_stances.csv
* train_bodies.csv
* test_bodies.csv

