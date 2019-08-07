# Fake New Challenge

Author: z5232245 Justin Ty

Original utility functions and data were sourced from:
https://github.com/FakeNewsChallenge/fnc-1-baseline

# Installation

Use `make install` or `pip install -r requirements` to install the required packages.

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

# Models

Choose the models from `modelling.py`

# Example

Train a tree model using only 25% of training set

```
$ python main.py --train_prop 0.25 --model tree > logs/tree_0.25.log
```

# Logs

The logs are located in the logs/ folder. The scores were reported here.

# Datasets
The datasets can be originally be found in https://github.com/FakeNewsChallenge/fnc-1/tree/29d473af2d15278f0464d5e41e4cbe7eb58231f2

It is assumed that the `data/` folder will contain these files:
* competition_test_bodies.csv
* competition_test_stances.csv
* train_bodies.csv
* test_bodies.csv

