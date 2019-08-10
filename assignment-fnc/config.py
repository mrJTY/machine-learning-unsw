LABEL_LOOKUP = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

DATASOURCES = {
    "train": {
        "bodies": "data/train_bodies.csv",
        "stances":"data/train_stances.csv"
    },
    "test": {
        "bodies": "data/competition_test_bodies.csv",
        "stances": "data/competition_test_stances.csv"
    }
}

RANDOM_STATE = 123

# These were found by plotting the validation scores from a simple tree
TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 50
