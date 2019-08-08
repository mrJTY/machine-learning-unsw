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

# These were found by plotting the validation scores from a simple tree
MAX_DEPTH = 5
