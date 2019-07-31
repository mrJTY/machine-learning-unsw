from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


LABELS = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

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

MODELS = {
    'tree': DecisionTreeClassifier(),
    'gradient_boost': GradientBoostingClassifier()
}
