from pathlib import Path
import tempfile
import json

import pandas as pd
from sklearn import tree, ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.sklearn

feature_labels = ["Age", "Embarked", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Cabin"]
target_label = ["Survived"]
estimators = {
    "random_forest": ensemble.RandomForestClassifier,
    "decision_tree": tree.DecisionTreeClassifier,
}
parameter_grids = {
    "random_forest": {
        "n_estimators": [10 * i for i in range(1,16)],
        "criterion": ["gini", "entropy"],
        "max_depth": [None] + list(range(1, 11)),
        "min_samples_split": [2 ** i for i in range(1, 11)],
        "max_features": list(range(1, len(feature_labels) + 1)),
    },
    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None] + list(range(1, 11)),
        "min_samples_split": [2 ** i for i in range(1, 11)],
        "max_features": list(range(1, len(feature_labels) + 1)),
    },
}


def load_train_data():
    return pd.read_csv(Path(__file__).parents[0].resolve() / "data/train.csv")


def preprocess(df, feature_labels, impute_dict={}):
    df = df.fillna(impute_dict)
    encoder = preprocessing.OrdinalEncoder()
    encoder.fit(df[feature_labels])
    df[feature_labels] = encoder.transform(df[feature_labels])
    return df


def random_forest(examples):
    pass


def decision_tree(examples, estimator, parameter_grid):
    clf = GridSearchCV(estimator=estimator, param_grid=parameter_grid)
    clf.fit(X=examples[feature_labels], y=examples[target_label])
    mlflow.log_params(clf.best_params_)
    mlflow.log_param("feature_labels", feature_labels)
    mlflow.log_metric("validation_accuracy", clf.best_score_)

    # TODO: Extract to helper function

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        # Note the mode is 'w' so json could be dumped
        # Note the suffix is .txt so the UI will show the file
        json.dump(clf.param_grid, f)
        f.seek(
            0
        )  # You cannot close the file as it will be removed. You have to move back to its head
        mlflow.log_artifact(f.name)

    feature_importances = dict(
        zip(examples[feature_labels].columns, clf.best_estimator_.feature_importances_,)
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        # Note the mode is 'w' so json could be dumped
        # Note the suffix is .txt so the UI will show the file
        json.dump(feature_importances, f)
        f.seek(
            0
        )  # You cannot close the file as it will be removed. You have to move back to its head
        mlflow.log_artifact(f.name)

    mlflow.sklearn.log_model(clf.best_estimator_, "titanic-survival")


def train_and_validate():
    data_train = load_train_data()
    impute_strat = {
        "Age": round(data_train["Age"].mean()),
        "Embarked": data_train["Embarked"].mode()[0],
        "Cabin": "Unknown",
    }
    data_train = preprocess(data_train, feature_labels, impute_strat)
    decision_tree(data_train)


if __name__ == "__main__":
    with mlflow.start_run():
        train_and_validate()
