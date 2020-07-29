from pathlib import Path

import pandas as pd
from sklearn import tree, preprocessing, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.sklearn

feature_labels = ["Age", "Embarked", "Pclass", "Sex", "SibSp", "Parch"]
target_label = ["Survived"]


def load_train_data():
    return pd.read_csv(Path(__file__).parents[0].resolve() / "data/train.csv")


def preprocess(df, feature_labels, impute_dict={}):
    df = df.fillna(impute_dict)
    encoder = preprocessing.OrdinalEncoder()
    encoder.fit(df[feature_labels])
    df[feature_labels] = encoder.transform(df[feature_labels])
    return df


def decision_tree(examples):
    parameter_grid = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 1, 2, 3, 4, 5, 6, 7],
    }
    estimator = tree.DecisionTreeClassifier()
    clf = GridSearchCV(estimator=estimator, param_grid=parameter_grid)
    clf.fit(X=examples[feature_labels], y=examples[target_label])
    mlflow.log_params(clf.best_params_)
    mlflow.log_metric("validation_accuracy", clf.best_score_)
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
