from pathlib import Path
import pandas as pd
from sklearn import tree
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import mlflow
import mlflow.sklearn


def load_train_data():
    return pd.read_csv(Path(__file__).parents[0].resolve() / "data/train.csv")


def preprocess(df, feature_labels, impute_dict={}):
    df = df.fillna(impute_dict) 
    # df = df.dropna()
    encoder = preprocessing.OrdinalEncoder()
    encoder.fit(df[feature_labels])
    df[feature_labels] = encoder.transform(df[feature_labels])
    return df


def train_and_validate():
    # mlflow.set_tracking_uri('http://192.168.1.34:5000')
    # mlflow.set_experiment('titanic-survival')

    feature_labels = ['Age', 'Embarked', 'Pclass', 'Sex']
    target_label = ['Survived']
    test_split_size = 0.2
    random_number = 42
    criterion = "gini"

    data_train = load_train_data()
    impute_strat = {
        "Age": round(data_train["Age"].mean()),
        "Embarked": data_train["Embarked"].mode()[0],
        "Cabin": "Unknown",
    }

    data_train = preprocess(data_train, feature_labels, impute_strat)
    data_train, data_validate = train_test_split(data_train, test_size=test_split_size, random_state=random_number)

    with mlflow.start_run():
        mlflow.log_params(
            {
            'test_split_size': test_split_size,
            'feature_labels': feature_labels,
            'split_criterion': criterion,
            }
        )
        clf = tree.DecisionTreeClassifier(criterion=criterion)
        clf.fit(X=data_train[feature_labels], y=data_train[target_label])
        predictions = clf.predict(data_validate[feature_labels])
        accuracy = metrics.accuracy_score(y_true=data_validate[target_label], y_pred=predictions)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric('validation_accuracy', accuracy)
        mlflow.sklearn.log_model(clf, 'titanic-survival')

if __name__ == '__main__':
    train_and_validate()