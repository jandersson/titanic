"""
Get predictions for test set and submit to kaggle. After submitting, update the model run_id with the score returned by kaggle.
"""


import mlflow
import pandas as pd
from pathlib import Path
from train import preprocess, feature_labels
import kaggle
import time
import os


if __name__ == "__main__":
    # Load Model and fetch run_id so we can record the kaggle accuracy score after submitting
    model_name = "titanic-survival"
    model_stage = "Production"
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_client = mlflow.tracking.MlflowClient(
        tracking_uri=mlflow_uri, registry_uri=mlflow_uri
    )
    for model in mlflow_client.search_registered_models(f"name='{model_name}'"):
        for version in dict(model)["latest_versions"]:
            if version.current_stage == model_stage:
                run_id = version.run_id
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")

    # Load data
    data_path = Path(__file__).parents[0].resolve() / "data"
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    # preprocessing
    impute_strats = {
        "Age": round(train_df["Age"].mean()),
        "Embarked": train_df["Embarked"].mode()[0],
        "Cabin": "Unknown",
    }
    test_df = preprocess(test_df, feature_labels, impute_strats)
    test_df["Survived"] = model.predict(test_df[feature_labels])
    test_df[["PassengerId", "Survived"]].to_csv(
        data_path / "submission.csv", index=False
    )

    # submit to kaggle
    kaggle.api.competition_submit(
        file_name=str(data_path / "submission.csv"),
        message="Testing submission api",
        competition="titanic",
    )
    time.sleep(30)
    test_accuracy = kaggle.api.process_response(
        kaggle.api.competitions_submissions_list_with_http_info("titanic")
    )[0]["publicScore"]
    mlflow_client.log_metric(run_id, "test_accuracy", float(test_accuracy))
