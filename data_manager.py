import pandas as pd
from typing import List


def read_data_from_csv(train_data_path: str, test_data_path: str):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def get_input_output_given_columns(
    train_data: pd.DataFrame, test_data: pd.DataFrame, features: List[str], output: str
):
    y_train = train_data[output]
    x_train = pd.get_dummies(train_data[features])
    x_test = pd.get_dummies(test_data[features])
    return x_train, y_train, x_test


def create_submission_file(test_data: pd.DataFrame, y_test: pd.DataFrame):
    output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": y_test})
    output.to_csv("submission.csv", index=False)
    print("Your submission was successfully saved!")
