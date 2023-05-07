from data_manager import read_data_from_csv, get_input_output_given_columns, create_submission_file
from models import random_forest

TEST_DATA_PATH = "dataset/test.csv"
TRAIN_DATA_PATH = "dataset/train.csv"

features = ["Pclass", "Sex", "SibSp", "Parch"]
output = "Survived"

train_data, test_data = read_data_from_csv(TRAIN_DATA_PATH, TEST_DATA_PATH)
x_train, y_train, x_test = get_input_output_given_columns(train_data, test_data, features, output)
y_test = random_forest(x_train, y_train, x_test)

create_submission_file(test_data, y_test)
