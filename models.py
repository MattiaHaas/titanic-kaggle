from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def random_forest(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(x_train, y_train)
    y_test = model.predict(x_test)
    return y_test
