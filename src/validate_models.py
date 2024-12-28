
import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, f1_score

from split_data import SplitData, SimpleSplitStrategy
from data_read import DataRead

models = os.listdir("./models")
# print(models)

df = DataRead().read_data("data/train_chunk_9.csv")
X_train, X_test, y_train, y_test  = SplitData(SimpleSplitStrategy()).execute_strategy(df, target_col="Class", random_state=23)

model_names = []
model_paths = []

training_accuracy = []
testing_accuracy = []

training_f1 = []
testing_f1 = []

for model in models:
    name = model.split("-")[0]
    model_path = f"./models/{model}/model.pkl"

    model = joblib.load(model_path)

    y_pred = model.predict(X_train)
    y_pred_ = model.predict(X_test)

    # names
    model_names.append(name)

    # file names
    model_paths.append(model_path)

    # accuracy
    training_accuracy.append(
        accuracy_score(y_train, y_pred)
    )

    testing_accuracy.append(
        accuracy_score(y_test, y_pred_)
    )

    # f1
    training_f1.append(
        f1_score(y_train, y_pred, average="weighted")
    )

    testing_f1.append(
        f1_score(y_test, y_pred_, average="weighted")
    )

df = (
    pd.DataFrame()
    .assign(
        model_name = model_names,
        training_accuracy = training_accuracy,
        testing_accuracy = testing_accuracy,
        trainig_f1 = training_f1, 
        testing_f1 = testing_f1,
        model_paths = model_paths
    )
)

print("=======" * 3)
print("= Model_performances", end="")
print(" =")
print("=======" * 3)
print(df)