
from src.data_read import DataRead
from src.validate import Analyzer
from src.split_data import SplitData, SimpleSplitStrategy
from src.train_model import TrainingAndBestFitSelector

import os
import time
import sys
from random import randint
from datetime import date

import joblib

# file_path = "./data/train_chunk_1.csv"

def run(file_path: str, output_path: str="./models"):
    # 1. reading data
    df = DataRead().read_data(file_path=file_path)

    # 2. checking for missing and null values
    Analyzer().analyze(df)

    # 3. splitting data
    data_split_params = {
        "target_col" : "Class",
        "test_size" : 0.3,
        "random_state" : 42
    }

    X_train, X_test, y_train, y_test = SplitData(SimpleSplitStrategy()).execute_strategy(df=df,**data_split_params)
    train = (X_train, y_train)
    test = (X_test, y_test)

    # 4. setting training model params,
    trainer = TrainingAndBestFitSelector()

    model_parameters = {
        "lor" : {"max_iter" : 1000},
        "dt" : {"max_depth" : 4},
        "rf" : {"n_estimators" : 100, "max_depth" : 5}
    }

    # 5. loading model
    trainer.loading_model(
        model_parameters,
        train,
        test
    )

    # 6. best fit model selection
    model_name, model, evaluation_mat = trainer.model_selection()

    print(f"""
    {"===" * 20}
    Best Fit Model Name : {model_name}
    {"===" * 20}
    Evaluation_mat :
    {"===" * 20}
    {evaluation_mat}
    {"===" * 20}
    """)

    # 7. saving best model with versioning on /models
    unique_id = int(time.time() * randint(0, 99) % 9999)
    today = date.today()

    dir_name = f"{output_path}/{model_name}-{today}-u{unique_id}"

    os.mkdir(f"{dir_name}")
    with open(f"{dir_name}/parameters_change.txt", "w") as f:
        f.write(f"""
        {"===" * 20}
        Model_name: {model_name}
        {"===" * 20}
        Training_Data_File_Paht: {file_path}
        {"===" * 20}
        {evaluation_mat}
        {"===" * 20}
        data_split_params: {data_split_params}
        {"===" * 20}
        model_params: {model_parameters}
        {"===" * 20}
        """)

    # 8. saving model
    joblib.dump(model, f"{dir_name}/model.pkl", compress=9)

    return f"All Info Saved!!!, here {output_path}"


if __name__ == "__main__":
    file_path = "./data/train_chunk_1.csv"

    args: list = sys.argv
    n: int = len(args)

    if n > 1:
        file_path = args[1]

    print(f"Running Model on this data '{file_path}'")
    run(file_path=file_path)
