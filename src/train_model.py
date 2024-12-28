
import pandas as pd
from abc import ABC, abstractmethod

from src.load_model import GetModel
from src.split_data import SplitData, SimpleSplitStrategy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression as lor
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class ModelTrainingTemplate(ABC):
    @abstractmethod
    def loading_model(self, df: pd.DataFrame, model_desc: dict) -> Pipeline:
        pass

    @abstractmethod
    def model_selection(self) -> tuple:
        pass

    @abstractmethod
    def get_scores(self, model) -> list:
        pass

class TrainingAndBestFitSelector(ModelTrainingTemplate):
    def loading_model(self, model_parameters: dict, train: tuple, test: tuple) -> Pipeline:
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test

        pipe = Pipeline([
            ("impute", SimpleImputer()),
            ("scaling", StandardScaler()),
        ])

        self.lor_pipe = GetModel().select(pipe, lor, model_parameters.get("lor"))
        self.dt_pipe = GetModel().select(pipe, dt, model_parameters.get("dt"))
        self.rf_pipe = GetModel().select(pipe, rf, model_parameters.get("rf"))

        self.models = {
            "lor" : self.lor_pipe,
            "dt" : self.dt_pipe,
            "rf" : self.rf_pipe
        }

    def model_selection(self) -> tuple:
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

            accuracy_score, precision_score, recall_score, f1_score = self.get_scores(model=model)
            accuracy_scores.append(accuracy_score)
            precision_scores.append(precision_score)
            recall_scores.append(recall_score)
            f1_scores.append(f1_score)

        evaluation = (
            pd.DataFrame()
            .assign(
                models=self.models.keys(),
                accuracy=accuracy_scores,
                precision=precision_scores,
                recall=recall_scores,
                f1_score=f1_scores
            )
            .sort_values(by=['f1_score', 'accuracy', 'recall', 'precision'], ascending=False)
        )

        best_fit_model_name = evaluation.models.values[0]
        return best_fit_model_name, self.models[best_fit_model_name], evaluation

    def get_scores(self, model) -> list:
        y_pred = model.predict(self.X_test)
        y_test = self.y_test
        return (
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        )

if __name__ == "__main__":
    df = pd.read_csv("./data/train_chunk_0.csv")
    trainer = TrainingAndBestFitSelector()
    trainer.loading_model(df, {"lor" : {"max_iter" : 1000}})

    name, model, evaluation = trainer.model_selection()
    print(name)

    y_pred_ = model.predict(trainer.X_test)
    print(accuracy_score(trainer.y_test, y_pred_))

    print("==="*10)
    print(evaluation)