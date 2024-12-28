
from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

class ModelSelectionStrategy(ABC):
    @abstractmethod
    def select(self, preprocessing_pipe: Pipeline, parameters: dict) -> Pipeline:
        """
        Parameters: Preprocessing Pipe, model, parameters dict
        Returns: Model Pipeline
        """
        pass

class GetModel(ModelSelectionStrategy):
    def select(self, preprocessing_pipe: Pipeline, model, parameters: dict) -> Pipeline:
        if parameters == None: parameters = {}
        
        complete_pipe = Pipeline([
            ("pre", preprocessing_pipe),
            ("model", model(**parameters))
        ])

        return complete_pipe


if __name__ == "__main__":
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    model_selector = GetModel()
    print(model_selector.select(
        Pipeline([("imputer", SimpleImputer())]), LogisticRegression, {"max_iter" : 100}
    ))
