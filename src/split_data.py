
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.model_selection import train_test_split

class SplitStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
        pass

class SimpleSplitStrategy(SplitStrategy):
    def split(self, df: pd.DataFrame, target_col: str, test_size : float, random_state: int):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

class SplitData():
    def __init__(self, strategy: SplitStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SplitStrategy):
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42):
        return self._strategy.split(df, target_col, test_size, random_state)


if __name__ == "__main__":
    df = pd.read_csv("./data/train_chunk_0.csv")
    split_data = SplitData(SimpleSplitStrategy())
    X_train, X_test, y_train, y_test = split_data.execute_strategy(df, "Class")

    print(X_train)