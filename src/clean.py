
from abc import ABC, abstractmethod
import pandas as pd

class CleaningStrategy(ABC):
    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class SimpleCleaningStratery(CleaningStrategy):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df
            .dropna()
            .drop_duplicates()
        )

class Cleaner:
    def __init__(self, strategy: CleaningStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: CleaningStrategy):
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame):
        return self._strategy.clean(df)


if __name__ == "__main__":
    df = pd.read_csv("./data/train_chunk_0.csv")
    
    cleaner = Cleaner(SimpleCleaningStratery())
    print(cleaner.execute_strategy(df))