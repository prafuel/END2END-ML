
from abc import ABC, abstractmethod
import pandas as pd

class Validate(ABC):
    @abstractmethod
    def check(self, df: pd.DataFrame):
        pass

"""
- Check for Null Values
- Check for Duplicate rows
"""

class CheckMissingValuesStrategy(Validate):
    def check(self, df: pd.DataFrame):
        return pd.DataFrame({
            "columns" : df.columns,
            "null_count" : df.isnull().sum().values
        })

class CheckDuplicateValuesStrategy(Validate):
    def check(self, df: pd.DataFrame):
        return f"Number of Duplicate Rows : {df[df.duplicated()].shape[0]}"

class Analyzer():
    def analyze(self, df: pd.DataFrame):
        print("Duplicate Values")
        print("=="*20)
        print(CheckDuplicateValuesStrategy().check(df))
        print("=="*20)


        print("Null Values")
        print(CheckMissingValuesStrategy().check(df))


if __name__ == "__main__":
    df = pd.read_csv("./data/train_chunk_0.csv")

    print(CheckMissingValuesStrategy().check(df))

    print(CheckDuplicateValuesStrategy().check(df))