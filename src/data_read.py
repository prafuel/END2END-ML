
from abc import ABC, abstractmethod

import pandas as pd
import zipfile

class Read(ABC):
    @abstractmethod
    def read(self, file_path: str) -> pd.DataFrame:
        pass

class ReadCSV(Read):
    def read(self, file_path: str) -> pd.DataFrame:
        ext = file_path.split(".")[-1]

        if ext == "csv":
            return pd.read_csv(file_path)
        
        if ext == "tsv":
            return pd.read_csv(file_path, sep="\t")
        
        raise ValueError(f"Provided file is not supported, '{file_path}', please provide valid csv")

class ReadZip(Read):
    def read(self, file_path: str) -> pd.DataFrame:
        extension = file_path.split(".")[-1]
        if extension == "zip":
            raise ValueError(f"Provided file is not supperted, '{file_path}', please provide valid zip")

        with zipfile.ZipFile(f"{file_path}") as zip_ref:
            zip_ref.extractall("./extracted_data/")

class DataRead():
    def read_data(self, file_path: str) -> pd.DataFrame:
        extension = file_path.split(".")[-1]

        if extension == "csv" or extension == "tsv":
            return ReadCSV().read(file_path)
        
        raise ValueError(f"Provided file is not supported, '{file_path}'")


if __name__ == "__main__":
    read_obj = DataRead()
    read_obj.read_data("./data/train_chunk_0.csv")