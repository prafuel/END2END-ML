
from run import run
import os

csv_files = sorted(os.listdir("./data"))

print(csv_files)

for i in range(0, 3):
    run(f"./data/{csv_files[i]}")