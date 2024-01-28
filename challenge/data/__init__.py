import importlib.resources
import pandas as pd
import sklearn


with importlib.resources.path("challenge.data", "project_data.csv") as file_path:
    df = pd.read_csv(file_path)
