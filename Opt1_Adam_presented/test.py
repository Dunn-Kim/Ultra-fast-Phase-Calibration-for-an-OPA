import pandas as pd

phase = pd.read_csv("Phase.csv")
print(phase.iloc[4].tolist())