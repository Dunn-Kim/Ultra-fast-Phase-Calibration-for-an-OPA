import torch as th
import pandas as pd

sim = th.nn.CosineSimilarity()

mimic = pd.read_csv("mimic.csv")['0'].tolist()
init = pd.read_csv("init.csv")['0'].tolist()

mimic = th.tensor(mimic, device='cuda', dtype=th.float64).reshape(1, 31) % 360
init = th.tensor(init, device='cuda', dtype=th.float64).reshape(1, 31) % 360

print(mimic)
print(init)


