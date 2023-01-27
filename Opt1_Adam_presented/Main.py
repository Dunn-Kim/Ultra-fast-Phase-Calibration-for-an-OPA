import Optimizer as Opt
import pandas as pd

line_quant = 32
iteration_std = 100

phase_log = []
intensity_log = []

# Default Learning Rate -> 5e+1 || Diversity 3.5e-3
Adam = Opt.Optimizer(line_N=line_quant - 1, angle=0)
phase_log.append(Adam.get_init())

for Epoch in range(1, iteration_std + 1):

    phase, intensity, loss = Adam.OPA_formula()
    phase_log.append(phase)
    intensity_log.append(intensity)

    print(f"@ Epoch : {Epoch:4d} || Loss : {loss:.8f} || Main Lobe : {intensity:.8f}")
    Adam.update_weight(Epoch=Epoch)

    if not (Epoch % 10):
        Adam.decay()

print(Adam.get_result())
del Adam

pd.Series(phase_log).to_csv("phase_log.csv")
pd.Series(intensity_log).to_csv("intensity_log.csv")
