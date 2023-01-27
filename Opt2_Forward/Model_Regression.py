import Optimization_func as optf
import Lumerical_Func as lumf
import pandas as pd

"""
!!!
Manually Set directory of lumerical API location in [Lumerical func]
interopapi.dll location in [lumapi.py]
line_quant = Number of SiN element to be built on MODE ( or FDTD )
Target_Angle = Desired steering Angle 
Learning_rate = Rate of applied learning parameter for Optimizer weight. 
32 (0) 8.5e+1 || (+)  
64 8.5e+1
128 8e+1
!!!
"""

line_quant = 4
for_lr = 1e+1
iteration_std = 100
path = f"{line_quant}_Periodic_MODE"

phase_log = []
beam_intensity_log = []
pattern_log = []

Adam = optf.Optimizer(line_N=line_quant - 1, learning_rate=for_lr)
phase_log.append(Adam.get_init_noise())

for Epoch in range(1, iteration_std + 1):
    phase = Adam.get_formula_grad()
    phase_log.append(phase)

    session = lumf.Phase(hide=True, N=line_quant - 1, path=path)
    session.phase_shift(phase=phase)

    E2, intensity = session.set_E2(Epoch=iteration_std + Epoch)
    beam_intensity_log.append(intensity)
    pattern_log.append(E2)

    session.terminate(Epoch=iteration_std + Epoch, loss=Adam.set_forward_loss(E2=E2))
    del session

    Adam.update_weight(Epoch=Epoch)

    if Epoch < 30:
        continue
    elif not ((Epoch - 30) % 15):
        Adam.decay(Epoch=Epoch)

pd.DataFrame(phase_log).to_csv(f"{path}\\phase_tracked.csv")
pd.DataFrame(beam_intensity_log).to_csv(f"{path}\\intensity_tracked.csv")
pd.DataFrame(pattern_log).to_csv(f"{path}\\pattern_tracked.csv")

"""
Copyright @ 2022, KIM DONG HWAN - Korea University AI Dep.
Project Manager - KIM DO HYUNG - Gwangun University Photonics Lab.
"""
