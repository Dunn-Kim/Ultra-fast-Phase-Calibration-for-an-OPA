import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

sys.path.append("C:\\Dev\\IDE\\SimTool\\Lumerical\\api\\python")  # Default windows lumapi path
sys.path.append(os.path.dirname(__file__))  # Current directory
import lumapi as api

class Phase:
    def __init__(self, N, hide=True):
        self.path = f"{N}_Periodic_MODE"
        self.delay_quant = N
        self.session = api.MODE(filename=f"{self.path}\\MODE", hide=hide)
        self.time = time.time()

    def phase_shift(self, phase):
        self.session.switchtolayout()
        for index in range(0, self.delay_quant - 1):
            # self.session.setnamed(f"MODE_{index + 1}", "phase", adjust)
            self.session.setnamed(f"mode{index + 2}", "phase", phase[index])
        self.session.run()

    def set_E2(self):
        angle = np.arange(-90.0, 90.1, 0.1)
        temp = self.session.farfieldangle("monitor", 1, 1801, 1)
        E2 = self.session.farfield2d("monitor", 1, 1801, 1, 1, 1, 1)
        E2 = self.session.interp(E2, temp, angle).T[0]

        plt.plot(angle, E2)
        plt.savefig("MODE_Result.jpg")
        plt.close()

        return np.max(E2), E2 / np.sum(E2)

    def terminate(self, intensity):
        end = time.time()
        hour = str(int((end - self.time) // 3600))
        minute = '{0:02d}'.format(int((end - self.time) % 3600 // 60))
        second = '{0:02d}'.format(int((end - self.time) % 60.0 // 1))
        print(f" || Computation Time : {hour}:{minute}:{second} || Mainlobe : {intensity}")

        self.session.close()
        self.time = time.time()
