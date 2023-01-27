import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

sys.path.append("C:\\Dev\\IDE\\SimTool\\Lumerical\\api\\python")  # Default windows lumapi path
sys.path.append(os.path.dirname(__file__))  # Current directory
import lumapi as api


class Phase:
    def __init__(self, hide, N, path):
        self.delay_quant = N
        self.angle = np.arange(-90.0, 90.1, 0.1)
        self.session = api.MODE(filename=f"{path}\\MODE", hide=hide)
        self.time = time.time()
        self.path = path

    def phase_shift(self, phase):
        self.session.switchtolayout()
        for index, adjust in enumerate(phase):
            self.session.setnamed(f"mode{index + 1}", "phase", adjust)
        self.session.run()

    def set_E2(self, Epoch):
        temp = self.session.farfieldangle("monitor", 1, 1801, 1)
        E2 = self.session.farfield2d("monitor", 1, 1801, 1, 1, 1, 1)
        E2 = self.session.interp(E2, temp, self.angle).T[0]

        '''
        plt.plot(self.angle, E2)
        plt.savefig(f"{self.path}\\{Epoch}_Result.jpg")
        plt.close()
        '''

        return E2.tolist(), E2[900]

    def terminate(self, Epoch, loss):
        end = time.time()
        minute = '{0:02d}'.format(int((end - self.time) % 3600 // 60))
        second = '{0:02.2f}'.format(round((end - self.time) % 60, 2))

        print("@ Epoch : " + '{0:02d}'.format(Epoch) +
              f" || Computation Time : {minute}:{second} || Loss : {loss}")

        self.session.close()
        self.time = time.time()
