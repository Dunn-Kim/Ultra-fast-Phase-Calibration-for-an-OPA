from math import pi, radians, ceil, sqrt, sin
import matplotlib.pyplot as plt
import numpy as np
import torch as th


class Optimizer:
    def __init__(self, line_N, randomness=3.5e-3, angle=0, learning_rate=5e+1):
        self.delay_quant = line_N
        self.target = angle

        self.lr = learning_rate
        self.div = randomness
        self.loss = None
        self.OPA_E2 = None

        theta = th.sin(th.deg2rad(th.arange(-90.0, 90.1, 0.1, dtype=th.float64, device='cuda'))).reshape((1, 1801))
        theta[0][900] = 0

        k_lit = 2 * pi / 1.55e-6
        beta = k_lit * 1e-6 * theta / 2
        sind = sin(radians(angle))

        self.tensor_quant = th.arange(0, line_N, 1, dtype=th.float64, device='cuda').reshape((1, line_N))
        delph = k_lit * 3e-6 * sind - 2 * pi * ceil(3e-6 * sind / 1.55e-6) * self.tensor_quant

        self.EF = th.square(th.sin(beta) / beta)
        self.EF[0][900] = 1
        self.alpha = k_lit * 3e-6 * theta / 2
        self.alpha[0][900] = 0

        self.init_noise = th.randint(low=-180, high=180, size=(1, self.delay_quant), dtype=th.float32, device='cuda')
        self.phase = (delph + th.zeros(size=(1, line_N), dtype=th.float64, device='cuda')).clone().requires_grad_()
        self.m_1 = th.zeros_like(self.phase, dtype=th.float64, device='cuda')
        self.v_1 = th.zeros_like(self.phase, dtype=th.float64, device='cuda')

    def OPA_formula(self):
        noise = th.normal(mean=1, std=self.div, size=self.phase.shape, dtype=th.float32, device='cuda')
        OPA_phase = self.init_noise - noise * self.phase

        AF = th.sum(th.exp(1j * (2 * self.alpha * self.tensor_quant.t() - OPA_phase.t())), dim=0, keepdim=True)
        self.OPA_E2 = self.EF * th.abs(AF)

        if self.target:
            loss = 1 / (self.OPA_E2[0][900] +
                        0.5 * self.OPA_E2[0][1236] + 0.5 * self.OPA_E2[0][599] + 0.5 * self.OPA_E2[0][207])
        else:
            loss = 1 / (self.OPA_E2[0][900] + 0.5 * self.OPA_E2[0][589] + 0.5 * self.OPA_E2[0][1211])

        loss.backward()
        return OPA_phase.tolist()[0], self.OPA_E2[0][900].item(), round(loss.item(), 8)

    def update_weight(self, Epoch, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_1 = beta1 * self.m_1 + (1 - beta1) * self.phase.grad
        self.v_1 = beta2 * self.v_1 + (1 - beta2) * th.square(self.phase.grad)

        m_n = self.m_1 / (1 - (beta1 ** Epoch))
        v_n = self.v_1 / (1 - (beta2 ** Epoch))

        step_lr = self.lr * sqrt(1 - (beta2 ** Epoch)) / (1 - (beta1 ** Epoch))
        self.phase.data -= step_lr * m_n / (th.sqrt(v_n) + epsilon)
        self.phase.grad.data.zero_()

    def decay(self):
        print("\n!!! Decay Proceed !!!")
        self.lr *= 0.9
        self.div *= 0.7

    def get_result(self):
        angle = np.arange(-90, 90.1, 0.1)
        E2 = np.array(self.OPA_E2.tolist()[0])

        plt.plot(angle, E2)
        plt.savefig("result.jpg")
        plt.close()

        return self.OPA_E2.tolist()[0]

    def get_init(self):
        return self.init_noise.tolist()[0]
