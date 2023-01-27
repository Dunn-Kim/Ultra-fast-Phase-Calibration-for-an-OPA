from math import pi, ceil, radians, sin, sqrt
import torch as th
# import matplotlib.pyplot as plt
# import numpy as np


class Optimizer:
    def __init__(self, line_N: int, learning_rate: float, clone=False, angle=0):
        print("!!! Optimization Procedure Initiated !!!")
        self.cost_function = th.nn.CosineSimilarity()
        self.lr = learning_rate
        self.div = 3.5e-3
        self.obj = {'max': -1, 'min': 1}

        # self.angle = np.arange(-90.0, 90.1, 0.1)
        self.delay_quant = line_N
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

        if clone:
            self.init_noise = th.randint(low=-180, high=180, size=(1, line_N), dtype=th.float64, device='cuda')
            self.init_pattern = None
            self.decay_table = {35: 1, 45: 1, 55: 0.9, 65: 0.9, 75: 0.9, 85: 0.9, 95: 1}
        else:
            self.init_noise = None
            self.replica = None
            self.decay_table = {30: 0.98, 45: 0.98, 60: 0.98, 75: 0.98, 90: 0.98}

        self.var_x = (th.zeros(size=(1, line_N), dtype=th.float64, device='cuda') + delph).clone().requires_grad_()
        self.m_w = th.zeros_like(self.var_x, dtype=th.float64, device='cuda')
        self.v_w = th.zeros_like(self.var_x, dtype=th.float64, device='cuda')

    def get_init_noise(self):
        return self.init_noise.tolist()[0]

    def get_replica(self):
        return self.var_x.tolist()[0]

    def get_formula_grad(self, clone=False):
        noise = th.normal(mean=1, std=self.div, size=self.var_x.shape, dtype=th.float64, device='cuda')
        Adj_phase = self.var_x * noise

        if clone:
            OPA_phase = Adj_phase % 360 / 360 * (2 * pi)
        else:
            OPA_phase = (self.init_noise - Adj_phase) % 360 / 360 * (2 * pi)

        y_1 = th.sum(th.exp(1j * (2 * self.alpha * self.tensor_quant.t() - OPA_phase.t())), dim=0, keepdim=True)
        self.OPA_E2 = self.EF * th.abs(y_1)

        """
        plt.plot(self.angle, np.array(self.OPA_E2.tolist()[0]))
        plt.savefig("Result.jpg")
        plt.close()
        """

        if clone:
            return Adj_phase.tolist()[0]
        else:
            return (self.init_noise - Adj_phase).tolist()[0]

    def set_init_pattern(self, origin_norm: list[float]):
        self.init_pattern = th.tensor(origin_norm, dtype=th.float64, device='cuda').reshape(1, 1801)

    def set_init(self, init_phase):
        self.init_noise = th.tensor(init_phase, dtype=th.float64, device='cuda').reshape(1, self.delay_quant)

    def set_backward_loss(self, E2: list[float]):
        self.OPA_E2.data = th.tensor(E2, dtype=th.float64, device='cuda').reshape(1, 1801)
        loss = self.cost_function(self.init_pattern, self.OPA_E2)
        loss.backward()

        return round(loss.item(), 4)

    def set_forward_loss(self, E2: list[float]):
        self.OPA_E2.data = th.tensor(E2, dtype=th.float64, device='cuda').reshape(1, 1801)
        loss = 1 / (self.OPA_E2[0][900] + 0.5 * self.OPA_E2[0][1211] + 0.5 * self.OPA_E2[0][589])
        loss.backward()

        return round(loss.item(), 4)

    def update_weight(self, Epoch: int, obj: str, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_w = beta1 * self.m_w + (1 - beta1) * (self.obj[obj] * self.var_x.grad)
        self.v_w = beta2 * self.v_w + (1 - beta2) * th.square(self.var_x.grad)

        m_n = self.m_w / (1 - (beta1 ** Epoch))
        v_n = self.v_w / (1 - (beta2 ** Epoch))

        step_lr = self.lr * sqrt(1 - (beta2 ** Epoch)) / (1 - (beta1 ** Epoch))
        print(step_lr)
        self.var_x.data -= step_lr * m_n / (th.sqrt(v_n) + epsilon)
        self.var_x.grad.data.zero_()

    def decay(self, Epoch):
        print("\n!!! Decay Proceed !!!")
        self.lr *= self.decay_table[Epoch]
        self.div *= 0.5
