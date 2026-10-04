# 서러게이트 모델 2종
#
# 1) MLPSurrogate — 순수 블랙박스 회귀 (사용자 요구의 "MLP/DNN 회귀모델").
#    입력 인코딩: d 정규화 + (cosφ, sinφ) + 캐리어 힌트 sin/cos(k·xₙ·u_a) (u_a 앵커 8개).
#    캐리어 힌트는 입력 변환일 뿐(출력 그리드와 무관) — 수식 답을 주는 것이 아니라
#    고차원 진동 표현을 MLP가 만들 수 있게 하는 표준 positional-encoding 계열 처리.
# 2) NeuralElementSurrogate — 물리 내장 하이브리드 (비교군).
#    E(u) = Σₙ gₙ·exp(j(k·xₙ·u − φₙ))·EF(u)/N, gₙ = 이웃 간격 조건 공유망 (init 1).
#    구조상 사전학습 오차 ≈ 0에서 시작 — fine-tune에서 결합성 불일치를 흡수하는 역할.
#
# 공통 출력: (B, 2, G) = 정규화 복소장 (Re, Im), G=1801 (θ-균등, MODE 격자)
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

U_ANCHORS = (1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125)


def encode_inputs(d, phi, k, d_mid=3.5, d_half=1.5):
    # d(B,31) µm, phi(B,32) rad → (B, 31+64+32·2·A) f32
    d_n = (d - d_mid) / d_half
    x = F.pad(d.cumsum(-1), (1, 0))                                # (B,32)
    feats = [d_n, th.cos(phi), th.sin(phi)]
    for ua in U_ANCHORS:
        arg = k * x * ua
        feats += [th.sin(arg), th.cos(arg)]
    return th.cat(feats, dim=1).float()


def input_dim(N=32):
    return (N - 1) + 2 * N + 2 * N * len(U_ANCHORS)


class MLPSurrogate(nn.Module):
    def __init__(self, cfg, n_grid, N=32, k=2 * math.pi / 1.55):
        super().__init__()
        self.k = k
        self.n_grid = n_grid
        dims = [input_dim(N)] + list(cfg.hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 2 * n_grid)

    def forward(self, d, phi):
        z = encode_inputs(d, phi, self.k)
        out = self.head(self.body(z))
        return out.reshape(-1, 2, self.n_grid)


class NeuralElementSurrogate(nn.Module):
    """물리 캐리어 exp(j(k·xₙ·u−φₙ)) 유지, 원소별 복소 게인만 학습 (init=1)."""

    def __init__(self, cfg, u_grid, ef_amp, N=32, k=2 * math.pi / 1.55):
        super().__init__()
        self.k = k
        self.N = N
        self.register_buffer('u', u_grid.float())
        self.register_buffer('ef', ef_amp.float())
        h = list(cfg.elem_hidden)
        dims = [3] + h + [2]      # 입력: [gap_left, gap_right, x/L] 정규화
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.gain = nn.Sequential(*layers[:-1])   # 마지막 GELU 제거
        nn.init.zeros_(self.gain[-1].weight)      # 출력 0 → 게인 (1+0)+j0 에서 출발
        nn.init.zeros_(self.gain[-1].bias)

    def forward(self, d, phi):
        # MPS는 복소 텐서를 지원하지 않으므로 g_c·exp(jα)를 cos/sin 실수 연산으로 전개
        #   (g_c = (1+g_re) + j·g_im, α = k·xₙ·u − φₙ — 수학적으로 복소판과 항등)
        x = F.pad(d.cumsum(-1), (1, 0))                            # (B,N)
        gl = th.cat([d[:, :1], d], dim=1)                          # 왼쪽 간격 (경계=복제)
        gr = th.cat([d, d[:, -1:]], dim=1)
        L = x[:, -1:].clamp(min=1.0)
        feat = th.stack([(gl - 3.5) / 1.5, (gr - 3.5) / 1.5, x / L], dim=2).float()
        g = self.gain(feat)                                        # (B,N,2)
        ang = (self.k * self.u.reshape(1, -1, 1) * x.unsqueeze(1).float()
               - phi.unsqueeze(1).float())                         # (B,G,N)
        ca, sa = th.cos(ang), th.sin(ang)
        g_re = (1.0 + g[..., 0]).unsqueeze(1)                      # (B,1,N)
        g_im = g[..., 1].unsqueeze(1)
        re = (g_re * ca - g_im * sa).sum(dim=2) * self.ef.unsqueeze(0) / self.N
        im = (g_re * sa + g_im * ca).sum(dim=2) * self.ef.unsqueeze(0) / self.N
        return th.stack([re, im], dim=1)                           # (B,2,G)


def build_model(cfg, oracle):
    if cfg.arch == 'mlp':
        return MLPSurrogate(cfg, oracle.n_grid, oracle.jcfg.line_N, oracle.k)
    if cfg.arch == 'element':
        return NeuralElementSurrogate(cfg, oracle.u, oracle.ef,
                                      oracle.jcfg.line_N, oracle.k)
    raise ValueError(cfg.arch)
