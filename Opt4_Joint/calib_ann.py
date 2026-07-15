# 역매핑 ANN 위상 캘리브레이션 — 원본 Opt3 "Back_Forward"의 미구현 약속을 실현
#
# 목표: 카메라 프레임 2장 → 위상오차 ε 직접 추정 (비반복). 문헌 SOTA Leng 2022
#   (Photonics Research 10(2):347) 가 N=16에서 2프레임 달성. 우리 N=32 재현 시도.
#
# 우리 5프레임 Adam 과의 차이 = amortization:
#   Adam  : 매 칩마다 처음부터 적합 (인스턴스별 최적화) → 모호성 해소에 프레임 다수 필요
#   ANN   : 오프라인에 역매핑을 학습 → 학습된 prior가 모호성을 해소 → 추론 1회
#
# 이전 실패("NN surrogate 착취")와 다른 점: surrogate를 목적함수로 삼아 최적화기가
#   모델오차를 파고든 게 아니라, 지도학습된 역매핑이고 출력은 물리(주엽 복원율)로 검증.
#
# 모호성 처리 (Leng 2022 핵심):
#   - 주기 모호성: ε 를 직접 회귀하지 않고 (cos ε, sin ε) 로 출력 → 2π 불연속 제거
#   - 켤레 모호성: |E|² 는 특정 대칭에서 ε ↔ −ε 를 구분 못함 → 0이 아닌 고정 프로브
#     마스크를 건 2번째 프레임이 대칭을 깸
#
# 하드웨어 고정: w=1µm, d_min=2µm, N=32, λ=1.55µm, 균일진폭, 송신전용. 위상만 조율.
# 실행: venv/bin/python calib_ann.py
import json
import math
import os

import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float32                      # 학습은 float32
LAM, W_EL, N = 1.55, 1.0, 32
K = 2 * math.pi / LAM
NG = 361                             # 패턴 다운샘플 (1801 → 361)
U = th.sin(th.deg2rad(th.linspace(-90, 90, NG, dtype=DT)))
EF = th.sinc(W_EL * U / LAM)
EF_TRUE = th.exp(-(math.pi * 0.5 * U / LAM) ** 2) * th.sqrt((1 - U ** 2).clamp(min=0))

th.manual_seed(0)
X = th.arange(N, dtype=DT) * 3.0                       # 등간격 3µm (원본 조건, 하드웨어 고정)
PROBE1 = (2 * math.pi * th.rand(N, generator=th.Generator().manual_seed(7)) - math.pi).to(DT)
PROBE1[0] = 0.0                                        # 고정 프로브 마스크 (켤레 대칭 파괴)
PROBES = [th.zeros(N, dtype=DT), PROBE1]               # 프레임 2장


def render(eps, probes, ef, noise=0.0, gen=None):
    """(B,N) ε → (B, F*NG) 강도 패턴. 미분 불필요(데이터 생성)."""
    outs = []
    for p in probes:
        tot = eps + p.reshape(1, -1)
        ph = K * U.reshape(1, -1, 1) * X.reshape(1, 1, -1) + tot.unsqueeze(1)
        I = (ef.reshape(1, -1) * th.exp(1j * ph).sum(-1).abs() / N) ** 2
        if noise > 0:
            I = I * (1 + noise * th.randn(I.shape, generator=gen, dtype=DT))
        outs.append(I.clamp(min=0))
    return th.cat(outs, 1)


def make_data(n, ef, noise=0.01, seed=0):
    g = th.Generator().manual_seed(seed)
    eps = (2 * math.pi * th.rand(n, N, generator=g, dtype=DT) - math.pi)
    eps[:, 0] = 0.0                                    # 채널 0 = 기준
    xb = render(eps, PROBES, ef, noise, g)
    yb = th.cat([th.cos(eps), th.sin(eps)], 1)         # 주기 모호성 제거
    return xb, yb, eps


class Net(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = th.nn.Sequential(
            th.nn.Linear(2 * NG, 1024), th.nn.SiLU(),
            th.nn.Linear(1024, 1024), th.nn.SiLU(),
            th.nn.Linear(1024, 512), th.nn.SiLU(),
            th.nn.Linear(512, 2 * N))

    def forward(self, x):
        return self.f(x)


def to_eps(y):
    c, s = y[:, :N], y[:, N:]
    return th.atan2(s, c)


def recovery(eps_true, eps_hat):
    """주엽 복원율 = |Σ exp(j(ε−ε̂))|²/N²  (보정 후 잔차 위상)"""
    r = eps_true - eps_hat
    return (th.exp(1j * r).sum(1).abs() ** 2 / N ** 2)


def main():
    print('학습 데이터 생성...')
    xtr, ytr, _ = make_data(30000, EF, seed=1)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-8
    xtr = (xtr - mu) / sd

    net = Net()
    opt = th.optim.Adam(net.parameters(), lr=1e-3)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    print('학습...')
    for ep in range(60):
        perm = th.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(xtr), 256):
            j = perm[i:i + 256]
            opt.zero_grad()
            loss = ((net(xtr[j]) - ytr[j]) ** 2).mean()
            loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f'  epoch {ep+1:2d} | loss {tot/(len(xtr)//256):.5f}')

    rep = {}
    print('\n=== 평가: 2프레임 ANN 추론 (비반복) ===')
    for tag, ef in (('모델 일치', EF), ('모델 불일치(실제=Gaussian×obliquity)', EF_TRUE)):
        xte, _, eps = make_data(500, ef, seed=99)
        with th.no_grad():
            eh = to_eps(net((xte - mu) / sd))
        r = recovery(eps, eh)
        rep[tag] = {'frames': 2, 'recovery_mean': round(r.mean().item(), 4),
                    'recovery_p10': round(r.kthvalue(50).values.item(), 4),
                    'recovery_min': round(r.min().item(), 4)}
        print(f"  {tag:36s} | 복원율 평균 {r.mean():.4f} | p10 {r.kthvalue(50).values:.4f} | 최저 {r.min():.4f}")

    # ANN 출력을 초기값으로 1프레임 추가 미세보정하면? (하이브리드)
    json.dump(rep, open(os.path.join(RESULTS, 'calib_ann.json'), 'w'), indent=2, ensure_ascii=False)
    print('\n→ results/calib_ann.json')


if __name__ == '__main__':
    main()
