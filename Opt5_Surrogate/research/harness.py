# 챔피언 발굴 연구 하네스 — 방법 모듈을 동일 규약으로 실행·판정한다 (규약: research/README.md)
#
#   python research/harness.py run m_multistart --tier B1 [--kw '{"R":256}'] [--tag name]
#   python research/harness.py board
import os
import sys
import json
import math
import time
import argparse
import importlib
import dataclasses
from types import SimpleNamespace

import torch as th

th.set_num_threads(int(os.environ.get('RESEARCH_THREADS', '3')))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from physics_oracle import PhysicsOracle                                     # noqa: E402
from opt_core import (DV_FINE, VDomain, coupling_penalty,                       # noqa: E402
                      d_to_logit, formula_hard_per_restart, logit_to_d,
                      resolve_device, sync, u0_vector)

SEEDS = tuple(range(42, 50))
TIERS = {'B1': 2e5, 'B2': 2e6}       # 단독 MPS 기준 약 3 s / 30 s
MAX_CANDIDATES = 64
OUT = os.path.join(os.path.dirname(HERE), 'results', 'research')


class Meter:
    # 평가 단위 = 설계 1개 × 전 각도 × 정밀 격자 순방향 1회. 역전파 포함 3단위.
    # 회차 = 순차 배치 평가 호출 수 (장비와 무관한 반복 횟수 — 병렬 폭은 회차를 늘리지 않는다)
    def __init__(self):
        self.used = 0.0
        self.rounds = 0

    def add(self, units):
        self.used += float(units)

    def round(self, n=1):
        self.rounds += n


class CountingVDomain(VDomain):
    def __init__(self, meter, *a, **kw):
        super().__init__(*a, **kw)
        self.meter = meter
        self.frac = DV_FINE / kw.get('dv', DV_FINE)

    def W2(self, d):
        grad = th.is_grad_enabled() and d.requires_grad
        self.meter.add(d.shape[0] * self.frac * (3 if grad else 1))
        if not getattr(self, 'quiet', False):
            self.meter.round()
        return super().W2(d)


def make_ctx(device, budget):
    o = PhysicsOracle()
    jc = o.jcfg
    meter = Meter()
    u0v = u0_vector()

    def vdomain(dv_mult=1.0, dtype=th.float32, dev=None):
        return CountingVDomain(meter, o, u0v, dv=DV_FINE * dv_mult,
                               device=dev or device, dtype=dtype)

    def penalty(d, w_barrier=None):
        # 최적화 압력용 커플링 페널티 — jc 를 변이하지 않고 barrier 만 바꾼 사본으로 계산
        j = jc if w_barrier is None else dataclasses.replace(jc, cpl_w_barrier=w_barrier)
        return coupling_penalty(d, j)

    return SimpleNamespace(o=o, jc=jc, u0v=u0v, device=device, budget=budget, meter=meter,
                           vdomain=vdomain, penalty=penalty,
                           d_to_logit=lambda d: d_to_logit(d, jc),
                           logit_to_d=lambda s: logit_to_d(s, jc),
                           s_center=float(d_to_logit(th.tensor(jc.d_init, dtype=th.float64), jc)),
                           sync=lambda: sync(device))


def judge(ctx, d):
    # 규약 판정: θ-격자 f64 hard worst-angle PSLL + 규약 기본 커플링 페널티
    d = d.double().cpu().clamp(ctx.jc.d_min, ctx.jc.d_max)
    p = formula_hard_per_restart(ctx.o, d, ctx.u0v)
    J = p + coupling_penalty(d, ctx.jc)
    i = int(J.argmin())
    return dict(J=float(J[i]), psll=float(p[i]), cpl=float(J[i] - p[i]),
                min_gap=float(d[i].min()), d=d[i].tolist())


def run_seed(module, tier, kw, tag, device, seed, threads=None):
    if threads:
        th.set_num_threads(threads)
    mod = importlib.import_module(module)
    budget = TIERS[tier]
    ctx = make_ctx(device, budget)
    th.manual_seed(seed)
    ctx.sync()
    t0 = time.perf_counter()
    cand = mod.run(ctx, seed, budget, **kw)
    ctx.sync()
    t = time.perf_counter() - t0
    cand = cand.reshape(-1, ctx.jc.line_N - 1)[:MAX_CANDIDATES]
    r = judge(ctx, cand)
    r.update(seed=seed, used=round(ctx.meter.used), rounds=ctx.meter.rounds,
             t_s=round(t, 2), device=device)
    print(f"  [{tag}] seed {seed} ({device}): J {r['J']:.3f}  PSLL {r['psll']:.3f}  "
          f"rounds {r['rounds']}  used {r['used']:.3g}  {r['t_s']}s", flush=True)
    return r


def run(module, tier, kw, tag, device, seeds, lanes=None):
    # lanes: 'mps,cpu,cpu' — 레인마다 프로세스 1개, 시드를 나눠 병렬 실행 (GPU 프로세스는 1개만)
    budget = TIERS[tier]
    if lanes:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        lanes = lanes.split(',')
        thr = max(1, int(os.environ.get('RESEARCH_THREADS', '3')))
        pools = [ProcessPoolExecutor(1, mp_context=mp.get_context('spawn')) for _ in lanes]
        futs = [pools[i % len(lanes)].submit(run_seed, module, tier, kw, tag,
                                             resolve_device(lanes[i % len(lanes)]), sd, thr)
                for i, sd in enumerate(seeds)]
        rows = [f.result() for f in futs]
        for p in pools:
            p.shutdown()
    else:
        rows = [run_seed(module, tier, kw, tag, device, sd) for sd in seeds]
    J = th.tensor([r['J'] for r in rows], dtype=th.float64)
    used = max(r['used'] for r in rows)
    summary = dict(tag=tag, module=module, tier=tier, kw=kw, device=device,
                   J_mean=round(float(J.mean()), 4), J_best=round(float(J.min()), 4),
                   J_worst=round(float(J.max()), 4),
                   J_std=round(float(J.std()) if len(J) > 1 else 0.0, 4),
                   used_max=used, over_budget=used > 1.1 * budget,
                   rounds_max=max(r['rounds'] for r in rows),
                   t_mean=round(sum(r['t_s'] for r in rows) / len(rows), 2), rows=rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{tag}__{tier}.json'), 'w') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"[{tag}/{tier}] J mean {summary['J_mean']:.3f}  best {summary['J_best']:.3f}  "
          f"worst {summary['J_worst']:.3f}  std {summary['J_std']:.3f}  "
          f"rounds {summary['rounds_max']}  used {used:.3g}"
          f"{' OVER' if summary['over_budget'] else ''}  t {summary['t_mean']}s")
    return summary


def board():
    rows = []
    for fn in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        if fn.endswith('.json'):
            rows.append(json.load(open(os.path.join(OUT, fn))))
    for tier in TIERS:
        rs = sorted((r for r in rows if r['tier'] == tier), key=lambda r: r['J_mean'])
        if not rs:
            continue
        print(f"\n== {tier} (예산 {TIERS[tier]:.0e} 단위) — J 평균 오름차순")
        print(f"{'tag':34s} {'n':>2s} {'mean':>7s} {'best':>7s} {'worst':>7s} {'std':>5s} "
              f"{'rounds':>6s} {'used':>8s} {'t[s]':>6s}")
        for r in rs:
            print(f"{r['tag']:34s} {len(r['rows']):2d} {r['J_mean']:7.3f} {r['J_best']:7.3f} "
                  f"{r['J_worst']:7.3f} {r['J_std']:5.3f} {str(r.get('rounds_max', '-')):>6s} "
                  f"{r['used_max']:8.3g}{'!' if r['over_budget'] else ' '}{r['t_mean']:6.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('module')
    r.add_argument('--tier', default='B1', choices=list(TIERS))
    r.add_argument('--kw', default='{}')
    r.add_argument('--tag', default=None)
    r.add_argument('--device', default='auto')
    r.add_argument('--seeds', default=None, help='예: 42,43 (기본 42..49)')
    r.add_argument('--lanes', default=None, help="시드 병렬 레인, 예: 'mps,cpu,cpu'")
    sub.add_parser('board')
    a = ap.parse_args()
    if a.cmd == 'board':
        board()
    else:
        seeds = tuple(int(x) for x in a.seeds.split(',')) if a.seeds else SEEDS
        run(a.module, a.tier, json.loads(a.kw), a.tag or a.module,
            resolve_device(a.device), seeds, a.lanes)
