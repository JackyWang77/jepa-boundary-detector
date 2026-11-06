import json, numpy as np
srs = {}
for s in [42, 43, 44]:
    import os
    p = rf'd:\Research-2026-jepa-boundary-detector\aiim2026\runs\flat_h15r15_d25_s{s}_n150.json'
    if not os.path.exists(p):
        print(f's{s}: NOT YET')
        continue
    j = json.loads(open(p).read())
    sr = j['success_rate'] / 100.0
    succ = int(sum(j['episode_successes']))
    srs[s] = sr
    print(f's{s}: {sr:.1%} ({succ}/150)')

if srs:
    vals = list(srs.values())
    print(f'\nmean={np.mean(vals):.1%}  range=[{np.min(vals):.1%}-{np.max(vals):.1%}]')

    print(f'\nd25 grid COMPARE (plan budget=50):')
    print(f'  flat(5,5)    = 85.6%  (receding=5 -> replan 10x)')
    print(f'  flat(10,10)  = 45.6%  (receding=10 -> replan 5x)')
    if len(srs) >= 2:
        print(f'  flat(15,15)  = {np.mean(vals)*100:.1f}%  (receding=15 -> 0 replan in budget)')
    print()
    print(f'DROPS: h5->h10 = 40.0pp  |  h10->h15 = {45.6 - np.mean(vals)*100:.1f}pp')
