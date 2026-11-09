# AIIM2026 投稿前补充实验清单（4060 机器）· 收敛版 2026-11-08

> 范围已由用户裁决收敛：只做 E1 + E3，另外顺手 push 一个 json。
> E2 / E4 / E5 / E6 / E7 / E8 全部砍掉，不再执行。
> 做完一项就 push（沿用同一 remote），并在
> `4060-HANDSHAKE-20261105.md` 追加产出与结论；Mac 侧 pull 后更新正文与脚注。

## 收敛后的范围

| 编号 | 实验 | 优先级 | 目的（对应审稿风险） |
|---|---|---|---|
| E1 | per-seed 成功率表 + 缺失 json + `cube_r2_dual.json` | P0 | "方差/稳定性呢" |
| E3 | PACE baseline arm | P0 | Related Work 讨论了 PACE 却没有同协议比较 |

已砍：E2 τ/ρ 敏感性、E4 PELT 蒸馏损失、E5 噪声 baseline、E6 Cube 双岗分离、
E7 跨任务迁移、E8 planning overhead。

环境沿用现有 python；数据路径 `D:\stable_worldmodel\datasets`。
脚本目录 `aiim2026\src`，结果一律写 `aiim2026\runs\`。

---

## E1. per-seed 成功率表（5 分钟，直接跑）

脚本已在 Mac 写好并验证过 schema：`src/make_per_seed_table.py`。

```powershell
python src\make_per_seed_table.py runs results\cube_probe_raw
```

- 输出 `results\per_seed_success.json`（+ 控制台 markdown 表 + 缺失清单）。
- 脚本按 (task, arm, d, seed) 取 episode 数最大的 run，因此 `_n150` 版本会
  正确覆盖 Mac 本地的 n=50 pilot；**不要手动混入** `probe_t_hr_d75_s42.json`
  这类 pilot 文件去覆盖。
- **若脚本报告缺格，把缺失对应 json 从 runs 目录 push 上来**（Mac 本地缺：
  PushT 的 d25/d50 全部 flat 与 probe/boundary/decoupled、d75 的 s43/s44
  boundary/decoupled，以及 Cube 的 flat d25/d50/d75 部分 seed）。
- 同时 push：`results\cube_r2_dual.json`（GRU vs CNN sweep 对比表，Mac 本地
  只有 PushT 的 `r2_dual.json`，缺 Cube 的；正文要它作"为什么 Cube 用 GRU"
  的脚注数字）。

## E3. PACE baseline（脚本已写好，直接跑）

PACE 是 training-free 的：只读 CEM 每次规划输出的 predicted action chunk，
按速度 profile 的低速谷底选截断点重锚，不进 CEM 循环、不用 latent head、
不 retrain（arXiv:2606.00537）。

Mac 侧已落地三个文件：

| 文件 | 作用 |
|---|---|
| `src/pace_arm.py` | `PACEPolicy`：固定 h=10，拦截 CEM chunk，按 θ·median(s) 选执行块数 |
| `src/e6_eval.py` | 新增 `+arm=pace`（参数 `+pace_theta`、`+pace_min_blocks`） |
| `src/run_pace.py` | 18 runs 驱动器，resume-safe，失败即停 |

规则（与 E3 规格一致）：

1. 规划 horizon 固定 10 块（= probe 臂上限）；
2. 每次 CEM 得到 a_1..a_10，每块 a_i 含 B=5 个 primitive 动作；
3. 块速度 `s_i = mean_j ||a_ij||_2`（先反标准化回原始动作单位）；
4. 取最早满足 `s_i <= θ · median(s_1..s_10)` 的块 i（i ∈ [2, 9]）作为
   re-anchor 点；无谷底则执行完整 10 块（退化为 flat h=10）；
5. θ ∈ {0.5, 0.3}。

运行：

```powershell
python src\run_pace.py            # 两个 θ 都跑，18 runs
python src\run_pace.py --theta 0.5   # 只跑主档
```

- 范围：PushT，d∈{25,50,75} × 3 seeds × 150 eps，与 flat/probe 完全同协议
  （`eval_budget=2d`，`num_eval=150`，seeds 配对以复用 McNemar）。
- 产出：`runs/pace_th05_d{d}_s{seed}_n150.json`、`pace_th03_...`，json 内含
  `pace_theta` 与逐 episode 的 `keep` 日志。
- 结果回来后：Table I 加一行 PACE（取两档中较优的作为主条目，另一档进脚注）。
- Cube 不跑（Cube 主表结论已定性为节律条件性，PACE 在 Cube 上的对照不增值）。

---

## 完成后统一动作

1. `git add aiim2026/runs aiim2026/results aiim2026/src` + commit + push；
2. 更新根目录 `4060-HANDSHAKE-20261105.md`：每个编号一行 [状态 + 关键数字]；
3. Mac 侧 pull 后负责把 E1 表放进 Table I/II 脚注、E3 结果加 PACE 行。
