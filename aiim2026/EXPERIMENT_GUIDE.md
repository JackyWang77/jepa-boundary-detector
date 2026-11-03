# AIIM 2026 实验指导书（唯一权威版本，执行前必读）

> 本文档是实验执行的**唯一权威指导**。任何执行 agent 在动手前必须完整读完本文档。
> 与本文档冲突的任何"自己的想法"、"优化"、"简化"一律不允许。
> 有疑问停下来问用户，不要自行发挥。
> **规则零**：收到本指导书后，第一动作是在 `aiim2026/PROGRESS.md` 追加一行"已读指导书 v1（日期）"，然后从"3. 当前状态"指出的第一个未完成项开始，按顺序执行。禁止跳步、禁止并行开多个实验、禁止先做 E6。

## 0. 项目一句话

在冻结 LeWM 世界模型的潜变量流上，用无监督变点切分（PELT）自动发现物理边界，蒸馏成轻量在线检测头，检测到边界就停止执行并重锚重规划。**验证一件事：边界触发的停止比不停止（平坦规划）和等频率停止的规划成功率高。**

截止：**2026-11-13 23:59**（AIIM 2026），目标 11-12 提交。论文 4 页 IEEE 双栏，骨架已在 `Templates/Conference-LaTeX-template_10-17-19/conference_101719.tex`（标题、作者、参考文献已定，正文等实验数字）。

## 1. 术语纪律（违反 = 论文被拒级别错误）

| 概念 | 必须这样说 | 禁止这样说 |
|---|---|---|
| 训练方式 | label-free training（伪标签来自变点检测自动切分） | unsupervised（不押注这个词） |
| 验证标签 | programmatic reference labels（从模拟器 state 几何规则计算） | 人工标注 / ground truth 标注 |
| 方法性质 | 训练回路无标签信号；特权信息（state）只在评估侧使用 | fully unsupervised / 无任何监督 |

关键事实：训练用伪标签（PELT 自动切分），**不是**规则标签。规则标签只在 E5 验证对齐率时出场。

## 2. 环境与资产（已就位，不要重复下载/转换）

| 资产 | 路径 | 状态 |
|---|---|---|
| conda 环境 | `aiim2026`（Python 3.10.22） | ✓ 已装 stable-worldmodel |
| python | `/Users/wanglanlan/miniforge3/envs/aiim2026/bin/python` | |
| checkpoint | `~/.stable-wm/pusht/lewm_object.ckpt` | ✓ 已转换，勿动 |
| 数据集 | `~/.stable-wm/pusht_expert_train.h5` → 软链到 `external/lewm-pusht-data/pusht_expert_train.h5`（46GB，18685 episodes，233 万帧，keys: action/pixels/state/proprio/ep_len/ep_offset/episode_idx/step_idx，state 7 维） | ✓ 已验证可读 |
| 源码 | `aiim2026/third_party/le-wm`（eval.py + config）、`aiim2026/third_party/stable-worldmodel` | ✓ 已克隆 |
| 编译 | `/Library/TeX/texbin/pdflatex`（不在 PATH，用全路径） | ✓ |

网络：PyPI 直连和清华镜像均可；HF 走 hf-mirror.com；GitHub 走 gh-proxy.com。

## 3. 当前状态（2026-11-02 晚）

- E1 平坦基线：9 个任务（d25/50/75 × seed 42/43/44）跑过，**1 成功 8 失败**
  - 成功：`aiim2026/runs/flat_d25_seed42.log`（含完整 success 数组，**尚未统计成功率**）
  - 失败原因全部相同：`Error locating target 'stable_worldmodel.solver.CEMSolver'`。le-wm 的 `config/eval/solver/cem.yaml` 用旧版库路径，PyPI 新版库中 CEMSolver 在 `stable_worldmodel.planning.solver.cem`。
  - 注意成功日志用的 `solver.device=mps`，说明 Mac Metal 可行。
- E2-E7：未开始。
- 已有脚本：`aiim2026/src/convert_ckpt.py`（转换已完成，勿重跑）、`aiim2026/src/collect_baseline.py`、`aiim2026/src/run_baseline.sh`。

## 4. 实验清单（定稿，不得增删修改判据）

### E1 平坦基线（复现，harness 正确性门）

**目的**：复现 Mind the Gap 的平坦规划衰减，锚点成功率 **d=25/50/75 = 94.0/52.7/18.0**（3 seeds 平均）。

**步骤**：
1. 修复 solver 路径：先验证 `python -c "from stable_worldmodel.planning.solver.cem import CEMSolver"` 可导入；然后在启动命令加 override `solver._target_=stable_worldmodel.planning.solver.cem.CEMSolver`（若 hydra override 语法报错，就复制 `config/eval/solver/cem.yaml` 到本地改 `_target_` 字段）。
2. 先统计 flat_d25_seed42.log 的 success 数组（数 True 占比），看是否接近 94%。
3. 重跑 8 个失败任务。参数从失败日志头部抄：`policy=pusht/weights.pt`（或 lewm_object 路径，以成功日志的配置为准）、`eval.goal_offset_steps={25,50,75}`、`eval.eval_budget={50,100,150}`、`solver.device=mps`、`seed={42,43,44}`。
4. 输出写 `aiim2026/runs/`，每个任务一个日志 + 一个 JSON（含逐 episode success 布尔值）。

**验收**：三距离平均成功率趋势对齐 94.0/52.7/18.0，容差 ±5 个百分点。达不到先排查 harness（规划步数、CEM 样本数、goal 采样）再考虑环境随机性，**禁止**为了凑数改协议。

### E2 通用特征提取

**输入**：训练集抽 **500 条轨迹**（固定 seed 抽，落盘 episode id 列表）。
**处理**：每条轨迹的 pixels 过冻结 encoder 得潜变量序列 z_t；action 过 predictor 与下一帧真实编码算 innovation。
**三路特征（固定设计，勿改勿加）**：
- f1 增量能量：滑窗内 ‖Δz_t‖ 的均值/标准差（窗口 W=8，因果）
- f2 方向偏转：相邻增量余弦 cos(Δz_t, Δz_{t−1}) 的滑窗统计（同窗口）
- f3 innovation：‖z_{t+1} − p(z_t, a_t)‖ 及其滑窗统计
**输出**：`aiim2026/features/{ep_id}.npz`（含 z、f1/f2/f3、action、state），加一个 `meta.json`。

### E3 无监督切分（teacher）

**方法**：`ruptures` 库。PELT 为主（cost 用 rbf 或 normal，惩罚用 BIC 准则，**不针对 PushT 调参**）。对照：BinSeg、BottomUp、BOCPD（同一特征流）。
**输入**：E2 的三路特征拼接成多维序列。
**输出**：每条轨迹的变化点集合 → 伪标签；四方法各自的切分结果落盘。

### E4 蒸馏 student（在线检测头）

**结构**：因果 1D CNN（2 层、8–16 通道、kernel 5，目标 2–5k 参数）或 tiny GRU，二选一（都试，报好的）。
**训练**：伪标签（E3 切分点标 1，其余 0），BCE + pos_weight（约 1:30 不平衡），early stop 用留出 20% 轨迹的伪标签 F1。
**推理规则**：p(boundary) > τ 判边界 + **不应期**（两次边界最少隔 5 步）。
**输出**：`aiim2026/models/head.pt` + 训练曲线。

### E5 物理验证（先于 E6 跑，半天内完成）

**程序化参考标签**（从 state 几何规则计算，零人工）：
- 接触：‖agent_pos − block_pos‖ 从 >阈值 变 ≤阈值的那步（阈值 = agent 半径 + 块等效半径，由几何尺寸定，不调参）
- 脱离：反向跨越
- 航向突变：块位移向量滑窗方向变化率超阈
- 标签形式：逐帧 0/1；评测用事件级 F1，容差 ±2 步

**必做表（Table I 数据）**：
1. 对齐主表：PELT 切分点 vs 事件标签 F1；student vs 事件标签 F1；下界对照 = 随机时刻、等间隔时刻
2. 特征递增消融（7 行）：f1 / f2 / f3 / f1+f2 / f1+f3 / f2+f3 / 全部（切分方法固定 PELT）
3. 切分方法对比（4 行）：PELT / BinSeg / BottomUp / BOCPD（特征固定三路全量）
4. 头部对比（3 行）：固定阈值 / 逻辑回归 / 小 CNN
5. 分事件类型 breakdown：接触类 vs 航向类的 F1 分解（检验理论预测：f2 主导转向检出、f3 主导接触检出）
6. 三特征流两两相关系数矩阵

**叙事分支**：对齐率高（F1 > 0.7）→ "变点切分即物理事件"；低 → "统计动力学分段"（成功率照测，措辞降级）。**禁止**因为对齐率低就改特征/调参去凑。

### E6 三臂主对比（Table II 数据）

| 臂 | 行为 |
|---|---|
| 平坦（flat） | E1 的协议，从不主动停止 |
| 边界停止（ours） | 执行段中 student 检测到边界 → 立即停止执行 → 从最新真实观测重锚 + 重规划 |
| 等频率停止（control） | 固定间隔停止重锚，间隔 = 边界臂实测平均边界间隔（频率对齐，从日志核实） |

**规模**：d=25/50/75 × 3 seeds × 50 episodes。三臂协议除停止条件外**完全一致**（同一 CEM 配置、同一 goal 采样、同一 eval_budget）。
**输出**：每臂每距离每 seed 一个 JSON（成功率 + 逐 episode 布尔 + 平均重锚次数）。

### E7 机制图（Fig 2 数据）

选一条成功 episode：记录 rollout 预测 vs 真实潜变量的漂移曲线，展示重锚点处漂移归零。数据从 E6 边界臂的运行中顺手落盘（每步 rollout 漂移值）。

## 5. 通用纪律（每一条都有血泪教训）

1. **零造数字**：论文所有数字必须回源到 `aiim2026/runs/` 或 `aiim2026/results/` 的 JSON。没跑过就没数字，没数字就不写。
2. **判据预注册**：E5/E6 的验收标准如上，跑之前写死，跑完不许因为结果不好看而调整。
3. **负结果如实记录**：如果边界臂没赢等频率臂，如实写。EI 会议可发实证型论文，数据造假不可救。
4. **增量落盘**：所有实验边跑边写 JSON（每 episode 落盘），中断可续。
5. **Smoke 先行**：任何新脚本先用 2 条轨迹 / 2 episodes 试跑，输出文件加 `_smoke` 后缀，确认无害后再上正式规模。
6. **不覆盖**：正式产物命名带任务参数（如 `flat_d50_seed43.json`），禁止覆盖已有产物。
7. **底座冻结**：encoder/predictor 权重一个字节都不改。改了 = 整个故事崩塌。
8. **勿动资产**：`lewm_object.ckpt`、46GB h5、已克隆源码，不重新下载不重新转换。
9. **依赖最小**：新依赖只允许 `ruptures`（切分）。别的库先问。
10. **进度汇报**：每个实验完成后写一行结果摘要到 `aiim2026/PROGRESS.md`（格式：日期 | 实验 | 关键数字 | 结论）。

## 6. 已知坑位

| 坑 | 解法 |
|---|---|
| CEMSolver hydra 路径失效 | 见 E1 步骤 1 |
| `swm.__version__` AttributeError | 无害，忽略，库已装好 |
| mps 算子不支持 | 降级 `solver.device=cpu`（慢但稳） |
| h5 全量遍历慢 | 只读需要的 episode（用 ep_offset/ep_len 索引切片），不要全表扫 |
| pixels 是 (N,224,224,3) uint8 | encoder 前记得归一化（对照 le-wm eval.py 的预处理，保持一致） |
| wandb/网络钩子报错 | 环境变量 `WANDB_MODE=disabled` |

## 7. 时间表

| 日期 | 里程碑 |
|---|---|
| 11-03 | E1 修复 + 9 任务全绿 |
| 11-04 ~ 11-05 | E2 + E3 + E5 前半（对齐主表，**先看结果定叙事**） |
| 11-06 | E4 + E5 全部消融表 |
| 11-07 ~ 11-09 | E6 三臂 + E7 机制图 |
| 11-09 ~ 11-11 | 论文正文（全部数字回源） |
| 11-12 | 提交（缓冲 11-13） |

## 8. 论文骨架对应（写作时回填）

- Intro：问题（Mind the Gap 衰减数字）→ 问题重述（冻结潜空间是否已含边界信息）→ 贡献四件套
- Method：A 问题设定 / B 三路特征（正交性论证：幅值/方向/模型偏差三轴 + 存在性场景）/ C PELT teacher + student 蒸馏
- Experiments：E1 复现 → Table I（E5）→ Table II（E6）→ Fig 2（E7）
- 已核实文献 10 条在 tex 里，Basseville & Nikiforov 1993 待核实后加
