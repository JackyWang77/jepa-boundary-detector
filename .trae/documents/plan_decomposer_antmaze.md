# 目标规划：分解器双环境验证 → 判据学论文 arXiv 闭环

> 日期：2026-10-21 ｜ 状态：待用户批准 ｜ 上游：PushT 分解器已完成（判据 1/4 + 零模型成立）

---

## 0. 一句话任务定义

**用已自检的分解器管线在 AntMaze（谱分离强环境）上完成 draft.md §6 预登记的"事前预测"第二次验证，然后把论文 §6 从"将来时"改写为"完成时"，形成可投 arXiv 的完整闭环。**

## 1. 方向判断：对，且比昨天更对

论据（全部有落盘证据，非观点）：

1. **PushT 结果不是失败，是事前预测的第一次应验**。draft.md §6（2026-10-20 写定，先于任何真实底座实验）明确预测："AntMaze 谱分离强→预期成功；Push-T 谱温和→预期难"。实测 PushT：C1/C2/C3 失败、C4 反超、快带零模态——"难"应验，且失败模式被精确定位（快内容无相干动力学、U 分配力 winner-take-all）。判据学论文的核心卖点就是"能事前预言何时失败"，第一次预言命中。
2. **管线可信性已闭环**：正对照 4/4（合成 8 带系统全判据通过）、零模型成立（时间打乱 C2=−0.237、谱塌缩）。剩下唯一未验证的是环境依赖性——这正是 AntMaze 复线要回答的。
3. **资产全部就绪**：AntMaze 双底座缓存已存在（LeWM 384d × 25110 帧 × 310 eps；H-JEPA 三层各 384d），DMD 基线已跑完（dmd_antmaze.json），分解器管线参数化程度足够（d_in、标签、数据路径是仅有的环境耦合点）。
4. **论文骨架 90% 完成**（draft.md §1-§7 + 图表清单），§6 是唯一"将来时"章节。PushT 已是半章，AntMaze 是另一半。

结论：当前路线 = 论文主张（判据学 + 事前预测能力）的最短证明路径。不转向、不扩题。

## 2. 北极星与本轮目标

- **北极星（不变）**：判据学论文投 arXiv——"谱无隙六重复证 + H-标准审计 + 组织层适配器在真实开源底座上的成败预测与验证"。
- **本轮可执行目标（M1-M4）**：
  - M1：分解器适配 AntMaze 双底座（LeWM z_native、H-JEPA l1），代码改动 ≤80 行
  - M2：跑前登记事前预测（写进结果 JSON 的 `prediction` 字段），跑 4 个实验（2 底座 × 真实/零模型）
  - M3：按预注册判据 C1-C4 打分、出图、memory 归档——成败都如实记录
  - M4：draft.md §6 重写为完成时（含两轮真实底座结果 + 预测应验表），§7 局限更新

**明确不做（防扩题）**：不修 PushT 慢带垄断（属新方法一轮，需新预注册，写入 future work）；不重跑 PC（管线已自检）；不动 H-标准审计章节；不从 pixel 端重训。

## 3. 现状盘点（已验证事实）

| 资产 | 状态 | 位置 |
|---|---|---|
| 分解器管线（measure-then-learn + F=U−T·S+anchor+X+C+B） | PC 4/4 PASS | external/decomposer.py |
| PushT 结果（C4 反超 0.809<0.953；慢带垄断；快带 0 模态） | 已落盘已归档 | external/decomposer_pushT.{json,png} |
| PushT 零模型（C2=−0.237 成立） | 已落盘 | external/decomposer_pushT_null.json |
| AntMaze LeWM 缓存：z_native (25110,384), xy, starts/ends, 310 eps | 已验证 keys | external/hjepa_ant_lewm_cache.npz |
| AntMaze H-JEPA 缓存：z_l1 (25110,384) / l2 / l3, xy, 310 eps | 已验证 keys | external/hjepa_ant_hjepa_l3_cache.npz |
| AntMaze DMD 基线（手放层级无谱簇，gap≤1.37） | 已落盘 | external/dmd_antmaze.{json,png} |
| 论文骨架（判据学定位、§6 B1 事前预测已写） | v0.1 | paper/draft.md |

关键格式差异（适配点）：d_in 192→384；内容标签 PushT(block 4d/agent 2d) → AntMaze(**慢标签=xy 位置 2d，快标签=xy 一阶差分即速度 2d**)；episode 数 200→310（train/val 按同比例 80%：248/62）；episode 均长 ~81 步（STRIDES [1..32] 仍合法，无需改）。

## 4. 实施步骤（决策完整， executor 无需再做选择）

### Step 1 — decomposer.py 数据适配（M1）

改动集中在三处，不动训练/判据逻辑：

1. `main()` 加 `--data {pusht, antmaze_lewm, antmaze_hjepa}` 参数（默认 pusht，保兼容）。
2. `load_data()` 抽出一个 `load_cache(path, z_key, starts_key, ends_key)`；AntMaze 分支构造标签：`block_label = xy`（慢内容），`agent_label = xy[1:]-xy[:-1]` 按 episode 内差分、首帧补零（快内容）。d_in 从 z 形状自动推断，`Decomposer(d_in=...)` 参数化（目前硬编码默认 192）。
3. `N_TRAIN` 改为按 episode 数 80% 动态计算。

### Step 2 — 事前预测登记（M2，跑实验之前写入，不可后补）

在结果 JSON 顶层加 `prediction` 字段，训练启动前先打印并落盘：

- **P-ant-1（C1）**：AntMaze 谱分离强（DMD 基线 gap 结构显著大于 PushT）→ dispR2 ≥ 0.37 预期**通过**
- **P-ant-2（C3）**：快带预期**有模态居住**（与 PushT 相反），C3 预期通过
- **P-ant-3（C2）**：不确定——xy 位置与迷宫动力学的时间尺度耦合是否足够强，是本轮真正的开放问题；如实记录
- **P-ant-4（零模型）**：C2 < 0.3 必须成立（管线判据，非科学主张）
- **P-ant-5（底座对比）**：H-JEPA l1 与 LeWM 同为帧级底座，预期定性一致；若不一致，差异本身是发现（底座影响 > 环境影响）

### Step 3 — 跑实验（M2）

顺序（MPS 单卡，两两并行）：

```bash
# 批次 1（并行）
python3 -u external/decomposer.py --sfa_init --data antmaze_lewm  > /tmp/dc_ant_lewm.log 2>&1
python3 -u external/decomposer.py --null --sfa_init --data antmaze_lewm > /tmp/dc_ant_lewm_null.log 2>&1
# 批次 2（并行）
python3 -u external/decomposer.py --sfa_init --data antmaze_hjepa > /tmp/dc_ant_hjepa.log 2>&1
python3 -u external/decomposer.py --null --sfa_init --data antmaze_hjepa > /tmp/dc_ant_hjepa_null.log 2>&1
```

预计每批 20-40 分钟（384d 输入、25110 帧，比 PushT 大 ~2×；若 MPS 内存吃紧则串行）。SFA 退化保护已在位（shuffled 数据模态不足时按模补齐）。

### Step 4 — 判据打分、出图、归档（M3）

- plot_decomposer.py 加 `--input/--out` 参数化（当前硬编码 PushT 文件名），每个真实跑生成一张三面板图
- 对照表：4 个 JSON 的 criteria + prediction 应验情况汇总（markdown 表，直接进论文 §6）
- memory 归档（project_memory.md 追加）：每条新事实一行，含负结果

### Step 5 — 论文 §6 升级（M4）

draft.md 编辑，不重写其他章节：

- §6 标题改为"组织层：真实开源底座上的适配与事前预测验证"
- 新增小节：分解器设计（一段，引用 decomposer.py docstring 的 PREREGISTERED 节）+ PushT 结果（预测应验"难"）+ AntMaze 结果（按 M3 实际）+ 预测应验总表
- §7 局限更新：慢带垄断、快带无相干动力学、C1 在稠密混合 latent 上未达成——全部如实写入
- 图表清单追加：decomposer_pushT.png、decomposer_antmaze_*.png、预测应验表

### 验收标准（Definition of Done）

- [ ] 4 个实验 JSON 全部落盘且含 prediction 字段；零模型 C2<0.3 两个都成立
- [ ] 事前预测应验表生成（5 条预测逐条 命中/未中/不适用）
- [ ] draft.md §6 为完成时语气，无"下一步跑 AntMaze"字样
- [ ] memory 归档完成；无后台进程残留

## 5. 风险与分支

| 风险 | 概率 | 应对 |
|---|---|---|
| AntMaze 快带同样无人住 → P-ant-2 未中 | 中 | 这本身是强结果："谱分离强的环境也不产生快带 Koopman 模态" → 快带问题从"环境属性"升级为"latent 表征的普遍属性"，论文主张反而加强。如实记录，不修法 |
| C2 两端都失败 | 中 | 慢带垄断复现 → 分配力问题成为论文 §7 头号开放问题，与"U 只会丢弃不会放置"理论预言互证 |
| 384d SFA 广义特征分解数值不稳 | 低 | 已有 keep 阈值 + 退化补齐；若 NaN 则加 1e-4 脊（记录为工程修正，不动判据） |
| 双并行 MPS 内存不足 | 低 | 改串行，总时长 +40 分钟，不影响判据 |

## 6. 与长期路线的关系

本轮结束后，论文达到"可投 arXiv"状态（draft.md 自定标准：无需等 B1 大规模适配）。后续线（不属本轮）：PushT 分配力修复（新一轮预注册：显式内容-谱匹配目标）、四任务适配扩展、英文精写投稿。
