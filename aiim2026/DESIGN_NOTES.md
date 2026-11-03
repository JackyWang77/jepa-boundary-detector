# 设计依据笔记（论文素材，写作期直接引用）

> 每一条设计决策 + 证据数字。所有数字回源：features/meta.json、segments/e3/、runs/。
> 与 EXPERIMENT_GUIDE.md（执行协议）和 PROGRESS.md（进度）互补，本文档只记"为什么这么设计"。

## 1. 三路特征的物理动机（方法节 B 素材）

**底层问题**：冻结世界模型的潜空间里，物理相变（接触 / 脱离 / 块起停）会在哪些统计量上留下痕迹？

**三轴正交性论证**——潜变量增量流 Δz_t = z_t − z_{t−1} 上，一个变化可分解为幅值、方向、可预测性三个正交分量，每路特征盯一个：

| 路 | 特征 | 轴 | 存在性场景（只有它能抓的物理事件） |
|---|---|---|---|
| f1 | 滑窗 ‖Δz_t‖ 均值/标准差 | 幅值 | 块被推动：画面结构剧变 → 增量能量激增，但方向可以不变 |
| f2 | 相邻增量余弦 cos(Δz_t, Δz_{t−1}) 滑窗统计 | 方向 | 推着推着拐弯：Δz 幅值不变、方向持续偏转 → 余弦走低 |
| f3 | innovation ‖z_{t+1} − p(z_{t−2..t}, a_{t−2..t})‖ 及滑窗统计 | 可预测性 | 接触瞬间：预测器按"上一段动态"外推，regime 突变使其失效 → innovation 跳升，即使幅值方向都连续 |

三路任何一路都有其他两路抓不住的专属场景——这是"为什么恰好是三个"的论证，也是 E5 特征消融表（7 行）的预期结构。

**f3 的有效性实测依据**（E2 全 corpus，500 条 / 12,187 模型步）：
- innovation 中位数 = ‖Δz‖ 中位数的 **19%** → 冻结 predictor 比什么都不预测好 ~5 倍，常规动态预测得准，"预测崩"才携带信息
- 分布：f1 p50=5.05（p10=1.99 / p90=8.90）；f2 cos p50=0.58（p10=0.11）；f3 p50=0.98（p90=1.65 / p99=3.31）

## 2. 关键工程口径（实现细节段素材，全部从源码核实）

| 口径 | 内容 | 依据 |
|---|---|---|
| 时间基 | **frameskip=5**：模型步 = 5 个原始帧，每步 action = 5 个原始动作拼成 10 维 | 训练配置 scripts/train/config/data/pusht.yaml；action_encoder.input_dim=10 = frameskip × action_dim(2) |
| 窗口语义 | 模型步 k 用原始帧 5k；action 块 k = 原始 action[5k..5k+4] | Dataset.__getitem__ + HDF5Dataset._load_slice（action 不做步进采样，其余列 [::frameskip]） |
| predictor 用法 | predict(emb, **act_emb**)：action 必须先过 action_encoder；pos_embedding 仅 3 位置 → 只能 3 帧窗推理 | lewm.py predict / module.py Predictor |
| action 归一化 | 全数据集列 z-score，std+1e-2 | get_column_normalizer（训练管线）；eval_wm.py StandardScaler 同口径 |
| 像素变换 | /255 → ImageNet 归一化 → 224 | get_img_preprocessor / eval_wm.py 一致 |
| 潜变量维度 | 192（ViT-tiny CLS token 过 projector） | checkpoint config.json |

**一致性原则**：E2 特征提取的每一步前向都严格复刻训练时口径——口径差一点，潜变量全变味。三路特征也活在模型步上，与 E4 在线检测、E6 边界臂的时间基天然一致。

## 3. 参考标签（programmatic reference labels）的修正依据（实验 setup 素材）

**原方案为什么被推翻**（指导书 E5 初版写"接触 = agent-块距离越过几何半径和"）：
- 实证反常：块在 agent-块**中心距 50-90px** 时高速移动（>2px/步）的帧有 1,989 个，比 dist<50 的 1,105 个还多——固定几何阈值下这等于"隔空推块"
- 根因（读 env 源码 stable_worldmodel/envs/pusht/env.py）：**块是 8 种多方块形状族**（'o','L','T','Z','square','I','small_tee','+'），scale∈[20,60]，T 形最远角距中心 2.24×scale ≈ 45-134px——推块边缘时中心距天然在 50-90px；且 `space.damping=0`，块被推后无阻力滑行
- state schema 本身无误：pair01=agent（与速度维相关性 0.989 验证）

**修正方案**（形状无关、直接可观测）：push_onset = 块位移 >0.5px/步持续 2 步；settle = 运动结束后 <0.1px/步持续 3 步首帧。
- 噪声地板实测：agent>120px 且块已静默 5 步的帧，位移 p99=0.89px，>0.5px 占 1.4% → 0.5 阈值 + 2 步持续条件干净
- 事件计数：500 条轨迹 PELT 时间基上共 2,808 个事件（推起 + 停住）

**论文写作提醒**：描述环境必须写清块形状族（不是经典小 T PushT）；"接触"类标签不可用中心距离定义。

## 4. 切分 teacher 的选择（方法节 C 素材）

- **cost = normal（非 rbf）**：BIC 惩罚只对似然类代价有定义——这是原则性依据；smoke 实测 rbf 在同惩罚下退化为 0 切点（双重确认）
- **惩罚 = BIC**：0.5·(d + d(d+1)/2)·log(n)，6 维 → 13.5·log(n)，零 PushT 调参
- **BinSeg / BottomUp 与 PELT 切点集逐条 100% 一致**（500/500）——BIC 惩罚下三种搜索收敛到同一解，切分对搜索方法鲁棒（是真正的不动点）。E5 方法对比表如实报告
- **BOCPD 自实现**（ruptures 无此方法、不允许新依赖）：Adams-MacKay 标准公式，对角高斯、obs var 1、先验 N(0,1)（信号已 z-score）、hazard=1/PELT 中位段长（粒度对齐），超参全录 meta.json
- 切分量级：PELT 每条中位 3 个切点（共 1,745）；BOCPD 更保守（693）
- **对齐预览**（非正式 E5）：79.5% PELT 切点落在物理事件 ±2 模型步内；78.6% 事件被覆盖 → "变点切分即物理事件"叙事大概率成立

## 5. 在线检测头的设计空间（方法节 C + 消融动机素材）

**三条硬约束**（由方法需求导出，非偏好）：
1. **因果**——E6 执行环内在线推理，每模型步出一次 p(boundary)，不能看未来帧
2. **O(1) 单步开销**——离线全序列方法（含 PELT 自身）不可用，这正是"蒸馏"存在的理由
3. **2-5k 参数**——学生必须远小于老师（特征流 + PELT 离线寻优），否则"轻量"叙事不成立

**记忆三档覆盖**（CNN vs GRU 的本质）：

| 记忆档 | 头 | 参数 | 角色 |
|---|---|---|---|
| 无记忆 | 固定阈值 / 逻辑回归 | ~0 | E5 基线行，回答"到底需不需要记忆" |
| 有限（感受野 ≈9 步） | 因果 1D CNN（2 层 × kernel 5，16 通道） | ~1.8k | E4 候选一："看最近一段就够" |
| 无界 | tiny GRU（hidden 16） | ~1.2k | E4 候选二："要记住 regime 持续多久" |

"都试，报好的" = 实验测**哪种记忆假设对这路信号重要**，本身是消融的一部分。

**排除清单**（写作时如被审稿人问起）：
- tiny Transformer：一层 attention（d=64）QKV+output ≈ 16k+ 参数，超预算 3-8 倍；序列仅 ~20 模型步
- MLP 滑窗输入：定窗无权重共享 = 退化 CNN，被 CNN 严格包含
- TCN：带空洞卷积的 1D CNN，同一家族（嫌 9 步感受野短加 dilation 即可），非独立选项
- HMM / BOCPD / CUSUM：teacher 家族（E3 对照）；其分布假设和超参正是蒸馏要消除的——用它们当学生等于没蒸馏

**推理规则**：p(boundary) > τ=0.5 判边界 + 5 步不应期（两次边界最少隔 5 步）。

## 6. 术语纪律（写作红线，违反 = 论文被拒级）

| 概念 | 必须这样说 | 禁止 |
|---|---|---|
| 训练方式 | **label-free training**（伪标签来自 PELT 自动切分） | unsupervised |
| 验证标签 | **programmatic reference labels**（从模拟器 state 几何规则计算） | 人工标注 / ground truth |
| 方法性质 | 训练回路无标签信号；**特权信息（state）只在评估侧使用** | fully unsupervised |
