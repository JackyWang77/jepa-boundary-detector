# Prompt for the 4060 agent: PushT reduced-capacity cross-model experiment (reviewer R2)

> 复制以下内容发给 4060 上的 agent 执行。mac 侧不重开任何评测,只消费 JSON。
> 详细 runbook:`aiim2026/R2_CROSS_CAPACITY.md`(Phase 0–3),本 prompt 是执行摘要。

## 背景

AIIM 2026 审稿意见 R2:所有结论只用了一个冻结 JEPA 世界模型,要求跨模型泛化实验。
方案:自训一个**减半 Predictor 容量的 LeWM**(encoder 不动),在 PushT d=50/75 上跑
与主模型完全相同的 flat grid + probe 对比,正面回应 "replacing with different-capacity
JEPA"。公开渠道没有不同容量的 PushT 世界模型 checkpoint(已搜索确认),所以必须自训。

## Phase 0 — 训练小容量 LeWM(几小时)

1. `cd aiim2026/third_party/le-wm`
2. 复制 `config/train/lewm.yaml` → `config/train/lewm_small.yaml`,只改两处:
   - `model.predictor.depth: 6` → **3**
   - `model.predictor.mlp_dim: 2048` → **1024**
   其余一律不动(data=pusht;encoder 不动 → embed 192 不变,下游零改动)。
3. 按 le-wm README 训练(先 `train.py --help` 确认入口)。记录 epochs 数与 wall time。

为什么减半 predictor 而不是换 ViT-Small encoder:换 encoder 潜空间 192→384,检测头、
策略、CEM 距离代码全套要改,半天预算消化不了,还会引入第二个变量。减半 predictor
容量差足够回应审稿人。

## Phase 1 — 转换到 stable-worldmodel 格式

按 handshake §2 / EXPERIMENT_GUIDE §2 里主模型(quentinll/lewm-pusht →
`checkpoints/pusht/`)那套转换流程,输出 `checkpoints/pusht_small/{config.json,weights.pt}`。
转换前后各 `torch.load` 验证一次。

## Phase 2 — flat grid + 零样本 probe(约半天)

cwd = `aiim2026/third_party/stable-worldmodel/scripts/plan`,命令模板 = handshake §4,
只改 policy 路径为 `pusht_small/weights.pt`:

- flat:`h∈{5,10,15}` × `d∈{50,75}` × seed 42/43/44 × `eval.num_eval=150` → 18 个 json
- probe:同网格 + `+arm=probe_t` → 6 个 json
- out_json 命名:`aiim2026/runs/pusht_small_{arm}_d{d}_s{seed}.json`

注意:`runs/*.json` 在 .gitignore 里,完成后把 24 个 json 复制到
`aiim2026/results/pusht_small/` 再提交推送(mac 从 results 消费)。

## Phase 3 — 条件重蒸馏(仅当 Phase 2 probe 输给最优 flat)

1. `e2_extract_features.py` 用小模型 checkpoint → `features_small/`
2. `r2_dualhead.py` 跑 features_small(事件标签、门限一律不动)→ 小模型专用 head
3. 用新 head 重跑 6 个 probe

## 预注册判读(禁止事后挑数据)

- probe 在两个距离都显著超最优 flat → 检测头跨容量零改动迁移,最强结论
- probe 零样本输 → 进 Phase 3
- Phase 3 后仍输 → 如实报为 limitation,同样写进 PROGRESS

## 完成标准

- `results/pusht_small/` 下 24 个 json(flat 18 + probe 6;Phase 3 后为新的 probe 6)
- PROGRESS.md 一条留言板记录(日期 | [4060 agent] **pusht_small** | 各配置成功率均值、
  probe vs 最优 flat 的显著性、训练耗时、是否进 Phase 3)
- git 提交推送

## 红线

- 不修改主模型与既有评测代码逻辑;阈值/种子/评测数一律照 handshake 模板
- 与 Cube 检测层任务(PROMPT_4060_CUBE.md,Phase 4)互不依赖,分开跑、分开汇报
- 任一步卡住(训练发散/转换失败/评测报错),停下来在 PROGRESS.md 留言,不要猜
