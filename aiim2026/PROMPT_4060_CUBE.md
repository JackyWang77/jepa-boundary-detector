# Prompt for the 4060 agent: Cube detector-layer evidence (reviewer R1)

> 复制以下内容发给 4060 上的 agent 执行。mac 侧不改任何评测代码,只消费结果。

## 背景

AIIM 2026 审稿意见 R1 要求:"补充不同任务类型下各类信号的分布可视化以及误触发率分析"。
mac 侧已完成 PushT 版本的分布图与 τ-sweep(见 `aiim2026/results/r1_review.json` 的
`part_b_pushT_holdout` / `part_d_tau_sweep_pushT`,脚本 `aiim2026/src/r1_review_analysis.py`)。
现在需要 **Cube 任务**的对应证据。注意:**这项分析与 Cube 上 probe 是否打赢 flat 无关**,
检测层证据独立成立;不重训任何东西、不跑任何评测,纯读取 + 推理,CPU 即可,分钟级。

## 本机已有资产(4060)

- `aiim2026/features_cube/`:ep*.npz(每集 18 通道特征由 cube_extract_features.py 生成)
  与 meta.json;
- cube head 权重与归一化:使用产出 `aiim2026/results/cube_r2_dual.json` 里
  `fire_micro_f1` 最优那一档对应的 head(见该 json 的 sweep 记录里 head 类型/pos_weight,
  权重文件以你本机 `aiim2026/models/` 下实际文件为准);
- cube 事件标签逻辑:沿用 `aiim2026/src/cube_train_head.py`(或 cube_events_check.py)
  里已有的 kinematic event 定义,不要新造。

## 任务:写并运行 `aiim2026/src/r1_cube_hist_tau.py`

脚本逻辑严格对照 mac 侧 `aiim2026/src/r1_review_analysis.py` 的 Part A 与 Part D:

1. **信号分布图**(Part A 对应):对 features_cube 全部 episode,画三信号
   f1 energy / f2 cosine / f3 novelty 在 kinematic event 步 vs non-event 步的
   密度直方图(step hist,bins=60,density=True,event 色 `#0072BD`、
   non-event 色 `#8C8C8C`,lw=1.4,面板标题 f1 energy / f2 cosine / f3 novelty,
   横轴无需 y 刻度,首面板 density + legend,figsize=(7.0, 2.4),tight_layout)
   → 存 `aiim2026/results/fig3_cube_dist.pdf`。
   注意 f1/f2/f3_dual 与 18 通道窗口统计的构造必须与 cube_train_head.py 训练时一致。
2. **τ-sweep**(Part D 对应):Cube hold-out 集上,部署头阈值
   τ ∈ {0.30, 0.35, ..., 0.70}(步长 0.05),refractory=5、match tol=2 不变,
   每档输出 precision / recall / F1 / fires_per_episode / false_fires_per_episode
   → 存 `aiim2026/results/r1_cube_hist_tau.json`。

## 输出与完成标准

- `aiim2026/results/fig3_cube_dist.pdf`:与 mac 版 `fig3_signal_dist.pdf` 同风格,
  mac 可直接拼成两行图。
- `aiim2026/results/r1_cube_hist_tau.json`:JSON 结构逐字段对齐 mac 侧
  `aiim2026/results/r1_review.json` 的 `part_d_tau_sweep_pushT` 条目
  (tau/tp/fp/fn/precision/recall/f1/fires_per_episode/false_fires_per_episode),
  额外加一个顶层字段说明 hold_episodes 数量。
- `aiim2026/PROGRESS.md` 追加一条留言板记录(日期 | [4060 agent] **cube 检测层证据** |
  两文件路径 | 关键数字一句话)。
- git 提交并推送这两个文件(JSON 必须 push;PDF 若超过 git 限制可不 push,
  留言板里说明,由用户手工传 mac)。

## 红线

- 不重训、不跑评测、不修改任何世界模型权重;
- 阈值/容差/事件标签定义与 mac 侧完全一致,不要"顺手改进";
- 若 cube head 权重文件缺失或对不上 cube_r2_dual.json,停下来在 PROGRESS.md 留言,不要猜。
