# 代码改动记录（2026-11-05 ~ 2026-11-06）

## 核心策略：ProbeT' 双岗位检测器

这轮实验的核心方法是 **probe_t_hr**（[probe_t_policy.py](src/probe_t_policy.py)），在同一检测器头上挂了两个岗位：

| 岗位 | 作用 | 输入流 | 决定什么 |
|------|------|--------|---------|
| **规划前探路** | 自回归 rollout 10 块零动作，找首个 fire 步 s | 想象流（模型生成） | T' = max(s, 5) → CEM horizon 和 receding 都设为 T' |
| **执行中守卫** | 真实 z 流上实时跑检测器 | 真实流 | P > τ → 清空缓冲区，提前截断（r ≤ h 天然成立） |

检测器头：18ch CausalCNN，输入 f1(‖dz‖)/f2(cos)/f3_dual(创新量) × W=2/4/8 的 mean/std。f3_dual 用线性预测器 F_z（最小二乘闭式拟合）替代原始 WM 创新量，解决了自回归 rollout 上 f3=0 的失活问题。

### 其他策略臂（对比用）
- **flat(H,H)**：horizon=receding=H，固定人定参数（[run_taskA.py](src/run_taskA.py), [run_taskB.py](src/run_taskB.py)）
- **boundary**：固定 horizon + 事件触发重锚（[boundary_policy.py](src/boundary_policy.py)）

## 本轮修改清单

| 文件 | 类型 | 说明 |
|------|------|------|
| [e6_eval.py](src/e6_eval.py#L193) | 修改 | video=None → video=cfg.get('video_dir', None)，支持 eval 时可选写视频 |
| [probe_t_policy.py](src/probe_t_policy.py) | 已有 | 双岗位 detector policy，probe rollout 定 T' + real stream fire 截断 |
| [run_taskD.py](src/run_taskD.py) | 新增 | lane D 启动：probe_t_hr × d{25,50} × 3 seeds（自动适配 claim 的决定性考场） |
| [pushT_final_summary.py](src/pushT_final_summary.py) | 新增 | 四 lane 全量汇总：成功率矩阵 + 配对 McNemar + horizon 代价 Δ |
| [probe_tprime_stats.py](src/probe_tprime_stats.py) | 新增 | T' 分布统计（57.8% T=5 / 42.2% T=10，mean=7.11） |
| [judge_abc.py](src/judge_abc.py) | 新增 | A/B/C 三 lane 判读 + abc_verdict.json 生成 |
| [dl_cube.py](src/dl_cube.py) | 新增 | HF mirror 分段 Range 下载 LeWM cube 官方数据集（46GB tar.zst） |
| [build_official_cube_h5.py](src/build_official_cube_h5.py) | 新增 | 官方 cube tar.zst → h5 转换 |
| [check_official_cube.py](src/check_official_cube.py) | 新增 | 官方 cube 数据审计（起点目标距离分布） |
| [cube_init_goal_dist.py](src/cube_init_goal_dist.py) | 新增 | cube 初始目标距离分布统计 |
| [cube_lift_check.py](src/cube_lift_check.py) | 新增 | cube 物体抬起阈值检查 |
| [diagnose_panel_black.py](src/diagnose_panel_black.py) | 新增 | cube 面板黑帧诊断 |
| [scan_black_eps.py](src/scan_black_eps.py) | 新增 | cube 本地 h5 黑帧扫描 |
| [scan_black_eps_pusht.py](src/scan_black_eps_pusht.py) | 新增 | PushT 本地 h5 黑帧扫描 |

## 2026-11-06 追加（d25 flat(15,15) 补跑）

| 文件 | 类型 | 说明 |
|------|------|------|
| [e6_eval.py](src/e6_eval.py#L69) | 修改 | `receding_horizon × action_block > eval_budget` 由 assert 降级为 warning —— 允许 plan-once 臂（budget 内不重锚）合法执行 |
| [run_flat15_d25.py](src/run_flat15_d25.py) | 新增 | flat(15,15)·d25 × 3 seeds × n=150 补跑（断言放开后） |
| [peek_flat15.py](src/peek_flat15.py) | 新增 | 读 3 seed 结果并对比 d25 网格 |
| [gen_dashboard_data.py](src/gen_dashboard_data.py) | 新增 | 由 runs/*.json 全量重生成 taskA/B/C/D_data.js（dashboard 数据源） |
| [audit_official_cube.py](src/audit_official_cube.py) / [audit_cube_flat.py](src/audit_cube_flat.py) | 新增 | 官方 HF cube h5 解压 + 结构/黑帧/成功率/目标距离审计 |

## .gitignore 更新

新增忽略规则（结果文件不上传，可由 runs/*.json 重算）：
- `aiim2026/runs/*.json` — 所有实验结果 JSON
- `aiim2026/runs/videos_*/` — 视频目录
- `*.js`, `*dashboard.html` — 数据面板
- `aiim2026/models/*.pt`, `aiim2026/checkpoints/` — 模型权重
- `*.h5`, `*.zst`, `aiim2026/datasets/` — 数据集

## 关键数据（写文章用）

所有原始结果 JSON 在 4060 本地 `aiim2026/runs/` 目录，汇总后的判读在 `aiim2026/PROGRESS.md`（L64 起）。另一台电脑需要结果时 pull 本 repo + 从 4060 拷贝 runs/*.json。
