# 进度日志

已读指导书 v1（2026-11-02）

2026-11-02 | E1 平坦基线 | 9/9 任务完成（d25: 87.3±4.1 / d50: 46.0±3.3 / d75: 16.0±1.6 vs 发表 94.0/52.7/18.0），d75、d50 在容差边缘，d25 差 -6.7 超出 ±5 容差 | harness 换用主线 scripts/plan/eval_wm.py（非 le-wm eval.py 修 solver 路径），单调衰减复现；d25 偏差待查（疑 history_len 口径）

2026-11-02 | E2 三路特征提取 | 500 条轨迹（seed=0，id 列表在 features/meta.json），12,187 模型步；f1 ‖Δz‖ p50=5.05、f2 cos p50=0.58、f3 innovation p50=0.98；innov/‖Δz‖ 中位比 0.19（predictor 有效） | 全部落盘 features/ep{id}.npz（z/三路raw/6通道滑窗统计/action/state），NaN 仅头部定义位，验收通过。关键口径：frameskip=5（模型步=5原始帧），action 全列 z-score（std+1e-2，训练口径），predict 吃 act_emb 非 raw action

2026-11-02 | E3 无监督切分 | PELT(normal, BIC pen=13.5·log n) 每条中位 3 个切点、共 1745；BinSeg/BottomUp 同惩罚下与 PELT 切点集 100% 一致（BIC 惩罚下三法收敛到同一最优，切分对搜索方法鲁棒）；BOCPD（hazard=1/中位段长）更保守 693 个 | 落盘 segments/e3/ep{id}.npz（四方法 cps + PELT 伪标签）。cost 选 normal 有原则依据：BIC 需要似然代价，rbf 在同惩罚下退化为 0 切点（smoke 实测）。**预览（非正式 E5）**：PELT 切点 79.5% 落在物理事件 ±2 模型步内，78.6% 事件被覆盖——label-free teacher 与物理边界高度对齐，E5 大概率走"高对齐"叙事

2026-11-02 | E4 student 蒸馏 | **GRU 胜出**：1169 参数，留出伪标签 F1=0.985（train 0.971，37 epochs early stop）；CNN 1809 参数 F1=0.966。二选一都试报好的，GRU 参数更少且 F1 更高——"无界记忆比 9 步感受野更有用"是 E5 消融的一个数据点 | models/head_{cnn,gru}.pt + e4_meta.json（含 train/hold episode id 划分，seed=0 80/20：400/100）。pos_weight 实测 5.2（指导书估 1:30，按数据修正并记录）。蒸馏有效性确认：1.2k 参数因果头复现 PELT 决策达 F1 0.985（τ=0.5 + 5 步不应期，±2 容差）

2026-11-02 | E5 对齐主表 | PELT macro F1=0.636（prec 82.6% / recall 53.7%），student 0.629（ΔF1=0.007 蒸馏近零损失），随机 0.547、等间隔 0.499（同预算口径）| 按预注册判据 F1<0.7 走"统计动力学分段"措辞。注意：recall 天花板 63%（PELT 3.73 切点/条 < 5.90 事件/条，78/100 条欠切分）；事件对（onset+settle）常共占一个切点 → results/e5_alignment.json

2026-11-02 | E6 代码审查交叉核对 | 外部 agent 审查报告（.trae/documents/E6代码审查与修复计划.md）五项问题与我独立发现并已修复的完全重合：P0-1 特征 index 错位（已修，train-serving 对齐回归 max diff 1.3e-6 通过）、P0-3 hidden device（已修，smoke2 v1 崩溃即此）、P1-2 fire 路径（修复版 smoke 已验证）、P0-2 equal interval（流程钉死：boundary 实测间隔后跑）、P1-1 落盘粒度（按审查结论维持） | boundary_policy.py 在线特征管线与 E2 逐值一致。另发现审查未覆盖项：头输出压缩（74.9% 行 p>0.5，p95=0.577），分布内 fire 率 3.55/条≈teacher 3.43，分布外（d75 失败 rollout）近饱和；τ 决策待用户定（0.5 vs 0.52）

2026-11-02 | E5 诊断（只读） | 五项：(1) FN 构成：push 漏 164（率45%）、settle 漏 109（率52%），settle 更难 (2) 对照 TP/FP/FN 一直在 json 里（呈现遗漏）；按事件配点的随机 micro F1=0.685>PELT——F1 绝对值被 ±2 容差抬高，真实信号=同预算 +0.09 (3) 78/100 条欠切分（3.73 vs 5.90/条），recall 天花板 63% (4) FP 是平淡处粒度噪声（f1=4.01 反而低），非几何事件 (5) FN 向尾部集中（settle 57% 在后 1/3）——弱事件（收尾精细动态）系统性不敏感 | 完整报告 results/e5_diagnosis.md，修正方向由用户定

2026-11-03 | REVISION v2 采纳 | 节律陷阱确诊：常数 5 步定时器对物理事件 micro F1 0.674-0.715（5 相位），完胜 PELT(0.650)/student(0.645)；旧 student 逐步 AUC 仅 0.524（±2 扩张口径 0.445，低于随机）——E4 蒸馏 F1 0.985 是 refractory 网格与 PELT 节律共振，label-free 路线降级为 negative finding | 方法切换 privileged distillation（训练用物理事件标签，部署纯潜变量）。E5 修订表落盘 results/e5_alignment_v2.json（节律基线 + AUC 行，原文件未动）

2026-11-03 | R2 检测器精修 | 精确事件标签（正例 24.0%，pw=3.16）+ 18 通道多尺度（W=2/4/8）+ hold AUC early stop。GRU 扫描 AUC 0.798-0.804；**CNN(18ch, pw=3.16) AUC=0.813**、fire F1=0.657、2769 参数——选中 | **判决：PARTIAL**——AUC 门通过（0.813≥0.70，检测信号真实），fire F1 门未过（0.657<0.745，未超节律基线 0.715+0.03）→ boundary 臂按"部分信号"诚实报告，decoupled 扛正结果。results/r2_rescue.json（sweep 全记录），原始 E4/E5 文件未覆盖

2026-11-03 | P1-7 修复 + 18ch 同步 + 对齐硬门 | R2 重跑逐位复现（AUC 0.813022…，火 F1 复算缺口闭合），头落盘 models/head_r2_cnn.pt + r2_meta.json（通道序/mu/sd/τ/refract/train-hold id）；boundary_policy 同步 18 通道 + CNN 前缀推理 + r2_meta 接线 | **train-serving 对齐回归 PASS**（test_deploy_align.py：特征行 max diff 4.29e-06、头概率 1.01e-06 < 1e-5 硬门，3 条轨迹）。decoupled smoke 机制通过：每 5 步自然重锚、CEM h=15 求解 20.7s/次稳定、0 崩溃（0/2 成功率无信息量，d75 flat 基线 16%）

2026-11-03 | R2 外部审查 | 标签/18通道/节律基线/门判定全部独立复现（正例率精确一致、特征 diff 1.6e-14、双口径节律逐相位一致、尾部 27 事件差异解释成立）；无 P0 违规；判决 PARTIAL 对选择规则稳健 | **放行：R2 数字可进论文（建议 fresh-test 或明示 hold 复用限制）；flat-short/decoupled 臂立即开跑；boundary 臂条件放行**——P1-7：CNN 头未持久化（权重已丢），需重训落盘 + r2_meta（通道序/mu/sd）+ boundary_policy 18 通道对齐回归后放行。results/r2_review.md

2026-11-03 | E6 smoke 阶段（boundary R2 头 + decoupled CEM 质量） | boundary R2 头在线 smoke ✓：fires 4-5/条、p 值 0.58-0.99（真实区分度，非旧头贴 0.5）、开火间隔不规则、fire→clear→重规划全通。**decoupled 红灯**：d25 上 flat h=5 对照 2/2 全成、decoupled h=15 同 2 条 0/2、600 样本仍 0/2 | 机制诊断（源码级）：GoalMSE 为纯终点代价，d25 目标在第 5 块 vs 终点第 15 块 → 规划器须优化"到目标后悬停 10 块"，穿过模型自身漂移区（论文主题现象反噬长程规划），终点代价噪声化；**d75 目标恰在第 15 块 = 与 h=15 终点完美对齐**——d25 失败是预期行为，不否定预注册主判据域（d75）

2026-11-03 | 主 agent 三疑点核查修正 + 日志 bug 修复 | 疑点 1/2 解除（decoupled 走 receding=5 自然重锚，标准 MPC 机制，6 次/30 blocks 吻合）；**真 bug 确认并已修**：非 boundary 臂跳过 _features 后 st['m'] 不推进 → replan_logs 的 block 字段全 -1，三处日志改用 self.steps[i]//FS（后续任务生效；在跑的 d75-seed42 决定性一战不受影响——success/逐episode/重锚计数完好，只丢 block 位置） | horizon=15 生效证据补齐：e6_eval 增 [plan] 一次性打印（horizon/receding/block/臂语义）；CEM 求解时间 h=5 6.2s vs h=15 20.7s（同 2 env，比值 3.4×≈rollout 长度比 3×）——论文实验设置节引用此两行。**Table II 控制变量确认干净：flat-short(5,5) vs decoupled(15,5) 唯一差异 = plan 长度**

2026-11-03 | **decoupled 判据战** | d75 seed42 全量 50 eps：**decoupled 8.0% vs flat-short 16.0%（配对同任务），判据 −8pp，FAIL**；d25 对照 flat 2/2 vs decoupled 0/2（600 样本仍 0/2） | 机制（用户直觉确认+源码级）：GoalMSE 纯终点代价，h=15 须 15 步开环 rollout，漂移逐步累积到终点 → 终点代价在失真预测上计算，CEM 优化信号噪声化；flat 开环只信 5 步、每 5 步真实观测重锚（闭环 15 开环 5）。**论文级发现：冻结 WM 的可用计划长度被其 rollout 漂移封顶**——Mind the Gap 的漂移现象在决策侧的后果。下一步三臂对照：decoupled(8%,无额外重锚) vs equal(定时盲重锚) vs boundary(事件触发重锚)——boundary vs equal = 检测器信息增益的关键对照

2026-11-03 | **第二任务侦察（用户提出跨任务适应性主张）** | HF quentinll 有 LeWM 全家四任务（pusht/cube/tworooms/reacher，model+dataset 齐全）。**选定 tworooms**：数据仅 3.4GB（vs cube 46GB/reacher 24GB）；语义=两房间导航穿门（1-3 门，burst 式 1-3 事件/10 块，与 PushT 准周期接触节律结构相反）= 定时器结构性失效地形；state 含门心坐标（事件提取比 PushT 更易）；协议与 PushT 完全同构（同 CEM/goal_mse/评测结构），管线机械迁移 | 下载坑：aria2 两次死在 cas-bridge CDN 收尾（文件全尺寸缺一块），resume 一次补齐；教训=大文件下载完成后必须 torch.load 验证再继续。checkpoint 已转换（18.0M strict 通过 → ~/.stable_worldmodel/checkpoints/tworoom/）。数据集 3.4GB 后台下载中；PushT boundary d75 seed42（R2 头）并行在跑

2026-11-03 | E6 d75 decoupled smoke 判据通过 + 全量启动 | smoke（2 ep，seed42）三项全过：进程正常退出 JSON 落盘；两 episode 重锚 block 均 0/5/10/15/20/25 全 natural（replan 日志修复生效，无 -1）；[plan] 行 horizon=15 receding=5 action_block=5 arm=decoupled（同配置 `python -u` 短验证进程捕获直接证据——主日志 stdout 全缓冲致 [plan] 延迟至进程退出才 flush，属预期）。成功率 0/2 无统计意义（d75 flat 基线 16%），不据此决策 | 全量 50 ep 后台运行：PID 12057，日志 runs/e6_decoupled_d75_seed42.log，结果 runs/e6_decoupled_d75_seed42.json；进程经 setsid 守护化（PPID=1，脱离终端会话——普通 nohup& 在本环境连续 4 次被清理后改用此法）；按 smoke 实测 ~5.3 min/ep 外推预计 ~4.4h 完成（约 23:15），完成标志 = JSON 落盘
2026-11-03 | E6 decoupled d75 全量（seed42, 50eps, PID 12057, 18:51-19:52 实际约 1h） | **4/50 = 8.0%**，重锚 294 次全部 natural（6/episode 节奏正确） | **预注册判据失败**：需 >23%（16%+7pp），实测 8% 低于 flat-short 基线 8pp——h=15 长视野在该任务上为负效应；机制上与论文主论点自洽：终点代价穿过 10 块模型漂移区，前 5 块动作的优化信号被尾部 rollout 不可靠性污染。decoupled 不再是正结果载体

2026-11-03 | E6 d75 三任务后台启动（两链并行，22:19） | 链A boundary seed42：PID 15373（PPID=1，fork+setsid 守护化），horizon=15 + R2 CNN 头经 r2_meta.json 自动接线（无 head_path 参数），日志 runs/e6_boundary_d75_seed42.log，结果 runs/e6_boundary_d75_seed42.json，预计 ~1h（约 23:20）完成 | 链B decoupled seed43→44 同链顺序执行（; 连接，避免 3 进程抢 mps）：zsh PID 15385（PPID=1），seed43 python PID 15386 在跑、seed44 排队，日志/结果 runs/e6_decoupled_d75_seed{43,44}.{log,json}，预计 ~2h（约 00:20）完成；启动 75s 后 ps 确认两链存活、日志加载至 ViT-tiny 无报错，完成标志 = JSON 落盘

2026-11-04 | E6 decoupled d75 seed43 落盘 + 系列取消决议（用户） | seed43：**3/50 = 6.0%**（01:52 落盘，与 seed42 8.0% 两 seed 均值 7.0% vs flat 16.0%——负结果坐实） | **decoupled 后续取消**：seed44 链（PID 15385）按用户决议 kill，结案只用 seed42+seed43；仅保留 boundary d75 seed42（PID 15373，02:11 实测 Rs 在跑、CPU 18.5min 接近尾声），落盘即判读 vs flat-short 16.0%。**第二任务策略变更**：弃 tworooms，改 **OGB cube**（机器人控制方向）；mac 专注 PushT 收尾，cube 迁 4060/CUDA 机器跑——本地 oracle 单视角采集（官方 46GB 排除：mac 磁盘仅剩 56GB + aria2/CBD 前科）+ lewm-cube 既有 checkpoint 底座（不自训 WM），d/budget 网格与全部冻结清单不变；cube 数据三用途：E1/E6 评测任务集、E2 特征提取、R2 头蒸馏。执行计划：.trae/documents/OGBcube迁移与E6判读执行计划-20261104.md
