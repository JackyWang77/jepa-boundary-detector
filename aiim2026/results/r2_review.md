# R2 检测器救援 外部审查报告（2026-11-03，只读审查）

## 1. Summary

**可信度结论：R2 的核心数字可信。** 标签构造、18 通道特征、节律基线、门判定逻辑全部通过独立复算（复现精度：标签统计精确一致、特征 1.6e-14、节律 F1 逐相位一致）。判决 PARTIAL 执行正确且对选择规则稳健。

**发现的问题**：1 个 P1（P1-7：选中的 CNN 头未持久化，权重已丢失，E6 boundary 臂阻塞）+ 2 个 P1 级风险记录（hold 集三重用途的选择偏置未量化上限；fire F1 0.657 因头未存无法独立复算）。无 P0 违规。

**放行建议**：R2 数字可进论文（附 hold 复用限制说明或补 fresh-test）；E6 的 flat-short/decoupled 臂**立即放行**；boundary 臂**条件放行**——先完成 P1-7 最小修复（重训+持久化+部署对齐）。

## 2. 逐项审查结果

### P0-1 标签正确性 — **PASS**

证据：抽 ep3846/ep2600/ep16620 独立重算（raw state → kinematic_events → ceil(t/5) → 去重 → ev<K 过滤），事件数与正例行数逐条一致；raw_len 与 h5 ep_len 相等（无读取差异）。全局复算：train rows=8441, positives=2030 (24.05%), data pos_weight=3.158——与 r2_rescue.json 的 0.24049 / 3.158 **精确一致**。
复现：`/tmp/r2_review_check.py` 第一段。

### P0-2 18 通道正确性 — **PASS**

证据：用独立实现的因果滑窗（[max(0,k−w+1)..k]、NaN-aware、单值 std=0）对拍 `e2_extract_features.causal_stats`，ep3846 上 {f1,f2,f3}×{mean,std}×W∈{2,4,8} 共 18 通道 **max diff = 1.6e-14**（机器精度）。行从 step 3 起，X 形状 (K−3, 18) 与标签 y 对齐（代码 61-62 行）。
复现：`/tmp/r2_review_check.py` 第二段。

### P0-3 泄漏与选择偏置 — **PASS with 风险记录**

已确认合规：z-score 统计仅来自 train 拼接（r2_rescue.py:194-198），hold 用同一 mu/sd；train/hold 划分直接取自 e4_meta.json（episode 级互斥，无泄漏）。

**风险（如实记录，不改变门判定）**：hold 100 条承担三重角色——每 epoch early stop、6 配置 sweep 选优、最终判决。量化：6 配置 hold AUC 跨度仅 0.798→0.813（0.015），在 100 episode（~2246 行）上取 max 的乐观偏置量级估计 <0.01；即使扣 0.02，0.79 仍远超 0.70 门，**门判定不变**。但 json 未记录 train-AUC，无法做同位性检查。**最小修复建议**：从 500 条池之外（18685−500）抽 fresh 100 条做一次最终 test（只需前向，分钟级），论文报告 fresh-test AUC。

### P0-4 门判定执行 — **PASS**

证据：`AUC_GATE=0.70`、`F1_GATE=0.745` 硬编码（= 预注册 0.715+0.03），无事后放宽；实测 rhythm_max=0.726 **如实记录**于 json（若按实测口径收紧到 0.756，F1 0.657/0.6795 更过不了——即现有 0.745 门反而略宽松，但结论 PARTIAL 在任何门下不变）。选择规则稳健性：selected 按 hold AUC 选 CNN（0.813），但即使按 fire F1 选 GRU pw=3.16（0.6795，sweep 中最高 F1），仍 <0.745，**判决对选择规则不敏感**。判决分支执行与预注册文字一致（AUC 过 + F1 未过 → PARTIAL）。

### P0-5 口径一致性 — **PASS，差异已解释**

证据：R2 节律基线 [0.685, 0.710, 0.726, 0.708, 0.690] vs e5_alignment_v2 [0.674, 0.697, 0.715, 0.699, 0.680]，两者我均独立复现（逐相位一致）。差异根源确认：**尾部 27 个事件（ev≥K，E5 口径未过滤）**——E5 合计 574 事件，R2 过滤后 547。R2 口径更合理（K=T//FS 截断后尾部事件不存在对应特征行，不可能被任何 fire 命中，保留它们只会虚增 FN）。两个 json 各自如实记录各自口径，无混用。**判决门本意对应 R2 口径**（检测器与基线同口径比较），0.745 门高于两口径的 max，无 cherry-pick。

### P0-6 fire F1 匹配代码 — **PASS**

证据：`r2_rescue.fire_f1` 与 `e5_alignment.f1_event` 同构（±2 容差、贪心一对一），fire 位置 `+3` 正确映射回 model step。事件数 547 = TP 287 + FN 260 ✓；574−547=27 与 P0-5 尾部过滤一致。
**限制**：fire F1 0.657 依赖 CNN 头前向，头未持久化（见 P1-7），我无法独立复算该数字；匹配代码本身正确，数字可信度中等。

### P1-7 头持久化 — **FAIL（gap 确认，阻塞 boundary 臂）**

证据：`r2_rescue.py` 全文无 `torch.save`；`models/` 下仅有 head_gru.pt（旧 6 通道头）与 head_cnn.pt（E4 旧 CNN，6 通道），**R2 选中的 18 通道 CNN( pw=3.16) 权重未保存，已随进程退出丢失**。E6 boundary 臂无法部署该头。
最小修复（建议，未执行）：重跑 r2_rescue.py 的 CNN 分支（SEED 固定，理论可复现；建议顺带 AUC/F1 重报一次以闭合 P0-6 的复算缺口），落盘：
1. `models/head_r2_cnn.pt`（state_dict）
2. `models/r2_meta.json`：**通道顺序（f1/f2/f3 × mean/std × W=2/4/8 的确切排列）、z-score 的 mu/sd（18 维，train 统计）、TAU/REFRACT、train/hold id**
3. boundary_policy.py 特征管线同步 18 通道 + 新 mu/sd，**重跑 train-serving 对齐回归测试**（复用 test_deploy_align 口径，max diff 须 <1e-5）——这是 E6 boundary 臂放行的硬门。

### P1-8 文件不覆盖合规 — **PASS**

证据：`results/` mtime——e5_alignment.json（11-02 15:09）、e5_diagnosis.md（11-02 16:31）均未被改动；新文件 e5_alignment_v2.json（11-03 13:22）、r2_rescue.json（11-03 13:35）。`ls -la` 留档。

### P1-9 PROGRESS 一致性 — **PASS**

证据：11-03 两行（REVISION v2 采纳 / R2 检测器精修）的关键数字（0.674-0.715、0.524/0.445、24.0%、0.798-0.804、0.813、0.657、2769、PARTIAL）与 e5_alignment_v2.json、r2_rescue.json 逐项一致。指令与实现的一处出入：审查指令称"sweep 7 项"，实际 6 项（CNN 的 pw 集合 `sorted({3.16, best_gru_pw=3.16})` 去重后仅 3.16 一档）——实现细节，非违规，如实记录。

### P1-10 参数预算 — **PASS**

CNN 2769 参数 ∈ [2k, 5k]（指导书 §E4 预算）。

## 3. 发现的问题（分级汇总）

| 级 | 问题 | 最小修复 |
|---|---|---|
| P1-7 | R2 CNN 头未持久化，权重丢失；fire F1 无法独立复算 | 重跑 CNN 分支 + torch.save + r2_meta.json（通道序/mu/sd/超参）+ boundary_policy 18 通道同步 + 对齐回归硬门 |
| P1 | hold 三重用途，AUC 0.813 含 <0.01 乐观偏置，无 train-AUC 记录 | fresh-test：500 池外抽 100 条前向评估（分钟级），论文报 fresh 数字或明示限制 |
| 记录 | 指令"sweep 7 项"vs 实际 6 项（集合去重） | 无需动作，报告留痕 |

## 4. 放行判定

1. **R2 数字可否进论文**：**可以**（附条件）。标签/特征/节律基线/判决全部独立复现；AUC 0.813 建议以 fresh-test 数字为准或明示 hold 复用限制；fire F1 0.657 待 P1-7 修复后重报一次再定稿。
2. **E6 boundary 臂（部分信号版）可否放行**：**条件放行**——完成 P1-7 三件套（持久化、meta、对齐回归）后放行。判决 PARTIAL 本身合规：AUC 门过（检测信号真实存在）+ F1 门未过（fire 策略未超节律基线）→ "部分信号"措辞与 REVISION §3.4 一致。
3. **P1-7 是否阻塞 E6**：**仅阻塞 boundary 臂**。flat-short（复用 E1）与 decoupled 臂不依赖 R2 头，**可立即开跑**——建议并行：decoupled 先行，P1-7 修复后 boundary 加入。

## 5. 复现命令清单

```
# 全部复算（标签对拍 + 18 通道对拍 + 双口径节律基线）
/Users/wanglanlan/miniforge3/envs/aiim/bin/python3 /tmp/r2_review_check.py
# 关键对照值：
#   train rows=8441 pos=2030 (24.05%) pw=3.158  == r2_rescue.json
#   18ch max diff = 1.6e-14（vs e2 causal_stats）
#   E5口径节律 [0.6736,0.6972,0.7147,0.6986,0.6796] == e5_alignment_v2.json
#   R2口径节律 [0.6854,0.7098,0.7260,0.7077,0.6905] == r2_rescue.json
#   hold 事件 E5=574 / R2=547 / 尾部=27
ls -la aiim2026/results/   # mtime 合规证据
tail -3 aiim2026/PROGRESS.md  # 进度行一致性
```
