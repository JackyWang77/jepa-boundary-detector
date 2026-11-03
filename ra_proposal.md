# RA Interview Proposal — A Ruler for Latent Space

**Characteristic time scales as the missing metric for hierarchical planning in world models**
（面试提案：给潜空间一把尺子——用特征时间尺度度量信息粗细度）

> 用法：§1–§7 是可以打印两页带进面试的正式版；§8 是 90 秒口头开场白；§9 是教授大概率会问的六个问题和建议回答。方括号 [ ] 处需要你按目标实验室填空。

---

## 1. The problem (三十秒版本的问题陈述)

**EN.** Long-horizon robot manipulation requires hierarchical planning: a coarse plan over *what should happen* and fine control over *how to make it happen*. Modern world models (JEPA family, LeWM, diffusion planners) learn latents that predict well, but the latent space carries no notion of **information granularity** — no measurable answer to "this feature is relevant for the next 2 seconds, that one for the next 20". The consequence is measurable: when hierarchy is bolted on by treating the upper level's predicted latents as subgoals, policy success drops to **38.7%** vs 73.3% with oracle subgoals (the central negative result of the hierarchical JEPA line). The missing piece is not a bigger model — it is a **metric**: a physically grounded measure of how coarse each piece of latent information is.

**中文要点.** 长时程操作必须分层规划，但现有世界模型的潜空间没有"信息粗细度"这个可测量的概念——上一层的预测到底"管多远"无从知道。把上层预测的 latent 直接当下层 subgoal，成功率 38.7% vs oracle 73.3%，这是文献里已有的负结果。我的研究主张：缺的不是更大的模型，是一把尺子。

## 2. The core idea (核心想法)

**EN.** I define information granularity as a **characteristic time** τ: the duration over which a latent dimension stays true. Concretely, each latent coordinate z_i carries a τ_i, and a hierarchy boundary belongs at the largest measured spectral gap of the τ spectrum. I built two independent estimators:
- **Estimator A (data side):** per-dimension autocorrelation integrals with initial-positive-sequence truncation — no model, no labels needed;
- **Estimator B (model side):** eigenvalues of the learned one-step dynamics' Jacobian, λ → τ = −1/ln|λ| — no data statistics needed.

Agreement of the two is evidence the time-scale structure is real, not an artifact of either side. This is the physics I was trained on: τ is the dynamic correlation time, the same object that separates slow and fast modes in statistical mechanics, and the same intuition that separates macro from micro under the renormalization group.

**中文要点.** 我把"粗细度"定义为**特征时间** τ：一条信息能管多久。两个独立估计量——数据侧自相关积分（不用模型）、模型侧动力学 Jacobian 本征谱（不用数据统计）——互为对照。这就是统计力学里的动力学关联时间，也是重整化群里区分宏观/微观的那个东西。

## 3. What I have already built and measured (已完成的工作——面试的重头)

**EN.** This is not a proposal from nothing — the measurement phase is complete, with results from five self-trained regimes plus an external validation on a published third-party model.

**3.1 The representation: E²-JEPA.** I train JEPA-style world models with a **variational free-energy objective F = U − T·S** in place of the usual energy-only loss. The entropy term (logdet covariance) is not an engineering patch: it prevents variance collapse, covariance redundancy, and arbitrary-direction collapse simultaneously — dominating VICReg's separate variance and covariance terms. A temperature schedule (high-T buys entropy, low-T refines) plays the role of a thermodynamic cooling path.

**3.2 The key empirical law: self-organized latents have no spectral gap.** Measured across all five regimes — energy-only (collapse control), JEPA, E²-JEPA, behavior cloning, flow-matching BC — and externally on the official **LeWM PushT world model** (Amsterdam; used for real MPC), the largest multiplicative gap of the τ spectrum never exceeds **×1.03**, i.e. the spectrum is continuous everywhere. There is no place where a hierarchy boundary can be cut. This upgrades "hierarchy is hard" from anecdote to a cross-lab measurement.

**3.3 Why: two complementary dead ends.** The official LeWM model gave me the mechanism, in a beautifully clean form:
- *Compressed regime* (LeWM: participation ratio **6.2** ≈ PushT's intrinsic state dimension): the data manifold is ~6-dimensional, so **any half of any basis spans the entire physical state** — probe decoding is necessarily flat (block pose: 0.61 vs 0.60 across the slow/fast split). A sufficient statistic has no room to organize.
- *Overcomplete regime* (my E²-JEPA: PR 114): redundancy exists, but dense mixing (LayerNorm) homogenizes per-dimension τ to 9±1.

**3.4 The dynamics know; the coordinates don't.** Model-side, LeWM's Jacobian spectrum contains exactly **7 neutral modes** (|λ|≥1 at all 200 contexts) — equal to the physical state dimension (agent 2 + block 3 + velocities 2): the model learned a unit-gain pass-through of the physical manifold, integrating everything else within one step. My models show the same dissociation: model-side τ spans 0.1–20, coordinate-side τ spans 9±1. **The time-scale structure exists in the learned dynamics but is not inherited by the coordinate system.**

**3.5 Two bonus findings with full external validation.** (a) **Information dissociation by training objective:** prediction-trained encoders keep slow variables (block pose probe R² 0.77) and discard fast ones (agent 0.15); task-trained policies do the reverse — information in a representation is what the objective pays for, and one regime's information does not transfer. LeWM, whose MPC objective needs everything, keeps both (agent 0.69). (b) **A cautionary counterexample:** the collapsed control measures *pseudo-slow* τ = 15.6 — a constant signal has infinite autocorrelation and zero information. **τ is only defined where entropy is positive: the free-energy form is the existence precondition of the metric.**

**中文要点.** 三条硬结果：① 谱无隙是普适测量结果——五个自己的训练方案加官方 LeWM 全部如此，"层级难"从轶事升级为跨实验室规律；② LeWM 给出了机制——压缩码（PR≈内禀维）没空间组织，过完备码被稠密混合抹平，两头堵死，所以组织必须由目标函数强制；③ 动力学知道时间尺度（7 个中性模态=物理状态维数）但坐标不继承。另有信息解离、塌缩对照反例两个附带发现——第二个反例在数学上重要：τ 只在 S>0 时有定义，说明熵项是度量的存在前提，不是 trick。

## 4. What I propose to do next (提案的核心——唯一缺的正结果)

**EN.** If self-organization cannot produce the gap, enforce it. I extend the scalar temperature of F = U − T·S to a **temperature vector**: partition the latent into K bands, each with its own entropy term S_k and temperature T_k, giving **F = U − Σ_k T_k·S_k** — the K=1 case is exactly E²-JEPA, so this is a strict generalization of my existing objective. A temperature gradient across bands (slow bands hot, fast bands cold) is the external field that lifts the degeneracy, in the same way a field selects a phase in statistical mechanics. The experiment runs in half a day on one GPU, and I have **pre-registered the acceptance criteria before running it**: inter-band τ gap ≥ 1.5 (vs ≤1.03 everywhere currently); slow band decodes block pose at least 0.2 better than fast band; fast band decodes agent position. If it passes, the layer boundary becomes a measured object and criterion-based subgoal filtering becomes testable; if it fails, the fallback is a spectral-gap regularizer on the Jacobian eigenvalues directly (enforcing in dynamics space, where §3.4 shows the structure already lives).

**中文要点.** 自发不行就强制：把标量温度推广成温度矢量——F = U − Σ T_k·S_k，E²-JEPA 是它 K=1 的特例。跨尺度温度梯度就是外场。验收判据我已经预注册好（带间隙 ≥1.5 等），过/不过当天出结论，过了就能落地层级边界判据，不过就退到动力学侧加谱隙正则。

## 5. Positioning (和现有工作的关系——教授必问)

**EN.**
- *Slow Feature Analysis / temporal contrastive learning:* these objectives push all dimensions slow — they assume the answer rather than measure it; my results explain why slowness-only objectives homogenize rather than organize.
- *H-JEPA / Hi-LeWM (hierarchy by architecture):* they fix the hierarchy by design and inherit the 38.7% subgoal-interface problem; I provide the metric that says where boundaries belong and the training mechanism that makes latents obey it.
- *SIGReg / VICReg (anti-collapse):* E²-JEPA's logdet entropy subsumes their variance and covariance terms with one principled quantity; more importantly, none of them treats entropy as conjugate to temperature, and none connects collapse-prevention to the existence of a time-scale metric.
- *The measurement itself:* the two-estimator protocol (data-side ACF + dynamics-side Jacobian spectrum, cross-validated on an official third-party model) is, to my knowledge, the first time-scale audit of a deployed world model.

**中文要点.** 一句话定位：SFA 假设答案而不测量答案（我的结果还解释了为什么纯慢目标只会同质化）；H-JEPA 系按设计固定层级所以继承了 38.7% 的接口问题，我给的是"边界切在哪"的度量加上让 latent 服从它的训练机制；SIGReg/VICReg 只管防塌缩，E²-JEPA 的熵项一项顶它们两项，而且熵在这里是温度的共轭量、是度量存在的前提。

## 6. Why me (我能带来什么)

**EN.** A physicist's training turned loose on embodied AI: I think in free energies, conjugate variables, correlation times and symmetry breaking, and I hold results to the standard of falsifiable, pre-registered criteria. Full-stack execution on this project was mine end to end: JEPA training pipelines (PyTorch, MPS/CUDA), teleop data collection (pymunk/Gymnasium), the τ measurement protocol (torch.func vmap/jacrev, eigen-analysis), streaming analysis of a 46 GB official dataset, and external validation against a published third-party checkpoint — including diagnosing a BatchNorm statistics quirk in their released model that my measurement had to be made robust against. [补充一句你此前的教育/研究背景，一两行即可]

**中文要点.** 卖点是物理训练 + 全栈工程：整个项目从训练管线、遥操作数据采集、τ 协议、46 GB 官方数据流式分析到第三方模型外部验证都是我独立完成的——包括顺手诊断出他们发布模型里 BatchNorm 统计量的怪癖并让测量对它稳健。研究品味上：判据先注册后实验，负结果也讲机制。

## 7. Proposed timeline (六个月路线图)

**EN.**
- **Month 1:** temperature-vector arm v1 (fixed band assignment) → run pre-registered evaluation; either way, a clean result.
- **Month 2:** v2 with dynamic reassignment of bands by measured τ rank (self-organizing spectral order); if the gap appears, backtest subgoal filtering on top of the boundary (directly targeting the 38.7% negative result).
- **Month 3–4:** transfer the protocol to [目标实验室的数据/机器人平台/LeRobot 基准] — the estimators are task-agnostic; write the first paper.
- **Month 5–6:** hierarchical controller demo: plan block trajectories in the slow subspace, execute agent tracking in the fast subspace; submission.

**中文要点.** 一个月出机制实验结果，两个月做判据二回测，三到四个月把协议迁到实验室平台上并写第一篇论文，五到六个月层级 demo。

---

## 8. 90-second spoken opening（口头开场白）

> 我做的是给分层规划补一个缺失的度量。长时程机器人任务需要分层规划——上层规划"发生什么"，下层控制"怎么做"。但现在的世界模型潜空间里，没有任何量能回答"这条信息能管多久"。文献里有个很典型的负结果：把上层预测的 latent 直接当下层的目标，策略成功率只有 38.7%，oracle 目标是 73.3%——差的不是模型，是度量。
>
> 我把信息粗细度定义成特征时间 τ，用两个互相独立的估计量去测：数据侧自相关积分，模型侧动力学 Jacobian 本征谱。测完之后发现一个普适规律：**自发训练的表征，τ 谱没有隙**——我自己训了五种方案，还解剖了阿姆斯特丹发布的、真正用于 MPC 的官方世界模型，全部无隙。而且 LeWM 把机制说清楚了：压缩表征没有组织空间，过完备表征被稠密混合抹平——两头堵死。
>
> 有趣的是，他们的模型动力学里其实有结构：Jacobian 谱里恰好 7 个中性模态，等于 PushT 物理状态的维数——模型学对了物理，但坐标系不继承。
>
> 所以我的下一步，也是这个 proposal 的核心：把自由能里的标量温度推广成温度矢量，用跨尺度温度梯度强制谱分离——判据我已经预注册，实验半天出结果。过了，层级边界就从"拍脑袋设计"变成"测出来的对象"。
>
> 这套东西的物理底子是统计力学：熵是温度的共轭量，τ 是关联时间，温度梯度是外场。我受的是物理训练，这是我最想把物理直觉变成机器人学里可用工具的工作。

**English one-liner if the interview is in English:** "I build world models with a free-energy objective, then audit them with a characteristic-time measurement — the audit shows self-organized latents never separate time scales, and the fix is a temperature vector that enforces the hierarchy. The ruler exists; now I'm making the latents obey it."

---

## 9. Anticipated questions (预演问答)

**Q1. "这不就是 Slow Feature Analysis 吗？"**
A: SFA 是目标函数，我的是测量协议——SFA 假设"越慢越好"，我的实测恰好解释了为什么这类目标不行：把所有维度都往慢推，结果是同质化而不是组织（我们对能量_only 对照测到伪慢 τ=15.6，但信息为零）。SFA 是"假设答案"，我是"测量答案 + 强制结构"。

**Q2. "为什么不直接用多尺度结构（金字塔/层级网络）？尺度是设计者定的。"**
A: 对——这就是我反对的。固定尺度假设了答案；而我的模型侧测量显示，学到的动力学里时间尺度结构是存在的（LeWM 7 个中性模态=物理维数），问题是坐标系不继承。所以正确路线是让边界从测量中出来，而不是从架构里出来。论文里这叫"边界是测出来的，不是调出来的"。

**Q3. "温度矢量要是造不出隙怎么办？"**
A: 判据已预注册，当天出结论，不会沉没。Fallback 是在 Jacobian 本征谱上直接加谱隙正则——因为 A/B 解离证明结构在动力学侧是存在的，坐标系不继承，那就在动力学侧强制。即使两条路都失败，测量阶段本身（τ 协议 + 跨实验室谱无隙 + 两头堵死机制 + 官方模型解剖）已经是完整可发表的发现型结果。

**Q4. "PushT 太简单了吧？内禀维才 5-7。"**
A: 故意的。PushT 是操作任务里的"箱中粒子"——内禀维已知、物理完全可控，所以能对测量做真值验证（7 个中性模态恰好对上物理维数，就是协议正确性的证据）。两个估计量都是任务无关的，下一步就迁到 [实验室的平台]。测量学的工作惯例就是先在真值已知的系统上校准仪器。

**Q5. "和 H-JEPA 的层级方案本质区别是什么？"**
A: H-JEPA 把层级做进架构，层间接口用"上层预测直接当下层目标"——这正是 38.7% 那个负结果的来源。我的路线是反的：先给潜空间一个可测的粗细度量，让"哪里切层、什么配当目标"变成可计算的判据，再用温度矢量让表征服从这个判据。架构层级是硬件，度量是语法——现在缺的是语法。

**Q6. "你一个人做的？规模有多大？"**
A: 全流程独立：训练管线、遥操作数据采集、τ 协议、46 GB 官方数据流式分析、第三方 checkpoint 外部验证。规模是刻意控制的——测量学的纪律是每个环节都能被真值校准，所以没有追求大模型，追求的是每个数字都有机制解释。[如有导师/合作者，在此如实补充]

---

## 10. 附：一页数字速查（面试前扫一眼）

| 数字 | 含义 |
|---|---|
| 38.7% vs 73.3% | 上层预测直接当 subgoal 的失败（文献负结果，动机） |
| ×1.03 | 五种方案 + 官方 LeWM 的最大 τ 谱隙上限（谱无隙普适） |
| PR 6.2 ≈ 7 个中性模态 | LeWM 压缩到物理内禀维；动力学直通物理流形 |
| block 0.77 / agent 0.15 | 预测训练的表征保留慢变量、丢快变量（信息解离） |
| 伪慢 τ 15.6, PR 4.6 | 塌缩对照的反例：τ 只在 S>0 时有定义 |
| ≥1.5 / +0.2 / ≥0.3 | 温度矢量实验的预注册验收判据（带间 gap / block 半谱差 / fast 带 agent） |
| 20 分钟 | 一次完整 τ 测量耗时（E²-JEPA 训练 20k 步约半小时） |
