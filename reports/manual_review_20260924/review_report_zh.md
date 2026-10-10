# TruthInsightBench：27题产物人工审阅评分

**审阅日期：2026年9月24日。状态：人工审阅暂评分，不是官方Evaluator执行结果。**

本次对27题、128条注册发现作了评分，按人工近重复合并后有125条独立计分发现，其中121条达到40分的数量奖励资格。27题等权均分为 **66.3/100**，中位数 **68.2**。这批产物来源混合，不应把总体均分解释为某一个模型的成绩。

## 1. 评分口径

每条发现按29个动作分别给0、0.5或1档，乘以权重并求和。六大类满分依次为：证据45、稳定性15、替代解释15、分离验证10、独立来源与新颖性10、下一步测试5。

每题先对近重复发现取组内最高分，再计算：

`总分 = clip(0.6 × S1 + 0.2 × S2 + Y + 5 × (A+B) − P, 0, 100)`

S1、S2是去重后的最高与次高发现分。按最终闭环脚本的实际实现，组分数≥40即有数量奖励资格，前四组分别奖励4、2、2、2分，Y最高10。两个目标A、B分别给0、0.5、1档。P针对仍保留且直接被反驳的不同错误，每个5分，上限10。一般证据不足不自动触发P。

目标锚点不是穷尽的正确答案。有效的新发现、方法审计、加工伪影和负结果均可取得发现分，即使没有目标恢复奖励。一个目标的完整命中必须由同一条发现覆盖，不能跨发现拼接局部量。

### 人工审阅与原执行流程的差异

人工核对Result.md、冻结锚点、相关源码/CSV/JSON。没有调用原模型裁判，也没有独立重跑27条完整流程。

1.1统一半分代表清单/回执支持但未全量独立哈希重验；这不是原det.data_identity代码必然输出的半分。

1.3对有可追溯源码和结果的运行保守给半分；导出platform_exit_code=0只是适配值，不等于捕获进程退出码。

1.5与1.6在可追溯但未完整独立重算时通常给半分，已核对的简单核心量可给满分；明确相反或无依据的核心判定给零。

科学内容、稳定性与替代解释按报告的已执行证据及反例作人工判断；不是把文字声称的成功视为全部机器检查已通过。

2.2、4.2、4.3、6.5未核验到满足完整形式条件的证据时不计分；这表示本次未确认获得该分，而不是证明仓库绝无相关文件。

5.2所有条目待核验暂计零，不将未知新颖性伪装成已查明的非新颖。没有把整个0-100分量表重标定为94分。

P仅对Physics_02 F5的可见代码/产物直接错误计5分；这是人工类比的直接反驳，不冒充原白名单闭环的执行回执。

人工合并3组近重复发现；在敏感性列给出去掉这3组合并的结果，不将相近分数解释为稳健排名。

隐藏锚点按单条发现匹配；不要求科学发现必须属于两张非穷尽锚点，也不跨发现拼核心量获得满分。

### 未完成事项与解释限制

未完成128条发现的冻结T0以前逐条先前工作搜索和语义重叠审计。

网页可读内容及定向数值转录不等于完整仓库字节副本；未全量核验输入和输出哈希。

未生成原Evaluator的nomination、工程闭环、确定性核验回执或官方final_scoring_result.json。

本文分数是人工审阅暂评分，不是正式Evaluator得分，也不能保证是其下界。

库内模型来源混合；总体均分仅描述这批27份产物，不作为某一个模型的成绩。

**尤其注意：128条发现的5.2新颖性全部为待核验、暂计0。不是已查明不新颖，也没有把总分按94分重新归一化。**

把所有待核验5.2都改成满档，并保持其余判断不变，重新汇总后的均分为 **71.3**。这是单一假设情景，不是预测、置信区间或保证可达到的分数区间。

## 2. 27题汇总

| 题目 | 总分 | S1 | S2 | Y | A | B | H | P | 新颖性全满情景 | 不人工去重情景 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Astronomy_01_radio_burst_drift | **62.1** | 66.5 | 61.0 | 10 | 0 | 0 | 0.0 | 0 | 66.9 | 62.1 |
| Astronomy_02_gravity_spectral_structure | **70.6** | 66.5 | 66.0 | 10 | 0.5 | 1 | 7.5 | 0 | 75.4 | 70.6 |
| Astronomy_03_stellar_spectral_boundary | **68.9** | 67.5 | 67.0 | 10 | 0.5 | 0.5 | 5.0 | 0 | 73.7 | 68.9 |
| Chemistry_01_bond_enthalpy_domain | **67.1** | 66.5 | 61.0 | 10 | 0.5 | 0.5 | 5.0 | 0 | 71.9 | 67.1 |
| Chemistry_02_photoionization_delay | **66.3** | 67.5 | 66.5 | 10 | 0 | 0.5 | 2.5 | 0 | 71.1 | 66.3 |
| Chemistry_03_catalysis_response_landscape | **70.1** | 69.0 | 68.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.9 | 70.1 |
| EarthScience_01_glacial_fluvial_erosion | **68.2** | 66.5 | 66.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 73.0 | 68.2 |
| EarthScience_02_ice_particle_composition | **68.2** | 66.5 | 66.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 73.0 | 68.2 |
| EarthScience_03_ice_core_event_structure | **69.2** | 68.0 | 67.0 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.0 | 69.2 |
| EarthScience_04_ocean_thermal_coupling | **69.8** | 68.5 | 68.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.6 | 69.8 |
| Energy_01_photocatalytic_tradeoffs | **69.8** | 68.5 | 68.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.6 | 69.8 |
| Energy_02_flow_battery_operating_boundary | **69.2** | 68.5 | 65.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.0 | 69.2 |
| Energy_03_interface_response_boundaries | **69.4** | 68.5 | 66.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.2 | 69.4 |
| Energy_04_biohybrid_operating_tradeoffs | **72.6** | 66.5 | 63.5 | 10 | 1 | 1 | 10.0 | 0 | 77.4 | 72.6 |
| Information_03_signal_surface_invariants | **66.9** | 68.5 | 66.5 | 10 | 0.5 | 0 | 2.5 | 0 | 71.7 | 66.9 |
| Life_02_contextual_immune_response | **69.8** | 68.5 | 68.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.6 | 69.8 |
| Life_04_environmental_transfer_boundary | **68.2** | 66.5 | 66.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 73.0 | 68.2 |
| Life_05_developmental_interaction_structure | **69.8** | 68.5 | 68.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.6 | 69.8 |
| Math_02_compositional_invariants | **69.8** | 68.5 | 68.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 74.6 | 69.8 |
| Neuro_02_calcium_response_regions | **65.7** | 63.5 | 63.0 | 10 | 0.5 | 0.5 | 5.0 | 0 | 70.5 | 65.7 |
| Neuro_03_spatial_code_stability | **68.2** | 66.5 | 66.5 | 10 | 0.5 | 0.5 | 5.0 | 0 | 73.0 | 68.2 |
| Neuro_04_temporal_geometry | **67.1** | 66.5 | 61.0 | 10 | 0.5 | 0.5 | 5.0 | 0 | 71.9 | 67.1 |
| Physics_01_dynamic_nuclear_polarization_echoes | **47.6** | 53.0 | 49.0 | 6 | 0 | 0 | 0.0 | 0 | 54.4 | 47.6 |
| Physics_02_multiorbital_transition | **54.4** | 56.5 | 52.5 | 10 | 0.5 | 0.5 | 5.0 | 5 | 59.2 | 54.4 |
| Physics_03_controlled_dynamics_forecast | **59.4** | 56.0 | 51.5 | 8 | 0.5 | 1 | 7.5 | 0 | 64.2 | 61.4 |
| Physics_04_temporal_delay_transition | **54.4** | 55.5 | 53.0 | 8 | 0 | 0.5 | 2.5 | 0 | 61.2 | 54.4 |
| Physics_05_variational_circuit_simulation | **67.5** | 62.0 | 61.5 | 8 | 1 | 1 | 10.0 | 0 | 72.3 | 69.5 |

## 3. 逐题、逐发现评分依据

F编号沿用各题注册顺序。表内为原发现分；去重只影响每题汇总，不删除原发现记录。完整29列动作等级见Excel“发现明细”、CSV和JSON。这里的判断是人工证据评审，不是自动评分脚本原样生成的日志。

### 01. Astronomy_01_radio_burst_drift — 62.1/100

前两名：F1=66.5、F2=61.0；Y=10，H=0.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 像素重建不能恢复公布漂移 | 66.5 | 45个方法×事件组合无一达到10%误差；正对照、阈值与删事件分析有用，但同源重建不是独立观测。 |
| F2 | 快慢分支受分箱与来源影响 | 61.0 | 分箱宽度相差约10倍，样本重叠削弱独立分支解释；自举失败及拟合退化已保留。 |
| F3 | 规范化数据丢失来源信息 | 58.5 | 表格、聚合、图像及空块的来源盘点有意义；212/11240约1.886%，不是2.8%，辅助数量降档。 |
| F4 | 偏振分布差异及检验 | 60.5 | 可见CSV支持分布差异，但把观测自举上界与置换零假设95%阈值混用；主差异方向保留，区间/数量解释不得满分。 |
| F5 | 无法映射的图像限制复核 | 58.5 | 9幅未映射图像及仅5个事件限制结论；低相关不能替代来源身份核验。 |

**目标匹配：** A=0，B=0。主要发现是像素重建失败和数据加工边界，没有恢复目标A的快分支比例或目标B的慢/快分支比例及漂移量。

**补充：** 偏振检验误述在1.6、1.7和1.8降分；主要分布差异未被推翻，不另加P。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Astronomy_01_radio_burst_drift/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Astronomy_01_radio_burst_drift/validation_anchors.json

### 02. Astronomy_02_gravity_spectral_structure — 70.6/100

前两名：F3=66.5、F2=66.0；Y=10，H=7.5，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | B<C<A的噪声幅度排序 | 65.0 | 22/22配对排序、区间与删频段结果一致；保留中位数比值与总体RMS比值的区别。 |
| F2 | 固定谱网格与模型选择优势 | 66.0 | 19/24个诊断量落在同一网格下限；IC差1478.2可核对，但不是多份独立实验证据。 |
| F3 | 未检出稳健窄谱特征 | 66.5 | AR(1)背景和全扫描校正后，多数条件不显著；负结果限定在实际搜索带宽。 |
| F4 | 分辨率与检测功效边界 | 61.0 | 分辨率高于候选间隔，限制可分辨性；不同噪声设定下功效不宜混为统一保证。 |
| F5 | 截断网格和参数退化 | 65.5 | 频率、宽度与速度强耦合，局部IC比较不能充当置信域；交互模型和截断敏感性支持这一边界。 |

**目标匹配：** A=0.5，B=1。A：恢复B<C<A的方向，但0.5413是中位比值，不是冻结总体RMS比0.7546。B：同一发现F2覆盖约1478.2的信息准则差和1010频率等核心结果。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Astronomy_02_gravity_spectral_structure/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Astronomy_02_gravity_spectral_structure/validation_anchors.json

### 03. Astronomy_03_stellar_spectral_boundary — 68.9/100

前两名：F3=67.5、F4=67.0；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 金属丰度效应存在波段换号 | 65.0 | 紫外和可见区方向不同，交叉点随温度移动；无辨别力的交叉点零模型已撤回。 |
| F2 | 温度比值随波长交叉 | 66.5 | 35个丰度条件及黑体基线支持交叉结构；量化与同源依赖已限定。 |
| F3 | 归一化曲线可追溯至特定温度谱 | 67.5 | 校准窗口与窗外检验分开，舍入误差量级匹配；没有独立来源的新数据。 |
| F4 | 紫外重分箱产生近五倍因子 | 67.0 | 约5.001的比例在重分箱解释下成立，并比较了平移与求和替代；限定为处理来源证据。 |

**目标匹配：** A=0.5，B=0.5。A：恢复紫外随金属丰度降低的方向，未齐备冻结极端比值。B：恢复归一化曲线结构相近，未完整报告最低相关与峰位跨度。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Astronomy_03_stellar_spectral_boundary/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Astronomy_03_stellar_spectral_boundary/validation_anchors.json

### 04. Chemistry_01_bond_enthalpy_domain — 67.1/100

前两名：F5=66.5、F4=61.0；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 数据库结构域与基准覆盖缺口 | 60.0 | 完整数据库和基准的分子/键层级分开；域外27.62%和6类未覆盖结论有定位证据。 |
| F2 | 量子与学习排序取决于区域 | 59.0 | 大分子/域外优势有支持，小分子差异不显著；旧显著性表与改正文字并存，且大小与域外身份混杂。 |
| F3 | 参考异质性与误差下限的条件解释 | 60.5 | 两种估计依赖非真实重复或误差独立假设；已承认非高斯及真BDE差异，不能把2.83视为实测噪声。 |
| F4 | 传统参考通道较差且分键类反向偏差 | 61.0 | MAE及两类相反偏差稳健；两类均值异号本身不充分证明总体误差分布双峰。 |
| F5 | 弱键高估受选择效应影响但有残差 | 66.5 | 改用预测值分层后高估缩小约54%，分子聚类区间仍偏正；回归均值替代得到实际检验。 |

**目标匹配：** A=0.5，B=0.5。A：F1恢复数据库规模、分子数与域边界，但缺C–H占比0.66394核心量。B：F4列出三通道MAE，缺±3 kcal/mol覆盖率0.72744，故部分恢复。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Chemistry_01_bond_enthalpy_domain/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Chemistry_01_bond_enthalpy_domain/validation_anchors.json

### 05. Chemistry_02_photoionization_delay — 66.3/100

前两名：F4=67.5、F3=66.5；Y=10，H=2.5，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 实验复振幅的低阶角向结构 | 65.0 | 低阶模型和留出角度检验充分；残差约9.56e-5并非数学意义的精确相等。 |
| F2 | 理论角结构不能由简单映射解释 | 61.5 | 测试多种基底与匹配，保留匹配控制失败；degree20/22残差不同，精确阶数与Nyquist措辞过强。 |
| F3 | 相位导数匹配半Wigner延迟约定 | 66.5 | 能量轴与系数二的竞争约定被显式比较；结论限定为给定数据的定义一致性。 |
| F4 | 稠密曲线并非可见稀疏块的简单插值 | 67.5 | 多种插值与能量偏移远大于舍入误差；只能排除已测试生成路径。 |
| F5 | 实验—理论对应不具能量特异性；早先跨表示冲突来自求和顺序 | 60.0 | 修正复振幅与幅度/强度求和顺序后，原跨表示冲突撤回，按阶幅值对应仍成立；共同单调能量趋势尚未排除，因此不能把对应写成能量特异的机制验证或独立验证。 |

**目标匹配：** A=0，B=0.5。A：没有在单条发现中恢复31 eV角向对比的两项冻结核心量。B：F3恢复相位导数与延迟的方向/换算关系，但未报告冻结的两个相关系数。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Chemistry_02_photoionization_delay/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Chemistry_02_photoionization_delay/validation_anchors.json

### 06. Chemistry_03_catalysis_response_landscape — 70.1/100

前两名：F2=69.0、F1=68.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 两块因子网格的主效应集中度不同 | 68.5 | 完整因子网格、删水平与留出误差支持结构差异；未重复设计不能估实验噪声。 |
| F2 | 强因子效应不等于单调效应 | 69.0 | 线性、保序和类别基线比较区分幅度与顺序；分半稳定性支持非单调边界。 |
| F3 | 归一化响应来自成对通道的代数比 | 65.5 | (B−A)/(B+A)匹配两块数据，并比较A/B等替代；是代数结构而非新增机制。 |
| F4 | 面板间转移仅在局部成立 | 68.5 | 独立面板预测与常数基线比较，明确部分指标失败；强相关不保证较低预测误差。 |
| F5 | 成对读数比例差异与测量误差边界 | 66.5 | 普通和分块估计均低于1；误差变量修正已纠正，仍不能把共同噪声当独立重复。 |

**目标匹配：** A=0.5，B=0.5。A：F1恢复主因子相对重要性，但方差贡献不是冻结中位跨度。B：F3/F5涉及成对响应，却没有同一发现齐备三面板的冻结相关量。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Chemistry_03_catalysis_response_landscape/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Chemistry_03_catalysis_response_landscape/validation_anchors.json

### 07. EarthScience_01_glacial_fluvial_erosion — 68.2/100

前两名：F1=66.5、F2=66.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 宇宙成因率与时间乘积近固定厚度 | 66.5 | 重建原始量、研究级斜率与单位检查支持定义耦合；子集和总体的厚度统计不可替换。 |
| F2 | 去掉乘积约束后时间趋势不稳健 | 66.5 | 研究聚类后区间跨零，与逐行显著性不同；有力界定伪独立的影响。 |
| F3 | 方法影响在各层级都大于过程 | 53.0 | 无调整比较支持方法较强，但加入模型最后时0.103<0.120，完整病例亦反向；『每层都更强』不由证据推出。 |
| F4 | 侵蚀过程差异高度依赖分层 | 66.5 | 5个方法层中3正2负且差异跨度大；匹配与地理混杂分析支持条件性而非统一过程效应。 |
| F5 | 研究内聚类大幅降低独立信息量 | 65.0 | 研究内相关约0.726，研究级与逐行结果不同；有效样本量只是设计效应近似。 |

**目标匹配：** A=0.5，B=0.5。A：比较了冰川和非冰川条件，但非冰川不等于冻结的河流范围，且几何比7.58895未恢复。B：恢复侵蚀率×时间的近固定厚度关系，子集统计并非全部3642个样本的冻结量。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/EarthScience_01_glacial_fluvial_erosion/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/EarthScience_01_glacial_fluvial_erosion/validation_anchors.json

### 08. EarthScience_02_ice_particle_composition — 68.2/100

前两名：F1=66.5、F2=66.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 深度趋势受站点杠杆影响 | 66.5 | 五站整组置换及删站结论不显著；把深度效应限制在实际统计功效内。 |
| F2 | 叶绿素与丰度的负关系不够稳健 | 66.5 | 常规聚类回归偏负，但仅五组的精确/删组检验不足显著；未用单一p值强行定论。 |
| F3 | 聚乙烯占比变化集中于低丰度区 | 61.0 | 分段边界有数值支持，高丰度段斜率接近零；个别方向措辞和『精确』重建表述需降档。 |
| F4 | 组成与派生计数具有量化约束 | 63.5 | 百分比量化及计数残差表明派生列不是独立测量；明确近似量化而非逐项严格整数。 |
| F5 | 粒径与丰度方向的报告矛盾 | 45.5 | 所列负相关与『低丰度更陡』标题的方向不一致；分箱后低功效不能证明无效应。此处扣内容分，未完成原始曲线复算，不加P。 |

**目标匹配：** A=0.5，B=0.5。A：站间异质性及站点杠杆得到部分恢复，未齐备中位丰度比16.7778。B：聚乙烯/涂料的组成优势被涉及，但未齐备78.69%和0.90两个冻结量。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/EarthScience_02_ice_particle_composition/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/EarthScience_02_ice_particle_composition/validation_anchors.json

### 09. EarthScience_03_ice_core_event_structure — 69.2/100

前两名：F2=68.0、F4=67.0；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 列间几何依赖被过度等同于不独立 | 49.0 | 确有累积、深度和密度关系，但唯一深度上可拟合函数不能证明随机独立性丧失；若干『精确』关系仍有残差。 |
| F2 | 水当量积分与所给累计量不一致 | 68.0 | 物理积分约21.222，对照累计27.060；左右端点、冰当量替代均不足弥合。解释仍取决于列含义。 |
| F3 | 背景硫酸盐的深度趋势较弱 | 62.5 | GLS和分块检验不支持强趋势，正对照有价值；80%功效界不等于已排除某幅度效应。 |
| F4 | 事件数量对检测合同高度敏感 | 67.0 | 720种设置可产生极不一致的事件数量；位置富集不等于恢复固定事件目录。 |
| F5 | 已命名与未命名事件差异未定 | 59.0 | 精确检验和宽区间不支持稳定组间幅度差；标题中的位置差异未获后续检验支持。 |

**目标匹配：** A=0.5，B=0.5。A：事件富集与检测阈值依赖得到部分覆盖，但14事件及局部中位富集2.7043未在同一发现齐备。B：背景弱趋势有部分对应，未恢复冻结四分位背景比。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/EarthScience_03_ice_core_event_structure/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/EarthScience_03_ice_core_event_structure/validation_anchors.json

### 10. EarthScience_04_ocean_thermal_coupling — 69.8/100

前两名：F2=68.5、F5=68.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 约109千年的宽周期结构 | 64.5 | 年龄/深度两轴与AR(1)搜索支持宽结构，分箱与留出结果较弱；217千年并非109千年的二次谐波。 |
| F2 | 双坐标相位吻合但并非独立重复 | 68.5 | 分离区段相位一致，并承认坐标高度共源和幅度随窗变化；不将重叠窗当新证据。 |
| F3 | 年龄坐标差异局限于浅部 | 65.5 | 删浅部和对抗性年龄处理定位了坐标敏感边界；局部一致不能外推所有变换稳健。 |
| F4 | 沉积速率尖峰依赖年龄差分处理 | 67.0 | 不同网格估计与原始平滑不一致，支持加工边界；不能直接宣称快速沉积事件。 |
| F5 | 变化率耦合主要来自共同分母 | 68.5 | 保结构零模型复现多数正相关；同时保留速率与水平的负相关，区分两个不同命题。 |

**目标匹配：** A=0.5，B=0.5。A：发现年龄坐标局部差别与总体近似一致，未恢复两个冻结相关量。B：非单调变化得到覆盖，但8块中位跨度/跳变量未齐备。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/EarthScience_04_ocean_thermal_coupling/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/EarthScience_04_ocean_thermal_coupling/validation_anchors.json

### 11. Energy_01_photocatalytic_tradeoffs — 69.8/100

前两名：F1=68.5、F2=68.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 动力学形状分组可辨而函数形式未定 | 68.5 | 各曲线仅5点；模型竞争、留点失败与AICc边界保留，未把拟合优胜解释为唯一机制。 |
| F2 | 红端作用衰减不能由吸收限制解释 | 68.5 | 留出红端预测严重失配，多个竞争模型和转换凹性分析有区分力；不越界指认机制。 |
| F3 | 两条瞬态轨迹的长期响应不同 | 68.5 | 多个处理伪影与模型被比较；仍只有两条完整轨迹，数百延迟点不是重复实验。 |
| F4 | 负载仅有局部采样峰而非可识别最优 | 65.0 | 4个设置给出内点峰，删点与模型比较不能固定连续最优位置；对不确定性保持克制。 |
| F5 | 跨模态映射受身份缺口限制 | 60.0 | 仅两种配方能跨模态连接且排序相反；无标识、不同量纲的数据不应拼接成通用规律。 |

**目标匹配：** A=0.5，B=0.5。A：四配方排序和约54.8倍尺度得到恢复，未齐备同估计器的最大/最小冻结斜率。B：吸收与作用谱总体同向，但重点为红端偏离，两个冻结相关量未齐备。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Energy_01_photocatalytic_tradeoffs/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Energy_01_photocatalytic_tradeoffs/validation_anchors.json

### 12. Energy_02_flow_battery_operating_boundary — 69.2/100

前两名：F5=68.5、F1=65.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 充放电间隙总体加大且有局部例外 | 65.5 | 全点比较揭示少量放电曲线交叉，改正了最初全域单调表述；模板重复不是独立采样。 |
| F2 | 效率汇总与中SOC电压比近似相容 | 65.5 | 比较替代映射与留出误差，主动剔除排序置换伪检验；舍入解释未获过度奖励。 |
| F3 | 功率峰局限于给定设置且末端不兼容 | 65.0 | 300处功率330为观测峰，删除峰后不稳定；末端隐含电压超出曲线范围，有明确范围边界。 |
| F4 | 效率损失具有曲率 | 65.0 | 局部斜率与二次/幂函数均支持非线性下降；没有量测重复，局部导数对舍入敏感。 |
| F5 | 循环衰减可见但分段改变未定 | 68.5 | 分块区间支持负斜率，变化点在分块零模型中不显著；不能把平台窗口当稳健机制。 |

**目标匹配：** A=0.5，B=0.5。A：效率下降与功率采样峰分别得到恢复，但没有单条发现齐备冻结联合条件。B：600周期及斜率接近目标；端点或20点均值保留率不同于冻结50点中位定义。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Energy_02_flow_battery_operating_boundary/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Energy_02_flow_battery_operating_boundary/validation_anchors.json

### 13. Energy_03_interface_response_boundaries — 69.4/100

前两名：F4=68.5、F1=66.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 对数曲线高温弯曲普遍存在 | 66.5 | 12条高温对数曲线曲率均负；原尺度不是全部同向，已限定分析尺度。 |
| F2 | 有效斜率存在条件差异 | 66.5 | 斜率范围与比例经更正，仍不能区分本征能量与接近饱和的位置效应。 |
| F3 | 中间区间排序比端点稳定 | 65.0 | 分温区和删条件分析定位排序边界；端点相关弱，不支持全范围不变序。 |
| F4 | 留出预测偏向饱和模型但天花板未观测 | 68.5 | 训练≤280、预测300/320的比较有实质性分离；上限位置仍欠稳定，不等于已测得天花板。 |
| F5 | 核对数据部分为完全重复 | 63.5 | 2/4轨迹逐值重复，另外共享曲线只是近似；直接否定把核对集全部视为新独立证据。 |

**目标匹配：** A=0.5，B=0.5。A：热激活的正方向与高温斜率得到部分恢复，但高温子集斜率不是全范围冻结最小斜率。B：核对曲线相近/重复得到部分恢复，未齐备所有共享曲线的冻结最小相关。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Energy_03_interface_response_boundaries/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Energy_03_interface_response_boundaries/validation_anchors.json

### 14. Energy_04_biohybrid_operating_tradeoffs — 72.6/100

前两名：F3=66.5、F5=63.5；Y=10，H=10.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 拼接区段结构不能证明时间轴仅局部有效 | 50.5 | 三段值域与支持集不同是可观测事实，但时间取值少不自动意味着无效；只保留分段数据结构结论。 |
| F2 | 同一时间出现分支差异 | 61.0 | 共享时间读数相差很大，支持路径/来源差异；不能在未知实验身份下唯一认定物理迟滞。 |
| F3 | 条件耦合主要由尾段与合并驱动 | 66.5 | 基线近零相关、尾段强相关而差分后减弱；分段/切点敏感性支持条件边界。 |
| F4 | 拟合列是支持表示而非新增测量 | 57.0 | 拟合与原始高度相关并更平滑；第三列0.945不等于严格单调，且平滑器来源未被唯一识别。 |
| F5 | 剂量最优未定而扰动响应排序清楚 | 63.5 | 三剂量无误差不能定最优；控制稳定且b>a衰减明显。SD非SE及时间相关性限制合并显著性。 |

**目标匹配：** A=1，B=1。A：F5表内端点可计算控制、a、b保留率，符合1.00478、0.86558、0.70862及方向。B：同一F5明确3个剂量、无误差项、删点最优不可识别。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Energy_04_biohybrid_operating_tradeoffs/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Energy_04_biohybrid_operating_tradeoffs/validation_anchors.json

### 15. Information_03_signal_surface_invariants — 66.9/100

前两名：F1=68.5、F2=66.5；Y=10，H=2.5，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 对数响应近似单增益轴 | 68.5 | 秩一解释率约0.955，异方差零模型和完整留出检验有区分力；其尺度依赖由F2明确。 |
| F2 | 归一化改变低维结构强弱 | 66.5 | 原尺度、对数和按通道标准化结果不同；是F1的适用边界，不应误写成任意归一化都稳健。 |
| F3 | 两面板同网格水平并不直接复现 | 65.0 | 共同网格上r≈0.0168且转移R²<0；对齐、重标度和弱共享成分均报告。逐格推断仍需系统级独立性限定。 |
| F4 | 跨面板噪声量存在有限共享 | 59.0 | 控制边际后相关仍在，但CV(log响应)不是原响应的尺度不变CV；不能仅据此断言乘性噪声机制。 |
| F5 | 零信号箱有面板依赖的边界 | 66.5 | 基线箱的偏离和换号是局部而非通用关系；方法与完整单位扰动支持边界定位。 |

**目标匹配：** A=0.5，B=0。A：主面板的跨重复结构稳定获得部分支持，未在同一发现齐备两面板的冻结重复相关量。B：跨面板水平/噪声分析不是独立面板逐行逐列的归一化通道对齐，核心边界未恢复。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Information_03_signal_surface_invariants/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Information_03_signal_surface_invariants/validation_anchors.json

### 16. Life_02_contextual_immune_response — 69.8/100

前两名：F2=68.5、F4=68.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 扰动提高各细胞的平均响应 | 65.0 | 18/18单元增加且不同汇总定义同向；共享assay下的18单元检验不能充当18次独立实验。 |
| F2 | 组织区室排序不可靠地跨assay转移 | 68.5 | 留一assay外预测与常数基线比较直接检验转移，失败被纳入最终结论。 |
| F3 | 计数与比例方向同向但排序不一致 | 65.0 | 18单元两端点均增加，某assay排序反向；端点差异边缘显著被如实限定。 |
| F4 | 时间轨迹差异未检出且功效不足 | 68.5 | 明确最小可检测效应远高于观测差，不把不显著当等效。 |
| F5 | 干预排序较一致但样本规模小 | 68.5 | 精确与渐近检验及交换敏感性并列；只有三个assay，不能无限推广排名。 |

**目标匹配：** A=0.5，B=0.5。A：F1的18/18增加与冻结方向一致，但用几何均值比而非冻结均值比。B：归入F1的支持结果覆盖A类强、B类弱，缺同口径的两项冻结比值。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Life_02_contextual_immune_response/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Life_02_contextual_immune_response/validation_anchors.json

### 17. Life_04_environmental_transfer_boundary — 68.2/100

前两名：F2=66.5、F3=66.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 采样层级与角色编码限制独立性 | 65.5 | 30个单位含212个位点，ICC与编码重复支持层级边界；不将逐位点当同量独立受试对象。 |
| F2 | assay关联强于普遍单调梯度 | 66.5 | 阈值关联和单调/多重检验结论不同；以单位分组检验避免放大样本数。 |
| F3 | 病程关联在单位级检验中消失 | 66.5 | 逐位点显著与单位置换p≈0.255形成直接对照；阈值搜索已纳入零模型。 |
| F4 | 表面与症状读数不能充当独立复现 | 57.5 | 表面阳性为环境阳性子集且症状分组很小；20%量化步长不能唯一证明分母恰为5。 |
| F5 | 综合层级校正再次限定显著性 | 66.5 | Holm与单位级重估有用，但病程关联丢失是F3的同一核心结论；综合汇总不另计独立发现数量。 |

**目标匹配：** A=0.5，B=0.5。A：恢复assay指标与环境阳性的关联边界，未齐备25.69/33.04及−0.60206核心量。B：位点异质性与症状弱差异分布在不同发现，不能拼接成完整恢复。

**去重：** F3+F5：F5的层级校正总结重复F3的核心病程显著性丢失，不因多报一种校正重复增加发现产量；保留组内最高分。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Life_04_environmental_transfer_boundary/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Life_04_environmental_transfer_boundary/validation_anchors.json

### 18. Life_05_developmental_interaction_structure — 69.8/100

前两名：F1=68.5、F4=68.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 阶段效应为家族内有限增量 | 68.5 | 家族聚类与214家族留出有支持；使用单位均值的条件预测不是冷启动预测，均衡子集变弱已报告。 |
| F2 | 家族和单位差异主导总体方差 | 66.5 | ICC、方差分解和分半结果一致；不能分开生物差异与技术漂移的限制明确。 |
| F3 | 离散化p值与强度并非同一证据 | 66.5 | 87.6%落在最小p值档，非地板数据关联弱；用受限置换而非把离散证据当连续精度。 |
| F4 | 交互方向存在广泛家族异质性 | 68.5 | 符号翻转和留家族结果显示非普遍效应；分类阈值及102/214改善边界已列出。 |
| F5 | 强度连续而显著性标签不稳定 | 66.5 | 强度r≈0.908而标签κ≈0.122；先前缺失填充错误已纠正，不处罚已撤回数值。 |

**目标匹配：** A=0.5，B=0.5。A：阶段正向变化获得部分覆盖，未齐备早晚阶段均值/位移及9对比较。B：家族占据和证据不稳定得到部分覆盖，但显著家族948及持久/单次占比未以冻结定义恢复。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Life_05_developmental_interaction_structure/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Life_05_developmental_interaction_structure/validation_anchors.json

### 19. Math_02_compositional_invariants — 69.8/100

前两名：F1=68.5、F2=68.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 简单线性组成不变量不足 | 68.5 | CLR、伪计数及PCA结果排除特定低维线性解释；不据此否定所有非线性不变量。 |
| F2 | 协变量与丰富度关系经控制仍存在 | 68.5 | 深度控制、稀释、删组及保序留出检验支持正关联；少量深度方向措辞不影响主结论。 |
| F3 | 场景仅保留次序而不能数值转移 | 68.5 | 场景排序在多数配对保留，直接转移R²为负；清楚区分序关系与尺度可迁移性。 |
| F4 | 丰富度关联在高协变量段饱和 | 68.5 | 阈值上下相关不同且差异区间不含零；分段模型与完整留出支持局部边界。 |
| F5 | 部分组成相关由零值驱动 | 67.0 | 去零后相关显著缩小，对组成约束与零值处理做了实际比较；未外推因果机制。 |

**目标匹配：** A=0.5，B=0.5。A：整体及条件相关的正方向获得覆盖，未完整对齐冻结最小组内相关。B：场景间次序稳定有支持，报告平均差而不是两个冻结中位差。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Math_02_compositional_invariants/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Math_02_compositional_invariants/validation_anchors.json

### 20. Neuro_02_calcium_response_regions — 65.7/100

前两名：F3=63.5、F2=63.0；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 刺激对多响应量的影响相对稳定 | 57.5 | 平衡子集及删batch保持多项方向；细胞嵌套与批次相关使逐细胞显著性不能获满分。 |
| F2 | 区域差异部分独立于条件覆盖 | 63.0 | 宽度/频率在平衡及删batch后仍有差别；AUC不稳健与覆盖不足被保留。 |
| F3 | 年龄与批次不可分离 | 63.5 | 年龄与batch一一对应，设计矩阵秩缺失及VIF支持不可识别性；个别显著批次数量表述不准确。 |
| F4 | 年龄效应不能被普遍推广 | 63.0 | 只有部分响应量出现年龄关联，且被批次混杂限制；不将拟合系数解释成独立年龄机制。 |

**目标匹配：** A=0.5，B=0.5。A：刺激增加宽度方向一致，但报告调整后差值而非两区域冻结宽度比。B：区域宽度/频率反向关系得到部分恢复，未报告两项冻结比值。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Neuro_02_calcium_response_regions/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Neuro_02_calcium_response_regions/validation_anchors.json

### 21. Neuro_03_spatial_code_stability — 68.2/100

前两名：F1=66.5、F2=66.5；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 新环境稳定性在动物级下降 | 66.5 | 区分单元与动物平均，动物级汇总仍偏负；只有3个对照稳定性动物，范围必须受限。 |
| F2 | 信息量下降在动物级证据较弱 | 66.5 | 逐单元与动物p值不同，保留动物级约0.074的不确定性而不宣称强显著。 |
| F3 | 规则性不显著不能证明真零效应 | 46.0 | 把未拒绝写成『真零』缺少等效检验或充分功效，且区间很宽；核心推论不获支持分。 |
| F4 | 部分空间指标显著性受分析单位影响 | 66.5 | 方向/场内响应在动物层级变弱，体现伪重复风险；原p≈0.056也不应称已显著。 |

**目标匹配：** A=0.5，B=0.5。A：动物级环境改变后稳定性下降得到恢复，但文内主要量不是冻结的两个动物平均差。B：平均活动与稳定性的联合解离未在同一发现完整量化。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Neuro_03_spatial_code_stability/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Neuro_03_spatial_code_stability/validation_anchors.json

### 22. Neuro_04_temporal_geometry — 67.1/100

前两名：F1=66.5、F2=61.0；Y=10，H=5.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 距离与平均延迟无明显单调关系 | 66.5 | 41个单位的Pearson、Spearman、置换、区间及删点一致；限定为未支持简单单调关系。 |
| F2 | 距离与延迟离散度关联很弱 | 61.0 | 两相关量近零；零方差不自动意味着确定性，个别关于均值可用性的表述与F1矛盾。 |
| F3 | 事件前后信号有条件差别 | 55.5 | B组二分差异较强而A组较弱；事件嵌套与序列相关未由普通置换充分处理。 |
| F4 | 两组边际分布未检出显著差异 | 57.0 | KS不显著不能构成等效或通用分布不变性，小组样本量仅35。 |
| F5 | 跨文件只有三个单位可匹配 | 55.0 | 重叠不足以稳定外推，但n=3并不意味着相关系数数学上不可计算；保留覆盖边界，扣过强措辞。 |

**目标匹配：** A=0.5，B=0.5。A：事件前后信号差异方向与目标部分吻合，但二分正负时间不是冻结连续时间相关。B：距离与均值、离散度的两个冻结相关分别在F1/F2，不拼接为满恢复。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Neuro_04_temporal_geometry/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Neuro_04_temporal_geometry/validation_anchors.json

### 23. Physics_01_dynamic_nuclear_polarization_echoes — 47.6/100

前两名：F1=53.0、F2=49.0；Y=6，H=0.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | EPR整条包络随反转时间衰减 | 53.0 | 六个条件的包络幅度下降可描述；不等于定位每条轨迹中2*t_inv处的回波，也没有独立重复误差。 |
| F2 | NMR积分对反转时间呈局部非单调 | 49.0 | 固定相位下440条件峰值可见，但相位一变最优不同；只有六点，无重复不足以称『并非噪声』。 |
| F3 | EPR与NMR聚合量反向关联 | 38.5 | 仅六个条件r≈−0.78，双侧显著性不足约0.05且共用扫参；不能据此证明普遍耦合或排除噪声。 |

**目标匹配：** A=0，B=0。A：主要分析整条EPR包络幅度，未恢复2*t_inv局部回波时刻与短/长归一化幅度。B：NMR积分/相位条件峰值不等于冻结最低点时刻与局部残留的联合命题。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_01_dynamic_nuclear_polarization_echoes/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Physics_01_dynamic_nuclear_polarization_echoes/validation_anchors.json

### 24. Physics_02_multiorbital_transition — 54.4/100

前两名：F4=56.5、F1=52.5；Y=10，H=5.0，P=5。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 能隙过阈值具有分支依赖 | 52.5 | 6/8分支越过阈值，支持有限范围差异；不同分支起点不直接反驳中位曲线的共同起点。 |
| F2 | 最低纠缠谱简并在有限网格改变 | 50.0 | 最低层级从1到2的变化可描述；不等于全谱多重简并率，更不能独自确认热力学相变。 |
| F3 | 局域矩增大而电荷波动下降 | 51.5 | 单调趋势有数据支持；光滑有限网格无法据以排除临界行为。 |
| F4 | 有限网格相关长度不宜作相区证据 | 56.5 | 极少近零逆相关长度及负值伪影支持稳健性警告；没有形成独立相图验证。 |
| F5 | 外推不一致的主要数值由系数错误造成 | 25.5 | polyfit斜率/截距错置导致错误外推值与残差；核心数值、幅度和解释归零，另按持存确定性错误扣5分。 |

**目标匹配：** A=0.5，B=0.5。A：分支能隙过阈值方向覆盖部分对象，但不是冻结双中位曲线共同0.8–0.9起点。B：最低谱简并与局域矩趋势分别出现，未在同一发现齐备联合转变边界。

**额外错误扣分：** F5扣5分：np.polyfit返回的斜率和截距被交换使用，既把斜率写成外推截距，又用错误公式算拟合残差。CSV中U/W=1、dSz=1的3.74849其实是斜率，真实线性拟合截距字段值应为−0.0345247；−1386.74的所谓R²不是该OLS拟合的R²。未在最终报告撤回。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_02_multiorbital_transition/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Physics_02_multiorbital_transition/validation_anchors.json

### 25. Physics_03_controlled_dynamics_forecast — 59.4/100

前两名：F2=56.0、F1=51.5；Y=8，H=7.5，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 主频预测在各扫描上误差较小 | 51.5 | 三扫描的绝对/相对频率误差列明；FFT分辨率和有效匹配覆盖率限制精度解释。 |
| F2 | 全轨迹预测质量随条件明显下降 | 56.0 | 三条相关约0.9595/0.8595/0.6687及归一化误差量化了局部差异；无独立重复或稳健外推。 |
| F3 | 频率近似不保证完整波形准确 | 44.5 | 定性区分频率与波形合理，但直接比较不同量纲的MAE和RMSE不能建立大小矛盾。 |
| F4 | 第三条件的频率优势需区分绝对与相对误差 | 40.0 | 第三条件绝对MAE最小，但相对误差0.00491大于第一条件0.00305；过强『最准确』解释降分，与F3同属跨指标解离。 |

**目标匹配：** A=0.5，B=1。A：F1恢复低MAE与扫描方向，但缺冻结最小频率相关和全部斜率一致条件。B：F2的三条相关和归一化RMSE与冻结量完整对应。

**去重：** F3+F4：两条都用相同扫描/轨迹描述主频准确但波形不准，F4是F3的单条件展开；合并后取高分。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_03_controlled_dynamics_forecast/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Physics_03_controlled_dynamics_forecast/validation_anchors.json

### 26. Physics_04_temporal_delay_transition — 54.4/100

前两名：F3=55.5、F2=53.0；Y=8，H=2.5，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 分段转变的核心变量映射未成立 | 27.5 | 脚本直接把below_bc两条Plot列作为B和V；真正B,V表随B下降，报告则上涨。未解决语义冲突就排除理论，核心证据不通过；未全量重放，暂不另加P。 |
| F2 | 相关时间随条件大幅缩短 | 53.0 | 积分和指数拟合的约33–37倍变化同向；五点临界拟合病态，正确未把β≈544当物理指数。 |
| F3 | 有限尺寸变化可见而精确外推未定 | 55.5 | 1/N与自由指数外推明显不同；删小尺寸结果支持收敛趋势，而非唯一热力学临界值。 |
| F4 | 真实网络延迟总体下降但非严格单调 | 48.5 | 两网络下降有支持；78–82%相邻下降不是严格单调，端点比约1.34万和0.36万而不是10万。 |
| F5 | 辅助临界现象不能证明没有时间漂移 | 38.5 | 低R²不是无漂移的等效证据，非零趋势在27k点上尤其需处理自相关；『一致』也不排除强竞争解释。 |

**目标匹配：** A=0，B=0.5。A：报告的正向V(B)和约1.42边界与冻结约3.673及负斜率不符，变量映射未通过。B：相关时间按条件下降得到部分覆盖，但临界值与指数1.69357未恢复。

**补充：** F1的B/V列映射和F4的倍数错误降低发现分；除可见源码/表格直接定位的P2错误外，不把未充分重放的问题计入P。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_04_temporal_delay_transition/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Physics_04_temporal_delay_transition/validation_anchors.json

### 27. Physics_05_variational_circuit_simulation — 67.5/100

前两名：F3=62.0、F1=61.5；Y=8，H=10.0，P=0。

| 发现 | 标题 | 分数 | 人工评分依据 |
|---|---|---:|---|
| F1 | 冻结案例的精确成本层近机器精度 | 61.5 | 从12个冻结观测可直接核对最小值及舍入误差；是对提供数据的正确分析，不是本次重新实现量子模拟。 |
| F2 | 叠加层数后全态保真度降低 | 59.5 | 3/3案例p2低于p1，但不能据门级较高保真度断言误差只集中在最后阶段，累积误差仍是竞争解释。 |
| F3 | 三个图案例均满足冻结门槛 | 62.0 | 逐案例重算p1≥0.94、p2≥0.92、压缩≥0.95全部通过；阈值来自输入观测合同，不是参与者新做的分离验证。 |
| F4 | 优化角度记录一致需查明记录机制 | 54.0 | 初始/优化角度逐值相同与步数并存是可审计现象；输出cost非零不能证明cost改变，也不能排除记录别名或原地更新。 |

**目标匹配：** A=1，B=1。A：F1的12案例最小精确保真度与冻结原始JSON一致。B：F2/F3中单条表格齐备p1、p2、压缩及门操作最小保真度，覆盖全部核心量。

**去重：** F2+F3：同三个图、同一组p1/p2/压缩保真度，分别报告下降与过阈值，保守视为同一性能发现的幅度和边界两面；取高分，不双计发现数量。

产物来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_05_variational_circuit_simulation/workspace/output/Result.md
冻结目标来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/tasks/Physics_05_variational_circuit_simulation/validation_anchors.json

## 4. 定向核查

以下数值来自可读网页中的源码、表格或JSON，再作局部算术核对；没有把这种核查描述成独立复跑全部原始数据。

### physics_02_polyfit_order

返回顺序为斜率、截距；原脚本命名和预测公式错置。未重跑实际原始外推数据，所以没有编造修正后的真实R²。

```json
{
  "numpy_polyfit_on_y_2x_plus_3": [
    2.000000000000001,
    2.999999999999999
  ],
  "correct_interpretation_slope": 3.7484932756116995,
  "correct_interpretation_intercept": -0.034524695524818474,
  "ols_with_intercept_fit_r2_cannot_be_negative_on_training_data": true
}
```

来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_02_multiorbital_transition/workspace/output/analysis/correlation_and_extrap.py；https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_02_multiorbital_transition/workspace/output/results/gap_extrapolation_fit.csv

### physics_05_frozen_observations

两锚点核心量可直接核对；不意味着重新生成量子态或重跑原作者完整模拟。

```json
{
  "minimum_exact_fidelity": 0.9999999999999996,
  "maximum_absolute_deviation_from_one": 2.220446049250313e-15,
  "minimum_p1": 0.9805953975270472,
  "minimum_p2": 0.9756043403530013,
  "minimum_compression": 0.9991800836013702,
  "minimum_gate": 0.9904045325288312,
  "all_p2_lower_than_p1": true,
  "passes_9_threshold_comparisons": 9
}
```

来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_05_variational_circuit_simulation/workspace/data/source_data/computational_observations.json

### astronomy_01_tv_thresholds

两种阈值不能混用：相应比较为0/6与6/6；分布差异方向仍有支持，降辅助量与表述分而不额外P。

```json
{
  "number_above_observed_bootstrap_upper": 0,
  "number_above_null_p95": 6
}
```

来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Astronomy_01_radio_burst_drift/workspace/output/results/polc_s03_channel_contrast.csv

### physics_04_delay_fold

不是10^5倍；降1.6和解释相关项。没有对两条完整曲线独立重放，未把该幅度错误另计P。

```json
{
  "Thiers_fold": 13386.308068459659,
  "Workplace_fold": 3628.6549707602344
}
```

来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Physics_04_temporal_delay_transition/workspace/output/Result.md

### energy_04_endpoint_ratios

F5同一表格足以恢复三个端点比例，显示控制近1、a下降、b下降更多。

```json
{
  "control": 1.0047762304996433,
  "condition_a": 0.8655755048697762,
  "condition_b": 0.7086176552320376
}
```

来源：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/runs/Energy_04_biohybrid_operating_tradeoffs/workspace/output/Result.md

## 5. 29项权重

| 动作 | 名称 | 权重 | 类别 |
|---|---|---:|---|
| 1.1 | 数据身份 | 4 | 证据 |
| 1.2 | 对象、范围与独立单位 | 4 | 证据 |
| 1.3 | 执行回执 | 6 | 证据 |
| 1.4 | 产物可追溯性 | 4 | 证据 |
| 1.5 | 核心判定数量核验 | 10 | 证据 |
| 1.6 | 辅助数量核验 | 5 | 证据 |
| 1.7 | 证据能否推出结论 | 8 | 证据 |
| 1.8 | 纳入反例与全部结果 | 4 | 证据 |
| 2.1 | 不确定性 | 3 | 稳定性 |
| 2.2 | 完整流程重复 | 3 | 稳定性 |
| 2.3 | 样本扰动 | 3 | 稳定性 |
| 2.4 | 方法扰动 | 3 | 稳定性 |
| 2.5 | 有效性边界 | 3 | 稳定性 |
| 3.1 | 简单基线 | 3 | 替代解释 |
| 3.2 | 负对照 | 3 | 替代解释 |
| 3.3 | 最强竞争解释 | 4 | 替代解释 |
| 3.4 | 混杂或加工伪影 | 2 | 替代解释 |
| 3.5 | 独立交叉验证证据 | 3 | 替代解释 |
| 4.1 | 分离验证来源 | 2 | 分离验证 |
| 4.2 | 分离验证执行证据 | 3 | 分离验证 |
| 4.3 | 预冻结判据 | 3 | 分离验证 |
| 4.4 | 非循环验证链 | 2 | 分离验证 |
| 5.1 | 目标来源隔离 | 4 | 独立来源与新颖性 |
| 5.2 | 冻结时点以前的新颖性 | 6 | 独立来源与新颖性 |
| 6.1 | 独立的下一步证据路线 | 1 | 下一步测试 |
| 6.2 | 测量与预期的对齐 | 1 | 下一步测试 |
| 6.3 | 更新结论的规则 | 1 | 下一步测试 |
| 6.4 | 资源可行性 | 1 | 下一步测试 |
| 6.5 | 预冻结可执行测试规则 | 1 | 下一步测试 |

## 6. 文件与复算

`truthinsightbench_manual_scores.xlsx`：说明、总分、发现明细、去重、目标恢复、评分尺、参数、定向核查8个工作表。等级输入与汇总公式分开；输入等级改变后，发现分、去重分、S1/S2、Y/H/P和总分重新计算。Top2发现ID是原审阅记录，不随改分自动更新。

`manual_review.json`：完整128×29等级、题目与发现判断、目标匹配、去重、错误记录、来源和限制说明。

`task_scores.csv` 与 `finding_scores_29_actions.csv`：便于其他程序读取的汇总与评分矩阵。CSV为UTF-8 BOM。

`targeted_checks.json`：5项局部核查的输入、结果与来源。

`recompute_scores.py`：仅使用Python标准库，从人工等级重算分数并检查原记录是否一致。它不重新评判科学内容，不执行原Evaluator。

```bash
python recompute_scores.py manual_review.json
python recompute_scores.py manual_review.json --novelty full
python recompute_scores.py manual_review.json --no-dedup
```

### 资产与版本记录

仓库：https://github.com/ZYDTR/truthinsightbench-dwf-scoring

本次读取分支：main；未独立记录commit hash。导出清单创建时间：2026-09-24T05:40:05.534223+00:00。

导出评分资产SHA-256：`157eb63e27668a6412d4895b28ca9ad6d0da9f75887c9b15b79bff7ae027dd95`。这是仓库清单提供的标识，不是本次对全部仓库字节重新计算的哈希。

动作评分尺：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/specification.json

最终汇总脚本：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/finalize_scores.mjs

目标匹配规则：https://raw.githubusercontent.com/ZYDTR/truthinsightbench-dwf-scoring/main/TruthInsightBench/evaluation/evaluator/scoring_rules.mjs

分数保留一位小数只是汇总算术的精度，不代表人工判断的不确定性只有0.1分。