# Lookalike-AC 赛题完整方案

> 赛题来源：[2026 iFLYTEK AI 开发者大赛 — Lookalike-AC](https://challenge.xfyun.cn/topic/info?type=Lookalike-AC)  
> 用途：指导后续 vibe coding 的工程实现与迭代路线

---

## 一、赛题信息整理

### 1.1 赛事背景

在**广告投放、用户增长、推荐营销**等场景中，企业已沉淀一批**高价值用户**（高转化 / 高活跃 / 高付费 / 高商业价值）。核心问题是：

> 如何基于已知高价值种子用户，在更大规模的候选人群中，识别**更可能成为未来高价值用户**的目标对象？

传统人工规则 / 粗粒度标签难以在海量用户中高效扩展。随着用户画像、行为序列、平台响应、广告位偏好、出价等多模态特征积累，**数据驱动的高价值人群扩展（Lookalike / Audience Expansion）** 成为智能营销的关键能力。

本赛题是**真实业务场景**下的种子人群扩展任务，兼具业务价值与算法挑战。

### 1.2 赛事任务（你要做什么）

| 维度 | 说明 |
|------|------|
| **输入** | 某一历史窗口的**种子人群** + **候选大盘人群** + 脱敏后的 **UV 级多模态特征** |
| **输出** | 对候选大盘中**每个用户**输出一个 **score（排序分数）**，score 越大表示越可能成为未来高价值用户 |
| **本质** | **大规模用户排序 / 人群圈选**，不是传统单条样本分类 |
| **约束** | **不能泄露未来信息**；只能利用历史窗口内的种子 + 特征 |

**关键业务逻辑：**

```
窗口 T 的种子用户 S_T
    ↓  模型学习「像种子的人」
候选大盘 U_T（含种子，但评测时会剔除 S_T）
    ↓  按 score 排序，取 Top-K
扩展人群包
    ↓  与下一窗口 T+1 的真实高价值用户比对
未来标签 = seed_{T+1} \ S_T（新增高价值用户）
```

### 1.3 数据说明与字段

#### （1）种子人群数据（Seed）

- **格式**：Parquet
- **字段**：

| 字段 | 说明 |
|------|------|
| `plat` | 平台名称 |
| `masked_id` | 脱敏后的用户唯一标识 |

- **注意**：种子数据**不含用户特征**；种子用户**包含在候选大盘中**，需通过 `masked_id` 去大盘表 join 特征。

#### （2）候选大盘数据（Candidate Pool）

候选大盘为**待圈选的人群池**。每条数据对应一个 **UV 级用户样本**，包含匿名化后的**静态特征**、**动态离散特征**以及**连续统计特征**。

**字段明细：**

| 数据 ID | 数据类型 | 数据名称 | 说明 |
|---------|----------|----------|------|
| `masked_id` | string | 用户 ID | 脱敏后的用户唯一标识，与种子表 join 键 |
| `make` | string | 厂商 | 设备厂商（静态） |
| `model` | string | 机型 | 设备型号（静态） |
| `province` | string | 省份 | 地域（静态） |
| `city` | string | 城市 | 地域（静态） |
| `tags` | array\<int\> | 标签 ID 数组 | 用户标签，多值离散特征 |
| `adunit_req_map` | map\<int,int\> | 广告位 → 请求次数 | 动态统计：各广告位累计请求 |
| `adunit_imp_map` | map\<int,int\> | 广告位 → 曝光次数 | 动态统计：各广告位累计曝光 |
| `adunit_req_series` | map\<int,map\<string,int\>\> | 广告位 → (小时 → 请求次数) | 时序统计：按小时粒度的请求分布 |
| `adunit_imp_series` | map\<int,map\<string,int\>\> | 广告位 → (小时 → 曝光次数) | 时序统计：按小时粒度的曝光分布 |
| `plat_rsp_7d` | map\<int,int\> | 分平台近 7d 响应次数 | 平台维度近期响应统计 |
| `day` | string | 日期分区 | 格式 `yyyy-MM-dd`，与 task_id 对应 |
| `no_rsp` | int | 零填充标记 | 值为 **1** 表示「过去三天零填充」——该用户近期无响应，特征被填充为 0，建模时需特殊处理 |
| `avg_bidprice` | map\<int,map\<int,double\>\> | 广告位 → (平台 → 平均出价) | 嵌套 map，连续统计特征 |

**特征分组（建模视角）：**

| 分组 | 字段 | 处理建议 |
|------|------|----------|
| 静态类别 | `make`, `model`, `province`, `city` | Label Encoding / Target Encoding / 与 seed 分布对比 |
| 多值离散 | `tags` | Multi-hot / 频次统计 / embedding mean pooling |
| 聚合计数 | `adunit_req_map`, `adunit_imp_map`, `plat_rsp_7d` | 展开 Top-K 广告位 / 总量 / CTR 派生 |
| 时序行为 | `adunit_req_series`, `adunit_imp_series` | 小时分布熵 / 峰值时段 / 序列统计量 |
| 出价连续 | `avg_bidprice` | 按广告位·平台展开均值 / max / 与 seed 差值 |
| 数据质量 | `no_rsp` | 作为缺失/冷启动指示特征，或单独建模 |

**官方说明（务必遵守）：**

1. **脱敏**：所有用户 ID 与特征均经过脱敏处理，不包含任何可还原真实身份的信息。

2. **种子剔除规则**：候选大盘中**包含种子用户**。当前窗口的种子用户可能在下一窗口继续产生高价值行为，但他们是业务**已知人群**，不属于模型需要「发现」的新增潜客。因此：
   - 训练标签应使用 `seed_{t+1} \ seed_t`（只标新增）
   - 推断 / 提交时应对 `seed_t` 用户降权或剔除
   - **最终评分时**，若提交的扩展人群包含当前窗口种子，平台会**过滤去除后再计算指标**

3. **无原始日志**：不直接开放原始曝光/点击日志，提供的是**聚合 + 脱敏后的 UV 级特征**；特征工程应围绕 map / array 结构做展开与统计，而非还原事件流。

#### （3）未来标签（Ground Truth）

> **下一时间窗口的种子人群名单** = 当前任务的未来高价值标签

即：`label(user) = 1` 当且仅当 `user ∈ seed_{T+1}` 且 `user ∉ seed_T`

#### （4）数据集划分与时间窗口

| 集合 | 内容 | 典型用途 |
|------|------|----------|
| **训练集** | 种子 + 大盘 + 未来标签 | 模型训练 |
| **开发集** | 种子 + 大盘 + 未来标签 | 本地验证 + **线上公开榜单评测** |
| **测试集** | 种子 + 大盘，**无未来标签** | 隐藏评测，决定最终排名 |

**已知日期（赛题公开）：**

| 任务日 task_id | 大盘数据 | 种子数据 | 预测目标 | 通常归属 |
|----------------|----------|----------|----------|----------|
| 2026-03-03 | pool_2026-03-03 | seed_2026-03-03 | 2026-03-04 新增高价值用户 | 训练集 |
| 2026-03-04 | pool_2026-03-04 | seed_2026-03-04 | 2026-03-05 新增高价值用户 | 开发集 |
| 2026-03-05 | pool_2026-03-05 | seed_2026-03-05 | **2026-03-06 新增高价值用户** | **测试集（提交日）** |

> 赛题数据页需登录报名后下载；本地已有提交样例见 `data/submit_sample.json`。

#### （5）开发集能否用于训练？

**结论：可以，赛题未禁止；但建议分阶段使用。**

赛题原文对三份数据的定义是：
- 训练集 → 「供选手**训练**模型」
- 开发集 → 「供选手**本地验证与线上榜单评测**」
- 测试集 → 无标签，隐藏评测

开发集**同样提供完整未来标签**，与训练集格式一致，赛题规则中**没有**「开发集不可参与训练」的限制。测试集才是唯一没有标签、不能用于监督学习的部分。

**推荐用法：**

| 阶段 | 数据用法 | 目的 |
|------|----------|------|
| **调参 / 迭代期** | 用 03-03 **训练**，03-04 **验证** | 本地 metric 与线上 Dev 榜单对齐，避免在测试日上过拟合 |
| **最终提交前** | 用 03-03 + 03-04 **合并训练**，03-05 **推断** | 充分利用有标签数据，提升对测试任务的泛化 |
| **不建议** | 用 03-05 的标签做任何训练或特征统计 | 测试集无公开标签；且易引入泄露 |

```
迭代期:  train(03-03) → valid(03-04) → 调特征/调参/看 Dev 榜
终版:    train(03-03 + 03-04) → predict(03-05) → 提交 Test 榜
```

**注意**：开发集同时承担「线上榜单」职能——你在 Dev 榜上的分数反映的是对 03-04→03-05 任务的预测能力；若全程只用 03-03 训练、从不碰 03-04 标签，Dev 榜也能正常评估，但会浪费一半有标签数据。

### 1.4 评审规则与评估指标

#### 核心思想

业务目标是：**在有限圈选规模 K 下，尽可能命中未来高价值用户**。  
因此主指标是 **Recall@K / Precision@K 类排序指标**，而非普通分类准确率。

#### 符号定义

- `U_t`：任务 t 的候选大盘（**剔除当前窗口种子**后的可圈选集合）
- `S_t^+`：任务 t 的未来高价值用户 = `seed_{t+1} \ seed_t`
- `TopK_t`：按选手 score 降序取前 K 个用户
- `|S_t^+|`：未来新增高价值用户总数

#### 指标 1：Recall@K（任务得分 1）

```
Recall@K(t) = |TopK_t ∩ S_t^+| / |S_t^+|
```

**业务预算档位 K**：10W、20W、50W、100W、200W  
**任务得分 1** = **K = 20W** 时的 Recall@20W

#### 指标 2：Precision@K → EfficiencyScore@K（任务得分 2）

```
Precision@K(t) = |TopK_t ∩ S_t^+| / K

随机基线 Precision_random@K = |S_t^+| / |U_t|

EfficiencyScore@K = min( Precision@K / (target × Precision_random@K), 1.0 )
```

- **K = 20W 时 target = 200**（即相对随机基线 200 倍命中效率为满分）
- **任务得分 2** = K = 20W 时的 EfficiencyScore@20W

#### 最终榜单得分

```
TaskScore(t) = 0.5 × Recall@20W(t) + 0.5 × EfficiencyScore@20W(t)

FinalScore = mean(TaskScore(t))  # 全部测试任务平均
```

**同分排序**：先比任务得分 2（EfficiencyScore），再比提交时间（早者优先）。

#### 指标优化直觉

| 指标 | 优化方向 |
|------|----------|
| Recall@20W | 把尽可能多的真实新增高价值用户排进 Top 20W |
| Efficiency@20W | 在 20W 规模内提高「命中率 / 随机命中率」，避免把分打给大量无关用户 |
| 两者权重各 50% | 不能只追 Recall（乱给高分）也不能只追 Precision（过于保守、Recall 低） |

### 1.5 提交要求

| 项 | 要求 |
|----|------|
| 格式 | UTF-8 JSON Array |
| 字段 | `task_id`, `masked_id`, `score` |
| 测试 task_id | `"2026-03-05"`（预测 2026-03-06 高价值用户） |
| score | 数值越大 = 越可能高价值；**只需相对排序正确，不必是概率** |
| 提交频率 | 建议每队每天 ≤ 3 次 |
| 代码 | 初赛仅交结果；Top3 需交可复现代码 |

**样例**（见 `data/submit_sample.json`）：

```json
[
  {"task_id": "2026-03-05", "masked_id": 10000002565, "score": 0.9},
  {"task_id": "2026-03-05", "masked_id": 10000007232, "score": 0.8}
]
```

### 1.6 赛程

| 节点 | 时间 |
|------|------|
| 数据发布 & 榜单开启 | 2026-06-18 10:00 |
| 提交截止 | 2026-08-27 17:00 |
| 名次公布 | 2026-09 中旬 |
| 决赛答辩 | Top3，作品 70% + 答辩 30% |

---

## 二、问题建模

### 2.1 与传统 Lookalike 的差异

| 维度 | 2018 腾讯 Lookalike 大赛 | 本赛题 Lookalike-AC |
|------|--------------------------|---------------------|
| 任务数 | 数百个种子包（多广告） | 按**日期窗口**的时序任务 |
| 标签 | 是否属于**当前**种子包 | 是否属于**下一窗口新增**高价值用户 |
| 输出 | 二分类概率 → 排序 | 直接排序分数 |
| 评测 | AUC（多 seed 平均） | **Recall@20W + Efficiency@20W** |
| 已知用户处理 | 种子即正样本 | **必须剔除当前种子**，只找新增 |

### 2.2 推荐建模框架

**主路径：Learning to Rank / 二分类打分 + 排序优化**

```
Step 1: 构造训练样本
  - 正样本: user ∈ seed_{t+1} \ seed_t
  - 负样本: user ∈ U_t \ seed_{t+1}（或 hard negative 采样）

Step 2: 特征工程
  - 原始 UV 特征
  - 种子对比特征（与 seed 分布的距离/相似度）
  - 跨窗口统计特征（严格用 t 及之前的数据）

Step 3: 训练目标
  - 基线: binary logloss / focal loss
  - 进阶: LambdaRank / 近似 Recall@K 的 listwise loss
  - 多任务: 03-03、03-04 联合训练，03-05 推断

Step 4: 推断
  - 对 03-05 全量候选（剔除 seed_03-05）输出 score
  - 可选: 多模型融合、校准、后处理截断
```

### 2.3 标签与样本构造（关键）

```python
# 伪代码 — 每个训练日 t
seed_t = load(f"seed_{t}")
seed_next = load(f"seed_{t+1}")
pool_t = load(f"pool_{t}")

positives = seed_next - seed_t          # 未来新增高价值
candidates = pool_t - seed_t            # 可圈选集合（剔除已知种子）

# 负样本策略（按数据规模选择）
negatives = random_sample(candidates - positives, ratio=1:10~1:50)
# 或 hard negative: 与 seed_t 特征相似但 label=0 的用户
```

**严禁数据泄露：**

- 不能用 `seed_{t+1}` 的任何信息做 t 日特征统计（除作 label）
- 不能用测试日 03-06 的任何信息
- 跨日统计特征只能使用 `≤ t` 的窗口

---

## 三、相关领域调研

### 3.1 业务与算法脉络

1. **Lookalike / Audience Expansion**  
   广告行业标准能力：从种子包出发，在海量用户中扩展相似人群（Facebook Lookalike、Google Similar Audiences、腾讯 / 字节 / 讯飞营销云）。

2. **与 CTR / CVR 预估的关系**  
   技术栈高度重叠（特征工程 + GBDT / 深度交叉模型），但优化目标不同：本赛题是 **Top-K 召回 + 命中效率**，不是单点 AUC。

3. **学术视角**  
   - 经典做法：种子为正样本 + 随机负样本 → 二分类 → 按概率排序取 Top-K  
   - 论文 *Reframing Audience Expansion through the Lens of Probability Density Estimation* (arXiv:2311.05853)：指出传统二元分类存在训练样本选择偏差，提出基于密度估计的扩展框架；评测同样使用 **P@K / R@K**  
   - 工业界趋势：预训练用户表征（序列 / 图） + 任务微调

### 3.2 类似 Kaggle / 工业竞赛经验

| 竞赛 | 相似点 | 可借鉴 |
|------|--------|--------|
| **2018 腾讯广告 Lookalike** | 种子扩展、UV 特征、AUC 评测 | 统计特征、NFFM/XDeepFM、多模型融合 |
| **阿里 Mama CTR/CTCVR** | 广告表格数据、ID 统计 | 分桶统计、交叉特征、ESMM 思路 |
| **Avazu CTR** | 大规模稀疏离散特征 | 哈希编码、FFM/FM |
| **Otto Group Product Classification** | 多分类 + 排序 | 特征聚合思路 |

> 本赛题**没有直接对应的 Kaggle 赛题**（评测是 Recall@20W + Efficiency@20W），但 **2018 腾讯 Lookalike 是最近邻参考**。

### 3.3 GitHub 开源方案参考

| 仓库 | 成绩 | 核心方法 |
|------|------|----------|
| [guoday/Tencent2018_Lookalike_Rank7th](https://github.com/guoday/Tencent2018_Lookalike_Rank7th) | 复赛 Top7 | **NFFM + XDeepFM** 深度交叉，embedding 16 维 |
| [keyunluo/Tencent2018_Lookalike_Rank10th](https://github.com/keyunluo/Tencent2018_Lookalike_Rank10th) | Top10 | 基于 NFFM 的深度方案 |
| [ShawnyXiao/2018-Tencent-Lookalike](https://github.com/ShawnyXiao/2018-Tencent-Lookalike) | 初赛 Top10 | **纯 LightGBM** |
| [BladeCoda/Tencent2018_Final_Phrase_Presto](https://github.com/BladeCoda/Tencent2018_Final_Phrase_Presto) | 决赛 0.753 | **双 LGB 融合**（统计特征 + OneHot 统计） |
| [Tigeryang93/data-competition-topsolution](https://github.com/Tigeryang93/data-competition-topsolution) | 汇总 | 腾讯 2018 各 Rank 代码索引 |
| [DiligentPanda/Tencent_Ads_Algo_2018](https://github.com/DiligentPanda/Tencent_Ads_Algo_2018) | Top3 | 完整特征 + 模型 pipeline |

**腾讯 2018 Top 方案共性：**

1. **特征 > 模型**：ID 统计（click / ratio / cvr）、交叉统计、多值字段 sparse 编码
2. **分块 / 交叉验证**：防止统计特征过拟合
3. **多模型融合**：LGB + FFM/NFFM + 线性加权或 XGB 二层融合
4. **负采样**：控制正负比例，避免样本不平衡

---

## 四、完整解题方案（Vibe Coding 路线图）

### 4.1 项目目录结构（建议）

```
xf_lookalike/
├── data/
│   ├── raw/              # 官方下载原始 parquet
│   ├── processed/        # 清洗后中间表
│   └── submit_sample.json
├── configs/
│   └── default.yaml      # 路径、K 值、采样比例、模型参数
├── src/
│   ├── data/
│   │   ├── load.py       # 读取 seed / pool parquet
│   │   ├── label.py      # 构造 pos/neg 标签
│   │   └── split.py      # 时间窗口切分
│   ├── features/
│   │   ├── base.py       # 原始特征处理
│   │   ├── seed_sim.py   # 种子对比特征（均值距离、余弦等）
│   │   └── stats.py      # 统计交叉特征
│   ├── models/
│   │   ├── lgb.py
│   │   ├── ranker.py     # LGB LambdaRank 可选
│   │   └── ensemble.py
│   ├── metrics/
│   │   └── eval.py       # Recall@K, Efficiency@K 本地复现
│   ├── infer.py          # 生成提交 JSON
│   └── train.py          # 训练入口
├── notebooks/
│   └── eda.ipynb         # 数据探索
├── outputs/
│   ├── models/
│   └── submissions/
├── PLAN.md               # 本文档
└── README.md
```

### 4.2 分阶段实施计划

#### Phase 0：环境与 EDA（Day 1）

- [ ] 报名下载数据，确认 parquet schema（列名、类型、缺失率）
- [ ] 统计每日：`pool` 用户数、seed 数、新增 pos 数、pos 占比
- [ ] 确认 `masked_id` 在 seed 与 pool 的对齐率
- [ ] 可视化：pos/neg 特征分布差异、Top 特征 IV

#### Phase 1：Baseline（Day 2-3）— 目标：跑通全流程

**模型**：LightGBM 二分类  
**特征**：`make/model/province/city` 直接编码；`tags` 计数/多热；map 类字段先取 **总量 / Top-K 键** 作为 baseline（详见 §1.3 字段表）  
**样本**：

- **迭代期** Train: 2026-03-03 → label from `seed_03-04 \ seed_03-03`
- **迭代期** Valid: 2026-03-04 → label from `seed_03-05 \ seed_03-04`（对齐 Dev 榜）
- **终版提交** Train: 03-03 + 03-04 合并 → Predict: 2026-03-05 全量候选（剔除 `seed_03-05`）

**本地评测**：实现 `Recall@20W` + `Efficiency@20W`，与榜单对齐逻辑

```python
# 本地指标伪代码
def evaluate(scores, labels, candidates, K=200_000, target=200):
    order = np.argsort(-scores)
    topk = candidates[order[:K]]
    pos_set = set(labels[labels==1].index)
    recall = len(set(topk) & pos_set) / len(pos_set)
    prec = len(set(topk) & pos_set) / K
    prec_random = len(pos_set) / len(candidates)
    efficiency = min(prec / (target * prec_random), 1.0)
    return 0.5 * recall + 0.5 * efficiency
```

#### Phase 2：特征增强（Day 4-7）— 目标：显著提升 Recall

1. **种子对比特征**（本赛题特有，高优先级）
   - 数值特征：`|x - mean_seed(x)|`、`x / (mean_seed(x)+ε)`
   - 类别特征：与 seed 众数/分布的 JS 散度、共现比例
   - 嵌入：对 seed 用户特征做 mean pooling 得到 seed profile，算 cosine similarity

2. **统计交叉特征**（借鉴腾讯 2018）
   - 类别字段 count / ratio / 转化率式统计
   - 两两交叉统计（控制组合爆炸，Top 频次 only）

3. **时序特征**（若有跨日 pool）
   - 用户在 t-1 日是否出现、特征变化量
   - 严格避免用 t+1 信息

4. **负采样优化**
   - 随机负采样 + **hard negative**（score 高但 label=0）
   - 按 `|U|` 规模调整 neg:pos ≈ 20:1 ~ 100:1

#### Phase 3：模型升级（Day 8-12）

| 优先级 | 方案 | 说明 |
|--------|------|------|
| P0 | LightGBM + 调参 | num_leaves, min_data_in_leaf, feature_fraction, scale_pos_weight |
| P1 | LightGBM LambdaRank | 直接优化排序；group = 每个 task 的全部候选 |
| P2 | 双塔 / 向量召回 | seed encoder + user encoder，cosine 作 score（适合海量） |
| P3 | NFFM / DeepFM | 若离散特征多、数据量可上 GPU |
| P4 | 多模型融合 | LGB + 深度模型加权；用 valid 日 grid search 权重 |

**LambdaRank 提示**：query = 每个 task 的 candidate set，label = 0/1，metric 近似 NDCG，与 Recall@K 方向一致。

#### Phase 4：对齐评测的优化（Day 13-15）

1. **K=20W 专项调优**
   - 在 valid 上画 Recall@K、Efficiency@K 随 K 曲线
   - 若 Recall 高但 Efficiency 低 → 减少低分区间噪声（提高 score 区分度、温度缩放）
   - 若 Efficiency 高但 Recall 低 → 放松负采样、增加模型容量

2. **Score 后处理**
   - Rank average：多折 / 多模型排名平均（比概率平均更稳）
   - 不要对 seed 用户打分（或强制 score=-inf，虽然平台会过滤）

3. **Pseudo-label（可选）**
   - 用 03-05 高置信预测做半监督（谨慎，易过拟合）

#### Phase 5：提交与迭代

- 每天 3 次提交：分别测试 **LGB / 融合 / 后处理** 版本
- 记录：提交版本 ↔ 本地 valid 分数 ↔ 线上分数，校准本地 metric
- 截止前留 1 次 buffer 给最优融合

### 4.3 推荐默认超参（LightGBM Baseline）

```yaml
lgb:
  objective: binary
  metric: auc
  learning_rate: 0.05
  num_leaves: 64
  max_depth: 8
  min_data_in_leaf: 200
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  scale_pos_weight: auto  # len(neg)/len(pos)
  n_estimators: 2000
  early_stopping_rounds: 100
```

### 4.4 风险与注意事项

| 风险 | 应对 |
|------|------|
| 候选集千万级，内存爆炸 | 分块读取 parquet；训练时负采样；推断分批 predict |
| 正负极度不平衡（pos 可能 <0.1%） | 负采样 + scale_pos_weight + 关注 Recall@K 而非 AUC |
| 本地 metric 与线上不一致 | 严格复现「剔除 seed」「K=20W」「target=200」逻辑 |
| 过拟合 2 天训练数据 | 交叉验证 + 简单模型优先 + 特征正则 |
| 赛题数据页 schema 未公开 | 下载后第一时间更新 `PLAN.md` 字段表 |

---

## 五、Vibe Coding 任务清单（可直接喂给 AI）

按顺序复制以下 prompt 块驱动开发：

### Task 1 — 数据加载
```
实现 src/data/load.py：读取 seed_*.parquet 和 pool_*.parquet，
支持按 task_id 过滤，返回 DataFrame。
```

### Task 2 — 标签构造
```
实现 src/data/label.py：
给定 seed_t 和 seed_{t+1}，构造 positives = seed_{t+1} \ seed_t，
candidates = pool_t \ seed_t，生成 masked_id + label 表。
```

### Task 3 — 本地评测
```
实现 src/metrics/eval.py：Recall@20W、Efficiency@20W（target=200）、
FinalScore = 0.5*R + 0.5*E，输入 score 数组和 label。
```

### Task 4 — LGB Baseline
```
实现 src/train.py + src/models/lgb.py + src/infer.py：
迭代期用 03-03 训练、03-04 验证；终版用 03-03+03-04 合并训练，
对 03-05 全量推断，输出 submit JSON。
```

### Task 5 — 种子对比特征
```
实现 src/features/seed_sim.py：
对每个候选用户，计算与 seed_t 特征分布的相似度特征，join 到训练表。
```

### Task 6 — 融合与提交
```
实现 src/models/ensemble.py：多模型 rank average，
输出最终 submission 文件。
```

---

## 六、成功标准（Definition of Done）

| 阶段 | 标准 |
|------|------|
| MVP | 本地跑通 train → infer → json，成功提交一版 |
| Baseline | 本地 FinalScore 可计算，valid 分数可复现 |
|  Competitive | Recall@20W 和 Efficiency@20W 均优于随机基线 10× 以上 |
|  Top 梯队 | 特征 + 融合 + K 对齐优化，线上进入榜单前列 |

---

## 七、参考链接

- [赛题主页](https://challenge.xfyun.cn/topic/info?type=Lookalike-AC)
- [腾讯 Lookalike 大赛解析](https://blog.csdn.net/anyi6536/article/details/102253446)
- [2018 腾讯广告算法大赛 Rank11 总结](https://cloud.tencent.com/developer/article/1505673)
- [Audience Expansion 论文 arXiv:2311.05853](https://arxiv.org/abs/2311.05853)
- [GitHub 方案汇总 Tigeryang93/data-competition-topsolution](https://github.com/Tigeryang93/data-competition-topsolution)

---

*文档版本：v1.1 | 补充候选大盘字段明细、官方说明、开发集训练策略。*
