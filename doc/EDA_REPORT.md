# Lookalike-AC 数据分析报告 (EDA)

> 生成时间：2026-06-26
> 数据范围：train / dev / test 三个时间窗口
> 环境：conda env `ml` (Python 3.12.9)

## 一、数据规模总览

| 划分 | task_id | 候选大盘 UV | 当前种子 | 下一窗口新增正样本 | 正样本率 (vs 大盘) |
|------|---------|------------|----------|-------------------|-------------------|
| train | 2026-03-03 | 50,935,335 | 20,122 | 9,208 | 0.01808% |
| dev | 2026-03-04 | 53,698,743 | 15,693 | 10,651 | 0.01984% |
| test | 2026-03-05 | 53,574,838 | 18,219 | — (无标签) | — |

**关键观察：**

1. **极不平衡**：正样本率约 **0.018%**，即每 ~5,500 个候选用户里才有 1 个未来新增高价值用户。
   - 全量训练需 **负采样**（baseline 配置 neg_ratio=20，即 pos:neg=1:20）。
   - 不能用 accuracy/AUC 为主，需直接看 **Recall@20W + Efficiency@20W**。

2. **种子规模稳定**：每日种子 1.5w~2w，新增正样本 ~1w 量级。
   - Top-K 圈选规模 K=20W 远大于正样本数（~1w），意味着 **Recall@20W 理论上限就是 100%**（只要把全部正样本排进前 20W）。
   - 真正的区分度在 **Efficiency@20W**：20W 槽位里只命中 ~1w 正样本，命中率约 5%，相对随机基线 (1w/5000w ≈ 0.02%) 提升约 **250 倍**，已超过 target=200 的满分线。

3. **三个窗口大盘规模一致**（~5×10⁷），可认为分布稳定，跨日特征统计相对安全。

## 二、Schema 与字段类型

候选大盘（`final_*.parquet`，zstd 压缩，每个分片 ~290MB，10 个分片）：

| 字段 | 类型 | 说明 | baseline 处理 |
|------|------|------|--------------|
| `masked_id` | int64 | 用户唯一标识 | 主键，与 seed join |
| `make` | string | 设备厂商 | 类别编码 |
| `model` | string | 机型 | 类别编码 |
| `province` | string | 省份 | 类别编码 |
| `city` | string | 城市 | 类别编码 |
| `tags` | list\<int32\> | 标签数组 | → `tags_len` |
| `adunit_req_map` | map\<int,int\> | 广告位→请求次数 | → sum / len / max |
| `adunit_imp_map` | map\<int,int\> | 广告位→曝光次数 | → sum / len / max |
| `adunit_req_series` | map\<int,map\<str,int\>\> | 广告位→小时→请求 | → 嵌套 sum / len |
| `adunit_imp_series` | map\<int,map\<str,int\>\> | 广告位→小时→曝光 | → 嵌套 sum / len |
| `plat_rsp_7d` | map\<int,int\> | 平台→近7d响应 | → sum / len / max |
| `avg_bidprice` | map\<int,map\<int,double\>\> | 广告位→平台→均价 | → 嵌套 sum / len |
| `day` | string | 日期分区 yyyy-MM-dd | 与 task_id 一致 |
| `no_rsp` | int32 | 零填充标记(1=过去三天无响应) | 直接作特征 |

种子表（`seed_*.parquet`，snappy 压缩，单文件）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `masked_id` | int64 | 用户唯一标识 |
| `plat` | string | 平台名称 |

> 种子表 **不含特征**，需回大盘 join 取特征。文件名后缀含义：
> - `current-seed-feature-hit100`：当前窗口种子
> - `evaluation-seed-in-current-candidates`：下一窗口种子（用作 label）

## 三、标签与样本构造

```
正样本 (positive) = seed_{t+1} \ seed_t   # 未来新增高价值用户
候选集 (candidate) = pool_t \ seed_t        # 剔除已知种子后可圈选集合
负样本 (negative)  = candidate \ positive   # 大盘中非新增用户
```

**已验证**：`seed_{t+1} ∩ seed_t = ∅`（相邻窗口种子无重叠，即下一窗口种子全是"新增"）。

| 划分 | candidate 规模 | positive | positive/candidate |
|------|---------------|----------|--------------------|
| train 03-03 | ≈ 50.9M - 20k | 9,208 | ≈ 0.0181% |
| dev 03-04 | ≈ 53.7M - 15.7k | 10,651 | ≈ 0.0198% |

## 四、特征实例观察（取自 train 第一分片）

```
masked_id        10000002541            10000005511
make             Apple                  Apple
model            IPHONE18,2             IPHONE16,1
province         广东                    广东
city             汕尾市                  广州市
tags             None                   [98755, 102169, 102451, 102459]
adunit_req_map   [(80323, 2)]           [(82027, 2), (82259, 9), (82859, 2)]
adunit_imp_map   None                   None
adunit_req_series[(80323, [('17', 2)])] [(82027, [('10',1),('13',1)]), ...]
adunit_imp_series None                  None
plat_rsp_7d      [(306, 2)]             [(202,2),(269,1),(283,36),(306,76)]
avg_bidprice     [(80323, [(306,14.22)])] [(82027,[(202,0.79)]), ...]
day              2026-03-03             2026-03-03
no_rsp           1                      1
```

**观察：**

1. **大量字段为 None / 空 map**（如 `adunit_imp_map`、`adunit_imp_series` 多为空）→ 说明很多用户**有请求但无曝光**，冷启动/低活用户占比高。`no_rsp=1` 也佐证。
2. **`tags` 经常为 None** → `tags_len=0` 是高频值，需作为"无标签用户"信号。
3. **广告位 ID、平台 ID 是高频离散键** → 后续可做 Top-K 频次统计 / 共现特征。
4. **`avg_bidprice` 嵌套两层 map** → baseline 只取 sum/len，进阶可按平台展开均值/max。
5. **机型/省份/城市** 中文在 Windows 控制台显示乱码，但 parquet 存储正常（UTF-8），编码时按 string 处理即可。

## 五、评测指标与本数据的关系

主指标（K=20W，target=200）：

```
Recall@20W      = |Top20W ∩ positive| / |positive|
Efficiency@20W  = min( Precision@20W / (200 × |positive|/|candidate|), 1 )
FinalScore      = 0.5 × Recall@20W + 0.5 × Efficiency@20W
```

代入 dev 03-04 估算：

- `|positive|` ≈ 10,651，`|candidate|` ≈ 53.7M
- 随机基线 Precision_random = 10,651 / 53,700,000 ≈ **0.000198**
- 若模型把全部 10,651 正样本排进 Top 20W：
  - Recall@20W = 100%
  - Precision@20W = 10,651 / 200,000 ≈ 5.33%
  - Efficiency = 0.0533 / (200 × 0.000198) = 0.0533 / 0.0396 ≈ **1.0**（满分）
  - FinalScore ≈ 1.0

**结论**：在当前数据规模下，**Recall@20W 和 Efficiency@20W 方向高度一致**——只要把 ~1w 正样本尽量排进前 20W，两项都能拿高分。真正瓶颈是 **排序质量**（能否把正样本排到前面），而非 K 截断。

## 六、对建模的指导

1. **优先保证正样本被排到 Top**：用 AUC/排序 loss 调参，关注 Recall@20W 而非 logloss。
2. **负采样比例可调**：pos:neg = 1:20 起步；若 Efficiency 偏低（Top20W 噪声多），可降到 1:10；若 Recall 偏低（漏正样本），可升到 1:50 + hard negative。
3. **冷启动用户**（`no_rsp=1`、空 map）占比高，需单独建模或作指示特征，避免被误判为高分。
4. **种子对比特征是高价值方向**（PLAN §4.2 Phase 2）：候选与 seed_t 特征分布的距离/相似度，是 Lookalike 的核心信号，baseline 暂未加入。
5. **内存策略**：每池 5×10⁷ × 14 列（含 map），全量加载需分片；baseline 已用 `sample_frac=0.05` 快速迭代，验证流程后再全量。

## 七、下一步

- [x] EDA 完成，schema/规模/标签率已确认
- [ ] baseline 训练（进行中）
- [ ] baseline 推断 + 提交首版
- [ ] 加入种子对比特征（Phase 2）
- [ ] 调整 neg_ratio / K 对齐优化（Phase 4）
