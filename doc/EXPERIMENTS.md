# 实验记录 (Experiments Log)

> 每次改动在此追加一行，记录配置、本地 Dev 分数、线上 Test 分数。

| ID | 日期 | 改动摘要 | sample_frac | neg_ratio | 特征 | 本地 Dev FinalScore | 线上 Test FinalScore | 备注 |
|----|------|----------|-------------|-----------|------|---------------------|----------------------|------|
| v0-baseline | 2026-06-26 | 初版 LGB + map 扁平化特征；5% 采样训练；Top-20W 提交 | 0.05 | 20 | flatten only | 0.3580 (5% dev, 544 pos) | **0.12385** | 本地 dev 为采样评测，与线上偏差大 |
| v1-stream-full | 2026-06-26 | 流式全量扫描 train；正样本全保留 + 负样本 reservoir 采样；流式 dev 评测 | 1.0 | 20 | flatten only | _待跑_ | _待提交_ | 修复 OOM，支持全量 |

## 详细说明

### v0-baseline

- **代码**: `scripts/train.py` + `src/features/flatten.py`
- **训练**: train 03-03, 5% 采样 → 476 pos / 9996 rows
- **验证**: dev 03-04, 5% 采样 → 544 pos
- **推断**: Top-200W heap, 13 min
- **问题**: 采样导致正样本过少；dev 本地评测不代表全量

### v1-stream-full

- **改动**:
  - `src/data/streaming.py`: 流式构造训练集，不 concat 全量大盘
  - `src/metrics/streaming_eval.py`: 流式 dev 打分 + Top-K 指标
  - `scripts/train.py`: 改用流式 pipeline
- **预期**: 正样本 9208 全量；LGB 训练 ~19.3 万行；内存 <30GB

---

## 如何追加记录

每次实验完成后，在表格顶部（ID 递增）追加一行，并在「详细说明」追加小节。

```markdown
| v2-seed-sim | YYYY-MM-DD | 加入种子对比特征 | 1.0 | 20 | flatten+seed_sim | x.xxxx | x.xxxxx | ... |
```
