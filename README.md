# Lookalike-AC

讯飞 AI 开发者大赛 — 高价值人群扩展（Lookalike-AC）解题工程。

## 环境

```powershell
conda activate ml
pip install -r requirements.txt
```

## 目录结构

```
xf_lookalike/
├── configs/          # 配置文件
├── data/             # 官方数据（train/dev/test）
├── doc/              # 赛题方案文档
├── notebooks/        # 探索性分析
├── outputs/          # 模型与提交产物
├── scripts/          # 可执行入口脚本
├── src/              # 核心库代码
│   ├── data/         # 数据加载与标签
│   ├── features/     # 特征工程
│   ├── metrics/      # 评测指标
│   ├── models/       # 模型训练
│   └── utils/        # 工具函数
└── requirements.txt
```

## 快速开始

```powershell
# 1. 数据探索
E:\miniconda\envs\ml\python.exe scripts/eda.py

# 2. 训练 baseline（默认 5% 采样，快速迭代）
E:\miniconda\envs\ml\python.exe scripts/train.py

# 3. 生成测试集提交文件
E:\miniconda\envs\ml\python.exe scripts/infer.py
```

## 说明

- 详细赛题背景与方案见 `doc/PLAN.md`
- baseline 使用 LightGBM + 扁平化 map 特征，适合先跑通流程
- 全量训练可将 `configs/default.yaml` 中 `train.sample_frac` 调为 `1.0`
