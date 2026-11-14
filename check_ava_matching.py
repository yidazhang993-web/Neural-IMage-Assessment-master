
import pandas as pd
import numpy as np

# ==========================
# 1. 读取 NIMA 预测结果
# ==========================

pred_path = "data/ava_test_predictions.csv"
label_path = "data/test_labels.csv"

pred = pd.read_csv(pred_path)
labels = pd.read_csv(label_path, header=None)

print("========== 1. 文件基本信息 ==========")
print("预测结果形状：", pred.shape)
print("人工标注形状：", labels.shape)

print("\n人工标注前5行：")
print(labels.head())

# ==========================
# 2. 检查预测概率分布
# ==========================

prob_cols = [f"pred_vote_{i}" for i in range(1, 11)]

probs = pred[prob_cols].to_numpy(dtype=float)
prob_sums = probs.sum(axis=1)

print("\n========== 2. 概率分布检查 ==========")
print("概率最小值：", probs.min())
print("概率最大值：", probs.max())
print("概率和统计：")
print(pd.Series(prob_sums).describe())

print("概率和与1的最大误差：",
      np.max(np.abs(prob_sums - 1)))

print("概率非负检查：", np.all(probs >= 0))

# ==========================
# 3. 检查人工标注结构
# ==========================

print("\n========== 3. 人工标注字段 ==========")
print("前5行：")
print(labels.iloc[:5].to_string(index=False))

print("数据类型：")
print(labels.dtypes)

# ==========================
# 4. 检查图片编号匹配
# ==========================

if labels.shape[1] < 1:
    raise ValueError("人工标注文件没有有效列")

pred_ids = pd.to_numeric(
    pred["image_num"], errors="coerce"
)

label_ids = pd.to_numeric(
    labels.iloc[:, 0], errors="coerce"
)

print("\n========== 4. 图片编号检查 ==========")
print("预测编号无法转换数量：", pred_ids.isna().sum())
print("标注编号无法转换数量：", label_ids.isna().sum())

if pred_ids.isna().any() or label_ids.isna().any():
    raise ValueError("图片编号存在无法转换的值")

pred_ids = pred_ids.astype("int64")
label_ids = label_ids.astype("int64")

print("预测编号重复数量：", pred_ids.duplicated().sum())
print("标注编号重复数量：", label_ids.duplicated().sum())

pred_set = set(pred_ids)
label_set = set(label_ids)

print("预测编号数量：", len(pred_set))
print("标注编号数量：", len(label_set))
print("共同编号数量：", len(pred_set & label_set))
print("仅预测结果存在：", len(pred_set - label_set))
print("仅人工标注存在：", len(label_set - pred_set))

print("\n========== 5. 检查是否可以一一匹配 ==========")

if pred_set == label_set:
    print("通过：两份数据的图片编号完全一致")
else:
    print("未通过：两份数据的图片编号不完全一致")

# ==========================
# 6. 检查标注数据是否为11列
# ==========================

print("\n========== 6. 人工评分分布结构 ==========")

if labels.shape[1] == 11:
    votes = labels.iloc[:, 1:].apply(
        pd.to_numeric, errors="coerce"
    )

    print("检测到：图片编号 + 10维人工评分")
    print("人工评分是否存在缺失：", votes.isna().any().any())
    print("人工评分最小值：", votes.min().min())
    print("人工评分最大值：", votes.max().max())

    print("人工评分总票数统计：")
    print(votes.sum(axis=1).describe())

else:
    print("注意：人工标注不是11列，暂不假定其数据结构")
