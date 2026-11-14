
import pandas as pd
import numpy as np

# 读取预测结果
path = "data/ava_test_predictions.csv"
df = pd.read_csv(path)

print("========== 1. 基本信息 ==========")
print("数据形状：", df.shape)
print("列名：", df.columns.tolist())

print("\n========== 2. 前5行 ==========")
print(df.head())

print("\n========== 3. 缺失值检查 ==========")
print(df.isnull().sum())

print("\n========== 4. 重复图片检查 ==========")
if "image_num" in df.columns:
    print("重复图片编号：", df["image_num"].duplicated().sum())
elif "image_id" in df.columns:
    print("重复图片编号：", df["image_id"].duplicated().sum())
else:
    print("暂未识别图片编号列")

print("\n========== 5. 预测评分检查 ==========")
print(df["pred_score"].describe())

print("\n========== 6. 预测评分异常检查 ==========")
print("小于1的数量：", (df["pred_score"] < 1).sum())
print("大于10的数量：", (df["pred_score"] > 10).sum())

# 检查是否存在概率分布列
prob_cols = [
    str(i) for i in range(1, 11)
]

if all(col in df.columns for col in prob_cols):
    probs = df[prob_cols].to_numpy()

    print("\n========== 7. 概率分布检查 ==========")
    print("概率最小值：", probs.min())
    print("概率最大值：", probs.max())
    print("概率和统计：")
    print(pd.Series(probs.sum(axis=1)).describe())

    print("概率小于0的数量：", (probs < 0).sum())
    print("概率大于1的数量：", (probs > 1).sum())

else:
    print("\n未找到名称为1～10的概率列")
    print("实际列名：", df.columns.tolist())

