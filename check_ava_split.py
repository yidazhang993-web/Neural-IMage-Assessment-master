
import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "data")

# 1. 读取完整人工评分标注
ground = pd.read_csv(
    os.path.join(data_dir, "ground_truth_dataset.csv")
)

# 2. 训练集和验证集没有表头，必须 header=None
train = pd.read_csv(
    os.path.join(data_dir, "train_labels.csv"),
    header=None
)

val = pd.read_csv(
    os.path.join(data_dir, "val_labels.csv"),
    header=None
)

# 3. 检查字段数量
print("训练集字段数：", train.shape[1])
print("验证集字段数：", val.shape[1])

if train.shape[1] != 11 or val.shape[1] != 11:
    raise ValueError("训练集或验证集字段数量不是11，请检查文件格式")

# 4. 设置统一字段名称
columns = [
    "image_num",
    "vote_1", "vote_2", "vote_3", "vote_4", "vote_5",
    "vote_6", "vote_7", "vote_8", "vote_9", "vote_10"
]

train.columns = columns
val.columns = columns

# 5. 统一图片编号类型
ground["image_num"] = ground["image_num"].astype(int)
train["image_num"] = train["image_num"].astype(int)
val["image_num"] = val["image_num"].astype(int)

# 6. 转换为集合
all_ids = set(ground["image_num"])
train_ids = set(train["image_num"])
val_ids = set(val["image_num"])

print("\n========== 数据集数量 ==========")
print("全部数据：", len(ground))
print("训练集：", len(train))
print("验证集：", len(val))

# 7. 检查重复编号
print("\n========== 重复检查 ==========")
print("完整数据重复编号：",
      ground["image_num"].duplicated().sum())
print("训练集重复编号：",
      train["image_num"].duplicated().sum())
print("验证集重复编号：",
      val["image_num"].duplicated().sum())

# 8. 检查集合交叉
print("\n========== 集合交叉检查 ==========")
print("训练集与验证集交叉：", len(train_ids & val_ids))
print("训练集不在完整数据中的数量：",
      len(train_ids - all_ids))
print("验证集不在完整数据中的数量：",
      len(val_ids - all_ids))

# 9. 找出候选测试集
test_ids = all_ids - train_ids - val_ids

print("\n========== 候选测试集 ==========")
print("候选测试集图片数量：", len(test_ids))

# 10. 检查图片目录及文件名
image_dir = os.path.join(data_dir, "images")

files = os.listdir(image_dir)
print("\n图片目录文件示例：", files[:10])

# 11. 检查候选测试集对应图片是否存在
extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

missing = []

for image_id in list(test_ids)[:1000]:
    found = any(
        os.path.isfile(
            os.path.join(image_dir, str(image_id) + ext)
        )
        for ext in extensions
    )
    if not found:
        missing.append(image_id)

print("\n前1000个候选测试编号中，找不到对应图片的数量：",
      len(missing))

print("\n检查完成！")
