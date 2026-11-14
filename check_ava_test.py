
import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "data")
image_dir = os.path.join(data_dir, "images")

# 读取标注
ground = pd.read_csv(
    os.path.join(data_dir, "ground_truth_dataset.csv")
)

train = pd.read_csv(
    os.path.join(data_dir, "train_labels.csv"),
    header=None
)

val = pd.read_csv(
    os.path.join(data_dir, "val_labels.csv"),
    header=None
)

# 获取图片编号
all_ids = set(ground["image_num"].astype(int))
train_ids = set(train.iloc[:, 0].astype(int))
val_ids = set(val.iloc[:, 0].astype(int))

test_ids = all_ids - train_ids - val_ids

# 检查所有候选测试图片
extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

missing = []

for image_id in sorted(test_ids):
    found = any(
        os.path.isfile(
            os.path.join(image_dir, str(image_id) + ext)
        )
        for ext in extensions
    )

    if not found:
        missing.append(image_id)

print("候选测试集数量：", len(test_ids))
print("缺失图片数量：", len(missing))
print("缺失图片编号示例：", missing[:30])

# 查看 AVA.txt 前5行
ava_txt = os.path.join(data_dir, "AVA_Files", "AVA.txt")

print("\n========== AVA.txt 内容 ==========")

if os.path.exists(ava_txt):
    with open(ava_txt, "r", encoding="utf-8") as f:
        for i in range(5):
            line = f.readline()
            if not line:
                break
            print(line.strip())
else:
    print("没有找到 AVA.txt")

