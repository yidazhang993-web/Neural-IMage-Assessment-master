
import os
import pandas as pd

# 获取项目根目录
root = os.path.dirname(os.path.abspath(__file__))

# AVA 数据目录
data_dir = os.path.join(root, "data")

csv_files = [
    "ground_truth_dataset.csv",
    "train_labels.csv",
    "val_labels.csv",
]

print("=" * 60)
print("AVA 数据集 CSV 文件检查")
print("=" * 60)

# 检查 CSV 文件
for filename in csv_files:
    path = os.path.join(data_dir, filename)

    print(f"\n文件：{filename}")

    if not os.path.exists(path):
        print("文件不存在：", path)
        continue

    df = pd.read_csv(path)

    print("数据行数：", len(df))
    print("字段名称：", df.columns.tolist())

    print("\n前 3 行数据：")
    print(df.head(3).to_string(index=False))

# 检查图片目录
image_dir = os.path.join(data_dir, "images")

print("\n" + "=" * 60)
print("AVA 图片检查")
print("=" * 60)

if os.path.isdir(image_dir):
    image_count = 0

    for current_root, dirs, files in os.walk(image_dir):
        image_count += sum(
            1 for name in files
            if name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        )

    print("图片目录：", image_dir)
    print("图片总数：", image_count)
else:
    print("图片文件夹不存在：", image_dir)

print("\n检查完成！")
