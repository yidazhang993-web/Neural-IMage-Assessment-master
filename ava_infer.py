
import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from model.model import NIMA


# =========================
# 1. 路径配置
# =========================

ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(ROOT, "data", "images")
LABEL_PATH = os.path.join(ROOT, "data", "test_labels.csv")
WEIGHT_PATH = os.path.join(ROOT, "weights", "epoch-82.pth")

OUTPUT_PATH = os.path.join(ROOT, "data", "ava_test_predictions.csv")


# =========================
# 2. 参数配置
# =========================

BATCH_SIZE = 32
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)


# =========================
# 3. 图片预处理
# =========================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. 自定义数据集
# =========================

class AVADataset(Dataset):

    def __init__(self, image_ids, image_dir, transform=None):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform

        self.extensions = [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]

        self.image_paths = []

        for image_id in self.image_ids:

            image_path = None

            for ext in self.extensions:
                candidate = os.path.join(
                    self.image_dir,
                    str(image_id) + ext
                )

                if os.path.isfile(candidate):
                    image_path = candidate
                    break

            if image_path is None:
                raise FileNotFoundError(
                    f"找不到图片：{image_id}"
                )

            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_id


# =========================
# 5. 读取测试集标注
# =========================

labels = pd.read_csv(
    LABEL_PATH,
    header=None
)

if labels.shape[1] != 11:
    raise ValueError("测试集标注文件应该有11列")

image_ids = labels.iloc[:, 0].astype(int).tolist()

print("测试集图片数量：", len(image_ids))


# =========================
# 6. 创建 DataLoader
# =========================

dataset = AVADataset(
    image_ids=image_ids,
    image_dir=IMAGE_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("图片路径检查完成")


# =========================
# 7. 加载 NIMA 模型
# =========================

base_model = models.vgg16(weights=None)

model = NIMA(base_model)

state_dict = torch.load(
    WEIGHT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)

model = model.to(device)
model.eval()

print("NIMA 模型加载成功")


# =========================
# 8. 批量推理
# =========================

results = []

with torch.inference_mode():

    for images, ids in tqdm(loader, desc="AVA 推理"):

        images = images.to(device)

        outputs = model(images)

        probabilities = outputs.cpu().numpy()

        for image_id, probs in zip(ids, probabilities):

            scores = np.arange(1, 11)

            mean_score = np.sum(scores * probs)

            std_score = np.sqrt(
                np.sum(
                    ((scores - mean_score) ** 2) * probs
                )
            )

            row = {
                "image_num": int(image_id),
                "pred_score": float(mean_score),
                "pred_std": float(std_score)
            }

            for i in range(10):
                row[f"pred_vote_{i+1}"] = float(probs[i])

            results.append(row)


# =========================
# 9. 保存预测结果
# =========================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n推理完成！")
print("成功预测图片数量：", len(result_df))
print("结果保存位置：", OUTPUT_PATH)

print("\n预测评分统计：")
print(result_df["pred_score"].describe())
