
import csv
from pathlib import Path

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from model.model import NIMA


# ==============================
# 1. 路径配置
# ==============================

# 当前程序所在的项目根目录
ROOT_DIR = Path(__file__).resolve().parent

# 图片文件夹
IMAGE_DIR = ROOT_DIR / "test_images"

# NIMA预训练权重
WEIGHT_PATH = ROOT_DIR / "weights" / "epoch-82.pth"

# 结果保存路径
OUTPUT_CSV = ROOT_DIR / "batch_results.csv"


# ==============================
# 2. 图片预处理
# ==============================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# 3. 加载模型
# ==============================

def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备：", device)
    print("正在加载 NIMA 模型...")

    # 创建VGG16网络结构，不额外下载权重
    base_model = models.vgg16(weights=None)

    # 创建NIMA模型
    model = NIMA(base_model)

    # 加载已经训练好的权重
    state_dict = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)

    # 将模型放到GPU或CPU
    model.to(device)

    # 设置为推理模式
    model.eval()

    print("NIMA模型加载成功！\n")

    return model, device


# ==============================
# 4. 单张图片推理
# ==============================

def predict_image(model, device, image_path):

    # 打开图片，并统一转换为RGB
    image = Image.open(image_path).convert("RGB")

    # 图片预处理
    image_tensor = transform(image)

    # 增加batch维度
    image_tensor = image_tensor.unsqueeze(0)

    # 将数据放到对应设备
    image_tensor = image_tensor.to(device)

    # 不计算梯度，节省资源
    with torch.inference_mode():

        output = model(image_tensor)

    # 获取10个评分等级的概率
    probabilities = output.squeeze(0).cpu()

    # 评分等级：1～10
    scores = torch.arange(1, 11, dtype=torch.float32)

    # 计算期望评分
    mean_score = torch.sum(probabilities * scores).item()

    # 计算评分分布的标准差
    variance = torch.sum(
        probabilities * (scores - mean_score) ** 2
    )

    std_score = torch.sqrt(variance).item()

    return probabilities.tolist(), mean_score, std_score


# ==============================
# 5. 批量处理图片
# ==============================

def batch_inference():

    # 检查路径
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"图片文件夹不存在：{IMAGE_DIR}"
        )

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(
            f"模型权重不存在：{WEIGHT_PATH}"
        )

    # 加载一次模型，后续所有图片共用
    model, device = load_model()

    # 支持的图片格式
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    # 保存所有结果
    results = []

    # 统计数量
    success_count = 0
    failed_count = 0

    # 获取所有类别文件夹
    category_dirs = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.is_dir()]
    )

    if not category_dirs:
        print("没有找到类别文件夹！")
        return

    print("开始批量图片测试...\n")

    # 遍历每个类别
    for category_dir in category_dirs:

        category = category_dir.name

        # 获取当前类别的图片
        image_paths = sorted([
            p for p in category_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
        ])

        print(f"类别：{category}，图片数量：{len(image_paths)}")

        # 遍历当前类别的图片
        for image_path in image_paths:

            try:
                probabilities, mean_score, std_score = predict_image(
                    model,
                    device,
                    image_path
                )

                # 整理结果
                row = {
                    "image_name": image_path.name,
                    "category": category,
                    "score": round(mean_score, 4),
                    "std": round(std_score, 4)
                }

                # 添加1～10分的概率
                for i, probability in enumerate(probabilities):
                    row[f"p{i + 1}"] = round(probability, 6)

                results.append(row)

                success_count += 1

                print(
                    f"  {image_path.name:<25}"
                    f"评分：{mean_score:.4f}"
                    f"  标准差：{std_score:.4f}"
                )

            except Exception as e:

                failed_count += 1

                print(f"  处理失败：{image_path.name}")
                print(f"  错误原因：{e}")

        print()

    # ==============================
    # 6. 保存CSV结果
    # ==============================

    if results:

        fieldnames = [
            "image_name",
            "category",
            "score",
            "std"
        ] + [
            f"p{i}" for i in range(1, 11)
        ]

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

    # ==============================
    # 7. 输出统计信息
    # ==============================

    print("=" * 55)
    print("批量测试完成！")
    print("=" * 55)

    print(f"成功处理：{success_count} 张")
    print(f"处理失败：{failed_count} 张")
    print(f"总计结果：{len(results)} 张")

    if results:
        print(f"结果文件：{OUTPUT_CSV}")

    print("=" * 55)


# ==============================
# 8. 程序入口
# ==============================

if __name__ == "__main__":
    batch_inference()
