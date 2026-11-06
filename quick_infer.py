import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from model.model import NIMA

# 1. 选择设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)

# 2. 创建 NIMA 模型
base_model = models.vgg16(weights=None)
model = NIMA(base_model)

# 3. 加载预训练权重
state_dict = torch.load(
    "weights/epoch-82.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

print("NIMA 模型加载成功！")

# 4. 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("test.jpg").convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

# 5. 模型推理
with torch.no_grad():
    output = model(image)

# 6. 输出评分概率分布
probabilities = output.squeeze(0).cpu()

print("\n美学评分概率分布：")

for i, p in enumerate(probabilities, 1):
    print(f"{i}分: {p.item():.4f}")

# 7. 计算期望评分
scores = torch.arange(1, 11, dtype=torch.float32)

mean_score = torch.sum(scores * probabilities)

std_score = torch.sqrt(
    torch.sum(probabilities * (scores - mean_score) ** 2)
)

print("\n最终结果：")
print("美学评分:", mean_score.item())
print("评分标准差:", std_score.item())