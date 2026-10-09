import torch
from torchvision.models import resnet18, ResNet18_Weights

# 加载一个预训练好的经典模型（权重会自动下载，约 45MB）
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()   # 切到推理模式

x = torch.rand(1, 3, 224, 224)   # 造一个假输入：1张图，3通道，224x224
with torch.no_grad():            # 推理时不需要梯度，关掉省内存
    out = model(x)

print("输出形状:", out.shape)    # 预期 [1, 1000] —— 1000个类别的得分
print("最大值:", out.max().item())