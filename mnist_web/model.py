# -*- coding: utf-8 -*-
"""
model.py —— 神经网络定义（CNN，LeNet 改进版）

【整体数据流：输入 -> 输出 维度说明】
    输入        : (N, 1, 28, 28)          # 灰度图，单通道
    conv1 3x3   : (N, 32, 28, 28)         # 32 个卷积核, padding=1 保持尺寸
    relu        : (N, 32, 28, 28)
    pool1 2x2   : (N, 32, 14, 14)         # 最大池化, 尺寸减半
    conv2 3x3   : (N, 64, 14, 14)         # 64 个卷积核, padding=1 保持尺寸
    relu        : (N, 64, 14, 14)
    pool2 2x2   : (N, 64, 7, 7)           # 最大池化, 尺寸再减半
    flatten     : (N, 64*7*7 = 3136)
    fc1         : (N, 128)                # 全连接 + ReLU
    dropout     : (N, 128)                # 训练时随机丢弃 25%, 抑制过拟合
    fc2         : (N, 10)                 # 输出 10 类 logits(未归一化分数)

【结构要求对照】
    * 2 个卷积层 (conv1, conv2)                       ✔
    * 2 个池化层 (pool1, pool2, 均为 2x2 MaxPool)     ✔
    * 2 个全连接层 (fc1, fc2)                         ✔
    * 卷积后使用 ReLU 激活                            ✔
    * 最后一层输出 10 类                              ✔
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 固定随机种子（保证结果可复现，seed=42）
SEED = 42

# 默认模型保存文件名（训练脚本、网页应用共用一个模型文件）
MODEL_PATH = "mnist_cnn.pth"


class MnistCNN(nn.Module):
    """LeNet 改进版 CNN：2 卷积 + 2 池化 + 2 全连接。

    比原始 LeNet-5 更强的地方：使用 3x3 小卷积核 + padding 保持尺寸、
    通道数更多(32/64)、加入 Dropout 与 BatchNorm，MNIST 测试集准确率通常 99%+。
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        """初始化各层。

        Args:
            num_classes: 分类数，MNIST 为 10。
            dropout:     dropout 丢弃概率。
        """
        super().__init__()
        # ---- 卷积特征提取部分 ----
        # conv1: 输入 1 通道 -> 输出 32 通道，卷积核 3x3，padding=1 保证 28x28 不变
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)          # 批归一化，加速收敛、稳定训练
        # conv2: 输入 32 通道 -> 输出 64 通道，卷积核 3x3，padding=1 保证 14x14 不变
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # 2x2 最大池化，尺寸减半

        # ---- 全连接分类部分 ----
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64*7*7 = 3136 -> 128
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)  # 128 -> 10 类 logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：x 形状 (N,1,28,28) -> 返回 (N,10) 的 logits。"""
        # 第 1 组：卷积 -> ReLU -> 池化   (N,1,28,28) -> (N,32,14,14)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第 2 组：卷积 -> ReLU -> 池化   (N,32,14,14) -> (N,64,7,7)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 展平： (N,64,7,7) -> (N,3136)
        x = torch.flatten(x, start_dim=1)
        # 全连接 + ReLU + Dropout： (N,3136) -> (N,128)
        x = self.dropout(F.relu(self.fc1(x)))
        # 输出层： (N,128) -> (N,10)，注意这里不接 Softmax，
        # 因为 CrossEntropyLoss 内部自带 log_softmax（数值更稳定）
        x = self.fc2(x)
        return x


def get_device() -> torch.device:
    """自动检测运行设备：优先 CUDA(GPU)，其次 Apple MPS，最后回退 CPU。

    返回 torch.device，并打印实际使用的设备信息。
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[设备] 检测到 NVIDIA GPU: {gpu_name} (算力 sm_{cap[0]}{cap[1]})，使用 CUDA 加速")
            return device
    except Exception as exc:  # CUDA 驱动异常时不要让程序崩溃
        print(f"[设备] CUDA 检测失败（{exc}），尝试回退...")

    # Apple Silicon 的 MPS 后端（Mac 用户可用）
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[设备] 检测到 Apple MPS，使用 MPS 加速")
            return torch.device("mps")
    except Exception:
        pass

    print("[设备] 未检测到可用 GPU，自动回退 CPU（训练会慢一些，但功能完全一致）")
    return torch.device("cpu")


def create_model(num_classes: int = 10, device: Optional[torch.device] = None) -> nn.Module:
    """创建模型并移动到指定设备（device 为 None 时自动检测）。"""
    device = device or get_device()
    model = MnistCNN(num_classes=num_classes).to(device)
    return model


def save_model(model: nn.Module, path: str = MODEL_PATH, extra: Optional[dict] = None) -> str:
    """保存模型权重。

    Args:
        model: 待保存模型。
        path:  保存路径，默认 mnist_cnn.pth。
        extra: 额外信息（如 epoch、优化器状态、最佳准确率）一并写入。
    Returns:
        实际保存路径（绝对路径）。
    """
    payload = {"state_dict": model.state_dict(), "arch": "MnistCNN"}
    if extra:
        payload.update(extra)
    abs_path = os.path.abspath(path)
    torch.save(payload, abs_path)
    return abs_path


def load_model(path: str = MODEL_PATH,
               device: Optional[torch.device] = None,
               verbose: bool = True) -> Tuple[nn.Module, dict]:
    """加载模型权重。

    兼容两种保存格式：
      1) 新版 dict：{"state_dict": ..., "epoch": ..., "best_acc": ...}
      2) 老版/裸权重：直接就是 state_dict

    Args:
        path:    模型文件路径。
        device:  目标设备；None 表示自动检测。
        verbose: 是否打印日志。
    Returns:
        (model, meta) —— model 已 eval() 并放到 device 上；meta 含 epoch/best_acc 等。
    Raises:
        FileNotFoundError: 模型文件不存在。
        RuntimeError:      权重与网络结构不匹配。
    """
    device = device or get_device()
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型文件：{os.path.abspath(path)}")

    # weights_only=False：允许反序列化我们自定义保存的 dict（仅加载本地可信文件）
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # 老版本 PyTorch 没有 weights_only 参数
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state_dict, meta = ckpt, {}

    model = MnistCNN().to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"模型结构与权重不匹配，请删除 {os.path.basename(path)} 后重新训练。\n原始错误：{exc}"
        ) from exc
    model.eval()
    if verbose:
        acc = meta.get("best_acc")
        acc_txt = f"，记录准确率 {acc:.2%}" if isinstance(acc, (int, float)) else ""
        print(f"[模型] 已加载权重：{os.path.abspath(path)}{acc_txt}")
    return model, meta


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 直接运行 model.py 时，做一次自检：打印设备、参数量与前向输出形状
    from utils import set_seed  # 复用工具函数（避免重复实现）

    set_seed(SEED)
    dev = get_device()
    net = create_model(device=dev)
    print(net)
    print(f"[自检] 可训练参数量：{count_parameters(net):,}")
    dummy = torch.randn(4, 1, 28, 28, device=dev)
    with torch.no_grad():
        out = net(dummy)
    print(f"[自检] 输入 {tuple(dummy.shape)} -> 输出 {tuple(out.shape)}（应为 (4, 10)）")
