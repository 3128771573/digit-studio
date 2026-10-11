# -*- coding: utf-8 -*-
"""
utils.py —— 工具函数集合

包含四类功能：
  1) 环境与复现：随机种子固定、设备信息、模型结构摘要
  2) 数据相关  ：MNIST 下载/加载（torchvision + numpy）、数据划分
  3) 绘图相关  ：matplotlib 静态图（样本预览、训练曲线、混淆矩阵、预测对比、错误样本）
                 网页端图表由 static/app.js 绘制
  4) 预处理    ：把鼠标手写图像变成网络要求的 28x28 黑底白字张量
"""

import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# 用 Agg 后端：无显示器/服务器环境也能稳定出图，且不会弹窗卡住进程
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (必须在 use("Agg") 之后导入)
from matplotlib.figure import Figure  # noqa: E402

# ============================================================================
#  常量定义
# ============================================================================
SEED = 42                                  # 全局随机种子，保证可复现
DATA_DIR = "./data"                        # MNIST 自动下载目录
OUTPUT_DIR = "outputs"                     # 训练结束后各类可视化图片的输出目录
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081     # MNIST 官方统计的均值/标准差
IMAGE_SIZE = 28                            # 网络输入尺寸

# 给 matplotlib 设置中文字体候选，避免中文标题变成方框
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.max_open_warning"] = 0
for _font in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        continue


# ============================================================================
#  1) 环境与复现
# ============================================================================
def set_seed(seed: int = SEED) -> None:
    """固定所有随机源，保证每次运行结果一致（seed=42）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多卡/单卡 CUDA 随机源
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 下面两项牺牲一点点速度换取更强的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, str]:
    """返回当前设备信息字典，用于网页展示（设备名、CUDA 版本、显存等）。"""
    info = {"torch": torch.__version__, "cuda_available": str(torch.cuda.is_available())}
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["device"] = f"cuda:{idx} - {props.name}"
            info["cuda_version"] = str(torch.version.cuda)
            info["memory_GB"] = f"{props.total_memory / 1024 ** 3:.1f}"
        except Exception:
            info["device"] = "cuda"
    else:
        info["device"] = "cpu"
    return info


def model_summary(model: torch.nn.Module,
                  input_size: Tuple[int, int, int, int] = (1, 1, IMAGE_SIZE, IMAGE_SIZE),
                  device: Optional[torch.device] = None) -> str:
    """打印/返回模型结构摘要（每层输出形状 + 参数量）。

    优先使用 torchinfo；若未安装则用手动实现兜底，保证功能不缺失。
    """
    device = device or next(model.parameters()).device
    # ---- 优先方案：torchinfo ----
    try:
        from torchinfo import summary as torchinfo_summary
        text = str(torchinfo_summary(model, input_size=input_size, device=str(device),
                                     col_names=("input_size", "output_size", "num_params", "trainable"),
                                     verbose=0))
        return text
    except Exception:
        pass

    # ---- 兜底方案：手动前向 + hook 收集形状 ----
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"{'Layer (type)':<28}{'Output Shape':<24}{'Param #':>15}")
    lines.append("=" * 78)
    total = 0
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            nonlocal total
            n = sum(p.numel() for p in _module.parameters() if p.requires_grad)
            total += n
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else "?"
            lines.append(f"{name:<28}{str(shape):<24}{n:>15,}")
        return hook

    for name, module in model.named_children():
        handles.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(*input_size, device=device))
    finally:
        for h in handles:
            h.remove()
    lines.append("=" * 78)
    lines.append(f"Total trainable params: {total:,}")
    lines.append("（安装 torchinfo 可获得更详细的分层摘要：pip install torchinfo）")
    return "\n".join(lines)


# ============================================================================
#  2) 数据相关
# ============================================================================
def get_transforms(train: bool = False) -> transforms.Compose:
    """构建图像变换流水线。

    训练集做轻微随机平移（数据增强），验证/测试集只做标准化，
    保证评估结果稳定可比。

    Returns:
        torchvision.transforms.Compose
    """
    aug: List = []
    if train:
        # 随机平移 ±2 像素：模拟手写位置偏移，提升网页手写输入的鲁棒性
        aug.append(transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)))
    aug += [
        transforms.ToTensor(),                              # PIL -> Tensor, 并归一化到 [0,1]
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),  # 标准化到均值 0 / 方差 1 附近
    ]
    return transforms.Compose(aug)


def get_dataloaders(batch_size: int = 64,
                    data_dir: str = DATA_DIR,
                    num_workers: Optional[int] = None,
                    val_size: int = 5000,
                    download: bool = True,
                    verbose: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """下载（如需要）并加载 MNIST，返回 train / val / test 三个 DataLoader。

    做法：用 torchvision 下载到 ./data，从官方训练集中切出 5000 张作为验证集
    （官方测试集 10000 张只用于最终评估，不参与训练，避免数据泄漏）。

    Args:
        batch_size:  批大小。
        data_dir:    MNIST 保存目录，默认 ./data。
        num_workers: DataLoader 子进程数；None 表示自动（GPU 时 2，CPU 时 0）。
        val_size:    验证集大小。
        download:    是否允许联网下载。
        verbose:     是否打印日志。
    Returns:
        (train_loader, val_loader, test_loader)
    Raises:
        RuntimeError: 数据下载/读取失败时抛出，附带友好提示。
    """
    os.makedirs(data_dir, exist_ok=True)
    if num_workers is None:
        num_workers = 0  # Windows 上 CPU 训练用 0 最稳，避免多进程开销和卡顿

    try:
        # 训练集（带数据增强）
        train_full = datasets.MNIST(root=data_dir, train=True, download=download,
                                    transform=get_transforms(train=True))
        # 验证集用"无增强"的副本，保证评估稳定
        val_full = datasets.MNIST(root=data_dir, train=True, download=False,
                                  transform=get_transforms(train=False))
        test_set = datasets.MNIST(root=data_dir, train=False, download=download,
                                  transform=get_transforms(train=False))
    except Exception as exc:
        raise RuntimeError(
            "MNIST 数据集加载失败！请检查网络（首次运行需联网下载约 11MB 到 ./data 目录），"
            "或手动下载 4 个文件放入 ./data/MNIST/raw/ ：\n"
            "  train-images-idx3-ubyte.gz / train-labels-idx1-ubyte.gz / "
            "t10k-images-idx3-ubyte.gz / t10k-labels-idx1-ubyte.gz\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc

    n_train = len(train_full)
    val_size = min(val_size, n_train // 5)          # 防止验证集比例过大
    # 用固定种子打乱索引后再切分，保证每次运行划分一致（可复现）
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_train, generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_set: Dataset = Subset(train_full, train_idx)
    val_set: Dataset = Subset(val_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              drop_last=False)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())

    if verbose:
        print(f"[数据] 训练集 {len(train_set)} 张 / 验证集 {len(val_set)} 张 / 测试集 {len(test_set)} 张"
              f"（batch_size={batch_size}，目录={os.path.abspath(data_dir)}）")
    return train_loader, val_loader, test_loader


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """把标准化后的张量还原成 [0,1] 之间的 numpy 图像，便于绘图。

    支持 (C,H,W) 与 (N,C,H,W) 两种形状。
    """
    arr = tensor.detach().cpu().numpy()
    arr = arr * MNIST_STD + MNIST_MEAN          # 反标准化
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[:, 0]                          # 取单通道 -> (N,H,W)
    elif arr.ndim == 3:
        arr = arr[0]                             # (C,H,W) -> (H,W)
    return arr


# ============================================================================
#  3) 绘图相关 —— matplotlib（静态图，用于训练前后自动生成的报告图）
# ============================================================================
def _new_fig(nrows: int = 1, ncols: int = 1, figsize: Tuple[float, float] = (8, 4)) -> Tuple[Figure, np.ndarray]:
    """创建一个 matplotlib Figure 与坐标轴数组（AgAcc 后端，不弹窗）。"""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor("white")
    return fig, axes


def fig_to_array(fig: Figure, dpi: int = 110) -> np.ndarray:
    """把 matplotlib Figure 转成 RGB numpy 数组（用于 Gradio 显示 / 保存 PNG）。"""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def save_fig(fig: Figure, path: str, close: bool = True) -> str:
    """保存 Figure 到磁盘（自动建目录），返回绝对路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return os.path.abspath(path)


def plot_sample_grid(images: np.ndarray,
                     labels: Optional[Sequence[int]] = None,
                     n: int = 9,
                     title: str = "训练前：MNIST 随机样本预览") -> Figure:
    """训练前展示：随机 9 张 MNIST 样本图 + 对应标签。

    Args:
        images: (N,28,28) 或 (N,1,28,28) 的 numpy 数组，取值 [0,1]。
        labels: 对应标签列表。
        n:      展示张数，默认 9（3x3）。
        title:  图标题。
    Returns:
        matplotlib Figure
    """
    if images.ndim == 4:
        images = images[:, 0]
    n = min(n, len(images))
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = _new_fig(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            if labels is not None:
                # 绿色 = 真实标签
                ax.set_title(f"标签: {int(labels[i])}", fontsize=11, color="#1a7f37")
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_training_curves(history: Dict[str, List[float]],
                         title: str = "训练过程曲线") -> Figure:
    """Loss / Accuracy 双曲线图（训练集 + 验证集，共 4 条线）。"""
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    fig, axes = _new_fig(1, 2, figsize=(12, 4.2))

    ax = axes[0][0]
    if len(epochs):
        ax.plot(epochs, history["train_loss"], "o-", label="训练集 Loss", color="#1f77b4")
        ax.plot(epochs, history["val_loss"], "s-", label="验证集 Loss", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss 曲线")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0][1]
    if len(epochs):
        ax.plot(epochs, np.array(history["train_acc"]) * 100, "o-", label="训练集 Accuracy", color="#1f77b4")
        ax.plot(epochs, np.array(history["val_acc"]) * 100, "s-", label="验证集 Accuracy", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy 曲线")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          title: str = "混淆矩阵 (测试集)") -> Figure:
    """绘制 10x10 混淆矩阵热力图，格子里标注样本数量。"""
    cm = np.asarray(cm)
    fig, ax = _new_fig(1, 1, figsize=(7.6, 6.6))
    ax = ax[0][0]
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="样本数")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("预测类别 (Predicted)", fontsize=11)
    ax.set_ylabel("真实类别 (True)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    # 数值标注：深色背景用白字，浅色背景用黑字
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "#222222")
    fig.tight_layout()
    return fig


def plot_prediction_grid(images: np.ndarray,
                         labels: Sequence[int],
                         preds: Sequence[int],
                         probs: Optional[Sequence[float]] = None,
                         n: int = 16,
                         title: str = "测试集随机 16 张：预测值 vs 真实值（红色=预测错误）") -> Figure:
    """随机 16 张测试样本，标题显示「预测值 vs 真实值」，错误的用红色标注。"""
    if images.ndim == 4:
        images = images[:, 0]
    n = min(n, len(images))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = _new_fig(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i >= n:
            continue
        ok = int(preds[i]) == int(labels[i])
        color = "#1a7f37" if ok else "#d62728"     # 绿=正确，红=错误
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        prob_txt = f"\n{probs[i]:.1%}" if probs is not None else ""
        ax.set_title(f"预测:{int(preds[i])} vs 真实:{int(labels[i])}{prob_txt}",
                     fontsize=10, color=color, fontweight="bold" if not ok else "normal")
        if not ok:
            # 再画一圈红框，视觉上更醒目
            for spine in ax.spines.values():
                spine.set_edgecolor("#d62728")
                spine.set_linewidth(2.2)
                spine.set_visible(True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_error_gallery(images: np.ndarray,
                       labels: Sequence[int],
                       preds: Sequence[int],
                       max_n: int = 25,
                       title_prefix: str = "全部错误分类样本") -> Figure:
    """单独展示所有被错误分类的样本（最多 max_n=25 张）。"""
    if images.ndim == 4:
        images = images[:, 0]
    total = len(images)
    n = min(total, max_n)
    cols = 5
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = _new_fig(rows, cols, figsize=(cols * 2.3, rows * 2.5))
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"预测:{int(preds[i])} ≠ 真实:{int(labels[i])}", fontsize=9.5, color="#d62728")
    title = (f"{title_prefix}：共 {total} 张，展示前 {n} 张"
             if total > n else f"{title_prefix}：共 {total} 张")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_maps(feature_map: torch.Tensor,
                      max_channels: int = 16,
                      title: str = "第一层卷积特征图 (Conv1 ReLU 输出)") -> Figure:
    """可视化第一层卷积输出的特征图（feature maps）。

    Args:
        feature_map: 形状 (1, C, H, W) 或 (C, H, W) 的激活张量。
        max_channels: 最多显示多少个通道（默认 16，满足"至少 8 张"的要求）。
    """
    fm = feature_map.detach().cpu()
    if fm.dim() == 4:
        fm = fm[0]
    c = min(fm.shape[0], max_channels)
    cols = 4
    rows = int(np.ceil(c / cols))
    fig, axes = _new_fig(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i >= c:
            continue
        m = fm[i].numpy()
        ax.imshow(m, cmap="viridis")
        ax.set_title(f"通道 {i}  {tuple(m.shape)}", fontsize=9)
    fig.suptitle(f"{title}（显示 {c} 个通道）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================================
#  3) 绘图相关 —— Plotly（网页内动态刷新）
# ============================================================================
def _center_by_mass(binary: np.ndarray, out_size: int = IMAGE_SIZE, box: int = 20) -> np.ndarray:
    """把二值化数字按"质心"居中缩放，模仿 MNIST 的预处理方式。

    MNIST 官方做法：把数字等比缩放到 20x20 然后按质心放到 28x28 中央。
    网页手写图如果直接缩放到 28x28，写大/写小的数字分布会与训练集差异很大，
    这里做同样的归一化可以显著提升网页预测准确率。
    """
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros((out_size, out_size), dtype=np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = binary[y0:y1, x0:x1].astype(np.float32)

    # 等比缩放到 20x20 之内（保持长宽比，避免数字被拉变形）
    h, w = crop.shape
    scale = box / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    crop_img = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(crop_img, dtype=np.float32) / 255.0

    # 按质心对齐到 28x28 画布中心
    canvas = np.zeros((out_size, out_size), dtype=np.float32)
    cy, cx = out_size // 2, out_size // 2
    top, left = cy - new_h // 2, cx - new_w // 2
    canvas[top:top + new_h, left:left + new_w] = small
    return canvas


def _to_uint8_array(image) -> np.ndarray:
    """把任意输入图像统一转成 uint8 的 0~255 灰度数组。

    这一步很关键：PIL 的 ``Image.fromarray`` 会把 float 数组按"已经是 0~255"解释，
    如果直接把 [0,1] 的 float 图（例如 MNIST 张量反标准化后的结果）传进去，
    整张图会被截断成几乎全黑，从而出现"所有图片都预测成同一个数字"的 bug。
    因此这里先根据 dtype / 数值范围判断量纲，再安全地映射到 0~255。

    Args:
        image: PIL.Image 或 numpy 数组（可能是 0~1 的 float，也可能是 0~255 的 uint8）。
    Returns:
        np.ndarray(uint8)，形状 (H,W) 或 (H,W,3/4)
    """
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)

    # Gradio 6 的 Image 组件在"上传"场景可能传回文件路径字符串，或 {"path": ...} 字典
    if isinstance(image, (str, os.PathLike)):
        with Image.open(os.fspath(image)) as im:
            return np.asarray(im.convert("L"), dtype=np.uint8)
    if isinstance(image, dict):
        path = image.get("path") or image.get("url") or image.get("name")
        if isinstance(path, (str, os.PathLike)) and os.path.exists(os.fspath(path)):
            with Image.open(os.fspath(path)) as im:
                return np.asarray(im.convert("L"), dtype=np.uint8)
        # 有些组件把像素直接放在 'background' / 'image' 字段里
        for key in ("background", "image", "data"):
            val = image.get(key)
            if val is not None:
                return _to_uint8_array(val)

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("输入图像为空。")

    # 整数类型：直接按 0~255 处理（Gradio 画板返回的就是这种）
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)

    # 浮点类型：看最大值的量纲来判断是 0~1 还是 0~255
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("输入图像全为非有限值。")
    vmax = float(finite.max())
    if vmax <= 1.0001:                       # 0~1 量纲 -> 放大到 0~255
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def preprocess_digit(image,
                     invert: bool = True,
                     auto_center: bool = True,
                     apply_normalize: bool = True) -> np.ndarray:
    """把鼠标手写图像预处理成网络要求的 28x28 「黑底白字」图像。

    处理流程：
      1) 转灰度（RGB 用 ITU-R 601-2 亮度公式）
      2) 反色（Gradio 画板默认白底黑字，而 MNIST 是黑底白字）
      3) （可选）裁剪数字外接框 + 按质心居中缩放，对齐 MNIST 分布
      4) 缩放到 28x28
      5) （可选）用 MNIST 的均值/标准差标准化

    Args:
        image:           PIL.Image 或 numpy 数组（画板输出，uint8 0~255）。
        invert:          是否反色（白底黑字 -> 黑底白字）。
        auto_center:     是否做裁剪+质心居中（推荐开启，识别率更高）。
        apply_normalize: 是否做标准化（喂给网络时 True；单纯展示时 False）。
    Returns:
        (28,28) 的 float32 numpy 数组。
    """
    # --- 1) 统一转成 PIL 灰度图 ---
    if image is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    img = Image.fromarray(_to_uint8_array(image)).convert("L")

    # --- 2) 反色 ---
    if invert:
        img = ImageOps.invert(img)

    # --- 3~4) 归一化到 [0,1] 后做自动居中缩放 ---
    gray = np.asarray(img, dtype=np.float32) / 255.0
    if auto_center:
        binary = np.where(gray > 0.12, gray, 0.0)        # 去掉几乎为黑的底噪
        if binary.max() > 0:
            gray28 = _center_by_mass(binary)
        else:
            gray28 = np.asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS),
                                dtype=np.float32) / 255.0
    else:
        gray28 = np.asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS),
                            dtype=np.float32) / 255.0

    # --- 5) 标准化 ---
    if apply_normalize:
        return ((gray28 - MNIST_MEAN) / MNIST_STD).astype(np.float32)
    return gray28.astype(np.float32)


def preprocess_to_tensor(image, invert: bool = True, auto_center: bool = True) -> torch.Tensor:
    """预处理并转成可直接送入网络的张量，形状 (1,1,28,28)。"""
    arr = preprocess_digit(image, invert=invert, auto_center=auto_center, apply_normalize=True)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (28,28) -> (1,1,28,28)


def preprocess_uploaded_image(image, invert: bool = True, auto_center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """处理「模型分析」Tab 里用户上传的任意图片（可能是深色背景或浅色背景）。

    与画板不同，上传图可能已经是黑底白字（如 MNIST 截图），也可能是白底黑字。
    这里用「四周边框像素的平均亮度」自动判断是否需要反色，兼容两种情况。

    Args:
        image: PIL.Image / numpy 数组。
        invert: 是否启用自动反色判断（False 表示强制按白底黑字处理）。
        auto_center: 是否做裁剪 + 质心居中（与画板一致的归一化）。
    Returns:
        (display_28, tensor_1x1x28x28) —— 展示用 [0,1] 灰度图 与 送入网络的张量
    """
    if image is None:
        raise ValueError("请先上传图片或点击「随机抽取测试样本」。")
    # ---- 统一转成灰度 PIL（内部会先判断 0~1 / 0~255 量纲，避免整图变黑）----
    img = Image.fromarray(_to_uint8_array(image)).convert("L")

    # ---- 自动判断背景：取边框一圈像素的平均亮度 ----
    gray_full = np.asarray(img, dtype=np.float32) / 255.0
    h, w = gray_full.shape
    border = np.concatenate([
        gray_full[0, :].ravel(), gray_full[-1, :].ravel(),
        gray_full[:, 0].ravel(), gray_full[:, -1].ravel()])
    need_invert = invert and float(border.mean()) > 0.5   # 边框偏亮 => 白底 => 需要反色

    # ---- 复用画板的预处理流水线（裁剪 + 质心居中 + 缩放 + 标准化）----
    tensor = preprocess_to_tensor(img, invert=need_invert, auto_center=auto_center)
    display = preprocess_digit(img, invert=need_invert, auto_center=auto_center, apply_normalize=False)
    return display, tensor


def available_test_sample_images(test_set: datasets.MNIST) -> None:
    """占位说明函数：保留接口，实际随机抽样见 sample_test_batch。"""
    return None


def sample_test_batch(test_set: Dataset,
                      k: int = 16,
                      correct_only: bool = False,
                      wrong_only: bool = False,
                      model: Optional[torch.nn.Module] = None,
                      device: Optional[torch.device] = None,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从测试集中随机抽 k 张样本。

    Args:
        test_set:     测试集（MNIST 或 Subset）。
        k:            抽取数量。
        correct_only: 只要预测正确的样本（需提供 model）。
        wrong_only:   只要预测错误的样本（需提供 model）。
        model/device: 用于筛选时需要的模型与设备。
        seed:         抽样随机种子；None 表示真随机（每次不同）。
    Returns:
        (images, labels, preds)
        images: (k,28,28) float numpy，取值 [0,1]（已反标准化，可直接显示）
        labels: (k,) 真实标签
        preds:  (k,) 预测标签；未提供 model 时为 None
    """
    # 取底层 MNIST 数据集（Subset 的情况取其 .dataset 的子索引）
    targets = np.asarray(getattr(test_set, "targets", None))
    if targets is None and isinstance(test_set, Subset):
        base = test_set.dataset
        targets = np.asarray(base.targets)[test_set.indices]
    total = len(targets)

    rng = np.random.default_rng(seed)
    if model is not None and (wrong_only or correct_only):
        # 需要先预测一遍，再从对/错样本池里抽
        device = device or next(model.parameters()).device
        indices = rng.choice(total, size=min(2000, total), replace=False)
        imgs, labs = [], []
        for idx in indices:
            img, lab = test_set[int(idx)] if not isinstance(test_set, Subset) else test_set[int(idx)]
            imgs.append(img)
            labs.append(lab)
        batch = torch.stack(imgs)
        with torch.no_grad():
            out = model(batch.to(device))
            pred_all = out.argmax(dim=1).cpu().numpy()
        labs = np.asarray(labs)
        mask = (pred_all != labs) if wrong_only else (pred_all == labs)
        pool = np.where(mask)[0]
        if len(pool) == 0:                              # 兜底：没有符合条件的样本
            pool = np.arange(len(labs))
        pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
        images = denormalize(batch[pick])
        return images, labs[pick], pred_all[pick]

    # 普通随机抽样
    idx = rng.choice(total, size=min(k, total), replace=False)
    imgs, labs = [], []
    for i in idx:
        img, lab = test_set[int(i)]
        imgs.append(img)
        labs.append(lab)
    batch = torch.stack(imgs)
    preds = None
    if model is not None:
        device = device or next(model.parameters()).device
        with torch.no_grad():
            preds = model(batch.to(device)).argmax(dim=1).cpu().numpy()
    return denormalize(batch), np.asarray(labs), preds


def save_history_csv(history: Dict[str, List[float]],
                     path: str = os.path.join(OUTPUT_DIR, "training_log.csv")) -> Optional[str]:
    """把训练日志导出为 CSV（加分项）。

    Returns:
        成功返回文件绝对路径；pandas 不可用或无数据时返回 None。
    """
    if not history or not history.get("train_loss"):
        return None
    try:
        import pandas as pd
        n = len(history["train_loss"])
        df = pd.DataFrame({
            "epoch": np.arange(1, n + 1),
            "train_loss": history.get("train_loss", [None] * n),
            "val_loss": history.get("val_loss", [None] * n),
            "train_acc": history.get("train_acc", [None] * n),
            "val_acc": history.get("val_acc", [None] * n),
            "lr": history.get("lr", [None] * n),
            "epoch_time_sec": history.get("epoch_time", [None] * n),
        })
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return os.path.abspath(path)
    except Exception as exc:
        print(f"[警告] CSV 导出失败：{exc}")
        return None


# ============================================================================
#  训练结束后的完整可视化报告（供 train.py 与 app.py 复用）
# ============================================================================
def generate_post_training_report(model: torch.nn.Module,
                                  test_set: Dataset,
                                  history: Optional[Dict[str, List[float]]] = None,
                                  device: Optional[torch.device] = None,
                                  out_dir: str = OUTPUT_DIR,
                                  max_errors: int = 25,
                                  verbose: bool = True) -> Dict[str, str]:
    """训练结束后自动生成并保存四类可视化图片。

    产出文件（保存在 out_dir 目录）：
        training_curves.png   —— Loss/Accuracy 曲线
        confusion_matrix.png  —— 10x10 混淆矩阵热力图
        predictions_16.png    —— 随机 16 张：预测值 vs 真实值（错误标红）
        error_samples.png     —— 全部错误分类样本（最多 25 张）

    Returns:
        {名称: 绝对路径} 字典；失败时对应值为 None。
    """
    device = device or next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)
    art: Dict[str, str] = {}
    model.eval()

    # ---- (a) 训练曲线 ----
    if history and history.get("train_loss"):
        try:
            art["training_curves"] = save_fig(plot_training_curves(history), os.path.join(out_dir, "training_curves.png"))
        except Exception as exc:
            print(f"[警告] 训练曲线绘制失败：{exc}")

    # ---- 在测试集上整体推理一次，得到混淆矩阵与全部错误样本 ----
    try:
        cm = np.zeros((10, 10), dtype=np.int64)
        all_imgs, all_labels, all_preds, all_confs = [], [], [], []
        loader = DataLoader(test_set, batch_size=512, shuffle=False)
        total = correct = 0
        with torch.no_grad():
            for images, labels in loader:
                out = model(images.to(device))
                prob = torch.softmax(out, dim=1)
                conf, pred = prob.max(dim=1)
                pred_cpu, lab_cpu = pred.cpu().numpy(), labels.numpy()
                for t, p in zip(lab_cpu, pred_cpu):
                    cm[t, p] += 1
                total += len(lab_cpu)
                correct += int((pred_cpu == lab_cpu).sum())
                all_imgs.append(denormalize(images))
                all_labels.append(lab_cpu)
                all_preds.append(pred_cpu)
                all_confs.append(conf.cpu().numpy())

        images_all = np.concatenate(all_imgs)
        labels_all = np.concatenate(all_labels)
        preds_all = np.concatenate(all_preds)
        confs_all = np.concatenate(all_confs)
        acc = correct / max(total, 1)
        if verbose:
            print(f"[评估] 测试集准确率：{acc:.4%}（{correct}/{total}）")

        # ---- (b) 混淆矩阵热力图 ----
        art["confusion_matrix"] = save_fig(plot_confusion_matrix(cm), os.path.join(out_dir, "confusion_matrix.png"))

        # ---- (c) 随机 16 张预测对比 ----
        rng = np.random.default_rng(SEED)
        pick16 = rng.choice(len(images_all), size=min(16, len(images_all)), replace=False)
        art["predictions_16"] = save_fig(
            plot_prediction_grid(images_all[pick16], labels_all[pick16], preds_all[pick16], confs_all[pick16]),
            os.path.join(out_dir, "predictions_16.png"))

        # ---- (d) 全部错误样本（最多 25 张）----
        wrong = np.where(preds_all != labels_all)[0]
        art["error_samples"] = save_fig(
            plot_error_gallery(images_all[wrong], labels_all[wrong], preds_all[wrong], max_n=max_errors),
            os.path.join(out_dir, "error_samples.png"))
        if verbose:
            print(f"[报告] 已生成可视化图片到 {os.path.abspath(out_dir)}："
                  f"{', '.join(os.path.basename(v) for v in art.values() if v)}")
    except Exception as exc:
        print(f"[警告] 测试集评估/报告生成失败：{exc}")

    # ---- (e) 训练日志 CSV（加分项）----
    csv_path = save_history_csv(history or {}, os.path.join(out_dir, "training_log.csv"))
    if csv_path:
        art["history_csv"] = csv_path
        if verbose:
            print(f"[日志] 训练日志 CSV：{csv_path}")
    return art


def format_seconds(seconds: float) -> str:
    """把秒数格式化成易读字符串，如 12.3s / 1m23s。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{seconds % 60:.0f}s"


def now() -> float:
    """当前时间戳（秒），用于统计 epoch 耗时。"""
    return time.time()
