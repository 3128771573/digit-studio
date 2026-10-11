# -*- coding: utf-8 -*-
"""
train.py —— 训练脚本（含实时可视化回调）

两种用法：
  1) 命令行独立训练：
        python train.py --epochs 5 --batch-size 64 --lr 0.001
     训练前后会自动生成所有可视化图片到 ./outputs 目录，并保存模型 mnist_cnn.pth。

  2) 被网页 app.py 调用（推荐）：
        fit(epochs, batch_size, lr, stop_event, progress_cb) 是一个生成器，
        每训练若干 batch 就 yield 一次进度，网页 把 yield 的内容实时推给浏览器，
        从而实现「网页里 loss/accuracy 曲线动态刷新」。

可视化清单（本文件负责触发）：
  训练前  : 随机 9 张 MNIST 样本 + 标签        -> output_ 控制台打印 / outputs/samples_before_train.png
  训练中  : Loss 曲线(训练+验证)、Accuracy 曲线(训练+验证)、epoch 进度文字、当前学习率
            -> 网页 Plotly 实时刷新；命令行则实时打印
  训练后  : 混淆矩阵热力图(10x10)、随机 16 张预测vs真实(错误标红)、
            全部错误样本(最多 25 张)、训练日志 CSV
"""

import argparse
import os
import threading
import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import MODEL_PATH, MnistCNN, get_device, load_model, save_model
from utils import (
    OUTPUT_DIR,
    SEED,
    denormalize,
    fig_to_array,
    format_seconds,
    generate_post_training_report,
    get_dataloaders,
    plot_sample_grid,
    plot_training_curves,
    save_fig,
    set_seed,
)

# ============================================================================
#  基础组件：设备、模型、优化器
# ============================================================================
def build_optimizer(model: nn.Module, lr: float, optimizer_name: str = "adam",
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """构建优化器（默认 Adam，可选 SGD / RMSprop）。

    Args:
        model:          网络模型。
        lr:             学习率。
        optimizer_name: 'adam' | 'sgd' | 'rmsprop'（不区分大小写）。
        weight_decay:   L2 正则系数，抑制过拟合。
    """
    name = (optimizer_name or "adam").lower()
    if name == "sgd":
        # SGD + 动量，配合 StepLR 也能训练得不错
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int,
                    scheduler_name: str = "cosine"):
    """构建学习率调度器（默认余弦退火；也可选 step / none）。

    Returns:
        (scheduler, is_per_batch) —— is_per_batch=True 时每个 batch 调一次 step。
    """
    name = (scheduler_name or "cosine").lower()
    if name == "none" or epochs <= 1:
        return None, False
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3), False
    # 余弦退火：学习率平滑下降，通常收敛更稳、准确率更高
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5), False


# ============================================================================
#  单个 epoch 的训练 / 验证
# ============================================================================
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int,
                    stop_event: Optional[threading.Event] = None,
                    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
                    scheduler=None,
                    is_per_batch: bool = False) -> Dict[str, float]:
    """训练一个 epoch。

    Args:
        model/loader/criterion/optimizer/device: 常规训练组件。
        epoch/total_epochs: 用于进度显示。
        stop_event: threading.Event；被 set() 时在 batch 之间优雅退出（实现"中途停止"）。
        progress_cb: 回调 (batch_idx, total_batches, avg_loss, acc)，用于网页/控制台实时显示。
        scheduler / is_per_batch: 学习率调度器。
    Returns:
        {"loss": 平均损失, "acc": 准确率, "lr": 本 epoch 结束时的学习率,
         "stopped": 是否被中途停止, "batches_done": 实际完成的 batch 数}
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    stopped = False
    start = time.time()
    batch_idx = 0                                  # 预初始化，避免空 loader 时变量未定义

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        # ---- 中途停止检查（在 batch 边界检查，保证梯度状态一致）----
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # 清空梯度（set_to_none 更省显存）
        outputs = model(images)                 # 前向
        loss = criterion(outputs, labels)       # 计算损失
        loss.backward()                         # 反向
        optimizer.step()                        # 更新参数
        if is_per_batch and scheduler is not None:
            scheduler.step()

        # ---- 统计 ----
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

        # ---- 实时回调：每 20% 进度或最后一个 batch 触发一次 ----
        if progress_cb is not None:
            step = max(1, n_batches // 5)
            if batch_idx % step == 0 or batch_idx == n_batches:
                progress_cb(batch_idx, n_batches, running_loss / max(total, 1), correct / max(total, 1))

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    cur_lr = optimizer.param_groups[0]["lr"]
    return {"loss": avg_loss, "acc": acc, "lr": cur_lr, "stopped": stopped,
            "batches_done": batch_idx, "time": time.time() - start, "n": total}


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             collect_predictions: bool = False) -> Dict:
    """在验证集/测试集上评估。

    Args:
        collect_predictions: 为 True 时额外返回全部预测结果（用于混淆矩阵等）。
    Returns:
        {"loss":..., "acc":..., "preds": np.ndarray|None, "labels": np.ndarray|None,
         "probs": np.ndarray|None, "images": np.ndarray|None}
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all, probs_all, images_all = [], [], [], []

    for images, labels in loader:
        images_d = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        outputs = model(images_d)
        loss = criterion(outputs, labels_d)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += int((pred == labels_d).sum().item())
        total += labels.size(0)

        if collect_predictions:
            preds_all.append(pred.cpu().numpy())
            labels_all.append(labels.numpy())
            probs_all.append(probs.cpu().numpy())
            images_all.append(denormalize(images))

    result = {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "preds": np.concatenate(preds_all) if preds_all else None,
        "labels": np.concatenate(labels_all) if labels_all else None,
        "probs": np.concatenate(probs_all) if probs_all else None,
        "images": np.concatenate(images_all) if images_all else None,
    }
    return result


@torch.no_grad()
def predict_batch(model: nn.Module, images: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """对一批图像做预测。

    Returns:
        (preds, probs) —— preds 形状 (N,)，probs 形状 (N,10)
    """
    model.eval()
    outputs = model(images.to(device))
    probs = torch.softmax(outputs, dim=1)
    return probs.argmax(dim=1), probs


# ============================================================================
#  训练前可视化：随机展示 9 张 MNIST 样本 + 标签
# ============================================================================
def show_sample_images(train_set, n: int = 9, save_path: Optional[str] = None):
    """训练前随机展示 n(默认9) 张样本图与标签。

    Args:
        train_set: MNIST 数据集（或 Subset）。
        n:         展示张数，默认 9（3x3 网格）。
        save_path: 若给定，则把图保存为 PNG。
    Returns:
        (fig, images, labels) —— matplotlib Figure、图像数组 (n,28,28)、标签列表
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_set), size=n, replace=False)
    imgs, labels = [], []
    for i in idx:
        img, lab = train_set[int(i)]
        imgs.append(img)
        labels.append(int(lab))
    images = denormalize(torch.stack(imgs))
    fig = plot_sample_grid(images, labels, n=n)
    if save_path:
        save_fig(fig, save_path)          # 保存一份 PNG 供 README 展示
    return fig, images, labels


# ============================================================================
#  核心：fit() —— 生成器形式的训练主流程（供 train.py CLI 与 app.py 共用）
# ============================================================================
def fit(epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        stop_event: Optional[threading.Event] = None,
        seed: int = SEED,
        model_path: str = MODEL_PATH,
        data_dir: str = "./data",
        out_dir: str = OUTPUT_DIR,
        optimizer_name: str = "adam",
        scheduler_name: str = "cosine",
        num_workers: Optional[int] = None,
        post_report: bool = True,
        save_each_epoch: bool = True,
        yield_every_batch: bool = True,
        verbose: bool = True,
        model: Optional[nn.Module] = None) -> Iterator[dict]:
    """训练主流程（生成器）：每完成一部分工作就 yield 一次当前全部状态。

    yield 的字典结构（网页端直接消费）：
        {
          "status":   当前状态文字（含 epoch/总数、学习率、耗时、batch 进度），
          "history":  {"train_loss":[...], "val_loss":[...], "train_acc":[...], "val_acc":[...],
                       "lr":[...], "epoch_time":[...]},
          "log":      追加的日志文本（本轮新增），
          "artifacts":{名称: 图片文件路径}（训练结束后才有），
          "csv":      CSV 训练日志路径（训练结束后才有），
          "done":     是否全部结束,
          "stopped":  是否被用户中途停止,
        }

    好处：调用方（网页服务）只需把 history 交给 Plotly 重画，即可实现曲线动态刷新。
    """
    set_seed(seed)                                     # 固定随机种子，保证可复现
    device = get_device()                              # 自动选择 GPU / CPU
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [],
                                       "train_acc": [], "val_acc": [],
                                       "lr": [], "epoch_time": []}
    logs: List[str] = []
    artifacts: Dict[str, str] = {}

    def emit(status: str, done: bool = False, stopped: bool = False, log: str = "") -> dict:
        """统一构造 yield 内容；log 参数用于把多条日志合并成一次输出。"""
        if log:
            logs.append(log)
        return {"status": status, "history": {k: list(v) for k, v in history.items()},
                "log": "\n".join(logs), "artifacts": dict(artifacts),
                "csv": artifacts.get("history_csv", ""), "done": done, "stopped": stopped}

    # ---------------- 0) 准备数据 ----------------
    yield emit("⏳ 正在准备 MNIST 数据集（首次运行会自动下载到 ./data）...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size, data_dir=data_dir, num_workers=num_workers, verbose=verbose)
        train_set = train_loader.dataset
        test_set = test_loader.dataset
        if isinstance(train_set, Subset):
            vis_set = train_set.dataset                    # 展示样本时用完整 MNIST
        else:
            vis_set = train_set
    except Exception as exc:
        yield emit(f"❌ 数据加载失败：{exc}", done=True)
        return

    # ---------------- 1) 训练前：随机 9 张样本预览 ----------------
    try:
        fig, _, _ = show_sample_images(vis_set, n=9,
                                       save_path=os.path.join(out_dir, "samples_before_train.png"))
        artifacts["samples_before_train"] = os.path.abspath(os.path.join(out_dir, "samples_before_train.png"))
        plt.close(fig)
        yield emit("🖼️ 已生成训练前样本预览：outputs/samples_before_train.png（随机 9 张 + 标签）")
    except Exception as exc:
        yield emit(f"⚠️ 样本预览生成失败（不影响训练）：{exc}")

    # ---------------- 2) 准备模型 / 损失 / 优化器 ----------------
    if model is None:
        model = MnistCNN().to(device)
    else:
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()                       # 交叉熵（内部含 log_softmax）
    optimizer = build_optimizer(model, lr=lr, optimizer_name=optimizer_name)
    scheduler, is_per_batch = build_scheduler(optimizer, epochs=epochs, scheduler_name=scheduler_name)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    yield emit(f"🚀 开始训练：设备={device}，参数总量={n_params:,}，"
               f"epochs={epochs}，batch_size={batch_size}，lr={lr}，优化器={optimizer_name}")

    # ---------------- 3) 逐 epoch 训练 ----------------
    best_acc = 0.0
    stopped_by_user = False
    for epoch in range(1, epochs + 1):
        if stop_event is not None and stop_event.is_set():
            stopped_by_user = True
            break

        epoch_start = time.time()
        # 每个 batch 结束后回调，用于网页实时显示 batch 进度
        def on_batch(batch_idx: int, n_batches: int, avg_loss: float, acc: float,
                     _epoch=epoch, _start=epoch_start) -> None:
            if verbose:
                print(f"\r  epoch {_epoch}/{epochs} | batch {batch_idx}/{n_batches} "
                      f"| loss={avg_loss:.4f} | acc={acc:.2%}", end="", flush=True)

        try:
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch, total_epochs=epochs, stop_event=stop_event,
                progress_cb=on_batch if verbose else None,
                scheduler=scheduler, is_per_batch=is_per_batch)
        except KeyboardInterrupt:                      # 命令行 Ctrl+C 也算优雅停止
            stopped_by_user = True
            break
        except RuntimeError as exc:                    # 显存不足等运行时错误
            msg = str(exc)
            if "out of memory" in msg.lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                yield emit("❌ CUDA 显存不足，请把 batch size 调小（例如 32 或 64）后重试。", done=True)
            else:
                yield emit(f"❌ 训练出错：{msg}", done=True)
            return

        if train_metrics["stopped"]:
            stopped_by_user = True

        # ---- 每个 epoch 结束后在验证集上评估 ----
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None and not is_per_batch:
            scheduler.step()                            # 按 epoch 更新学习率

        epoch_time = time.time() - epoch_start
        history["train_loss"].append(round(train_metrics["loss"], 6))
        history["train_acc"].append(round(train_metrics["acc"], 6))
        history["val_loss"].append(round(val_metrics["loss"], 6))
        history["val_acc"].append(round(val_metrics["acc"], 6))
        history["lr"].append(round(optimizer.param_groups[0]["lr"], 8))
        history["epoch_time"].append(round(epoch_time, 3))

        # 记录最佳准确率（用于保存"最优模型"）
        best_acc = max(best_acc, val_metrics["acc"])
        if save_each_epoch:
            # 每个 epoch 都保存，网页端"手写预测"可以随时用到最新模型
            try:
                save_model(model, model_path, extra={"epoch": epoch, "best_acc": best_acc,
                                                     "history": history})
            except Exception as exc:
                print(f"[警告] 模型保存失败：{exc}")

        stage = "（已被用户停止，保存当前模型）" if stopped_by_user else ""
        status = (f"✅ Epoch {epoch}/{epochs} 完成 {stage}｜训练 loss={train_metrics['loss']:.4f} "
                  f"acc={train_metrics['acc']:.2%}｜验证 loss={val_metrics['loss']:.4f} "
                  f"acc={val_metrics['acc']:.2%}｜当前学习率={optimizer.param_groups[0]['lr']:.6f} "
                  f"｜本 epoch 耗时={format_seconds(epoch_time)}｜最佳验证准确率={best_acc:.2%}")
        if verbose:
            print(f"\r{status}")
        # 每结束一个 epoch 就 yield：网页的 loss/accuracy 曲线随之刷新
        yield emit(status)

        if stopped_by_user:
            break

    # ---------------- 4) 训练结束：保存模型 ----------------
    if not history["train_loss"]:
        yield emit("没有完成任何训练轮次，保留原有模型。", done=True, stopped=stopped_by_user)
        return

    try:
        saved_path = save_model(model, model_path, extra={"epoch": len(history["train_loss"]),
                                                         "best_acc": best_acc, "history": history})
        yield emit(f"💾 模型已保存：{saved_path}")
    except Exception as exc:
        yield emit(f"⚠️ 模型保存失败：{exc}")

    # ---------------- 5) 最终测试集评估 + 完整可视化报告 ----------------
    if post_report:
        yield emit("📊 正在测试集上评估并生成可视化报告（混淆矩阵 / 预测对比 / 错误样本 / CSV）...")
        try:
            artifacts.update(generate_post_training_report(model, test_set, history=history,
                                                           device=device, out_dir=out_dir))
        except Exception as exc:
            yield emit(f"⚠️ 报告生成部分失败：{exc}")

        # 用 matplotlib 再画一张训练曲线静态图，方便离线查看
        try:
            art = save_fig(plot_training_curves(history), os.path.join(out_dir, "training_curves.png"))
            artifacts["training_curves"] = art
        except Exception as exc:
            print(f"[警告] 训练曲线保存失败：{exc}")

    final_status = (f"🎉 训练结束{'（中途停止）' if stopped_by_user else ''}！"
                    f"共 {len(history['train_loss'])} 个 epoch，最佳验证准确率 {best_acc:.2%}，"
                    f"模型：{os.path.abspath(model_path)}")
    yield emit(final_status, done=True, stopped=stopped_by_user)


# ============================================================================
#  命令行入口：python train.py --epochs 5
# ============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="MNIST CNN 训练脚本（含实时可视化）")
    p.add_argument("--epochs", type=int, default=5, help="训练轮数（默认 5）")
    p.add_argument("--batch-size", type=int, default=64, help="批大小（默认 64）")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率（默认 0.001）")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--data-dir", type=str, default="./data", help="MNIST 数据目录")
    p.add_argument("--out-dir", type=str, default=OUTPUT_DIR, help="可视化图片输出目录")
    p.add_argument("--model-path", type=str, default=MODEL_PATH, help="模型保存路径")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader 子进程数")
    p.add_argument("--interactive-plot", action="store_true",
                   help="训练过程中额外用 Matplotlib 弹窗显示动态曲线（需要图形界面）")
    return p.parse_args()


def main() -> None:
    """命令行训练主函数：实时打印进度 + 可选弹窗动态曲线 + 结束后生成报告。"""
    args = parse_args()
    print("=" * 78)
    print(" MNIST 手写数字识别 —— 训练开始")
    print(f" 设备信息 / Device: ", end="")
    get_device()
    print("=" * 78)

    stop_event = threading.Event()
    live_fig = None
    live_ax = None

    # 可选：matplotlib 交互式弹窗（训练中每个 epoch 重画一次）
    if args.interactive_plot:
        try:
            plt.ion()
            live_fig, live_ax = plt.subplots(1, 2, figsize=(11, 4))

            def redraw(hist: Dict[str, List[float]]) -> None:
                for ax in live_ax:
                    ax.clear()
                ep = range(1, len(hist["train_loss"]) + 1)
                live_ax[0].plot(ep, hist["train_loss"], "o-", label="train")
                live_ax[0].plot(ep, hist["val_loss"], "s-", label="val")
                live_ax[0].set_title("Loss"); live_ax[0].legend(); live_ax[0].grid(alpha=.3)
                live_ax[1].plot(ep, np.array(hist["train_acc"]) * 100, "o-", label="train")
                live_ax[1].plot(ep, np.array(hist["val_acc"]) * 100, "s-", label="val")
                live_ax[1].set_title("Accuracy (%)"); live_ax[1].legend(); live_ax[1].grid(alpha=.3)
                live_fig.canvas.draw()
                live_fig.canvas.flush_events()
                plt.pause(0.05)
        except Exception as exc:
            print(f"[提示] 交互式弹窗不可用（{exc}），已自动改为命令行实时打印。")
            live_fig = None

    last_epochs = 0
    for state in fit(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     stop_event=stop_event, model_path=args.model_path,
                     data_dir=args.data_dir, out_dir=args.out_dir,
                     optimizer_name=args.optimizer, scheduler_name=args.scheduler,
                     num_workers=args.num_workers):
        # 用命令行运行时，实时打印状态文字；图表在 epoch 增加时重画
        if state["status"]:
            print(state["status"])
        if live_fig is not None and len(state["history"]["train_loss"]) != last_epochs:
            last_epochs = len(state["history"]["train_loss"])
            redraw(state["history"])
        if state["done"]:
            print("\n" + "-" * 78)
            print(state["log"].splitlines()[-1] if state["log"] else "训练结束")
            print("-" * 78)

    if live_fig is not None:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
