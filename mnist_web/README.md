# Digit Studio · 手写数字实验室

本地运行的 MNIST 手写数字识别工作台。使用独立 HTML Canvas 界面与 FastAPI 服务，识别和训练由 PyTorch 完成。
无需 Node.js、前端构建或远程字体，不依赖 Gradio。

## 启动与关闭

Windows 双击 `start.bat`，服务就绪后会自动用默认浏览器打开 http://127.0.0.1:7860 。
脚本优先使用项目根目录的 `.venv`。已有权重可直接识别，无需先训练。

也可以在项目根目录执行：

```powershell
cd mnist_web
..\.venv\Scripts\python.exe app.py
```

启动窗口需保持运行；关闭网页不会停止服务。点击网页右上角「关闭服务」并确认即可退出，也可以在启动窗口按 `Ctrl+C`。网页关闭服务时，如有训练任务，会先停止并保存模型。关闭后，再次双击 `start.bat` 即可重新启动。
页面只连接本机服务，打开网页不会自行启动后台，也未配置开机启动。

## 使用

- **手写识别**：鼠标、触控或手写笔写下 0–9。落笔后自动识别；支持笔迹粗细、撤销最近 20 次操作、清空、随机样本与指定数字样本。右侧展示真实预测和十类概率，下方展示模型实际看到的 28×28 输入。
- **模型训练**：调整轮数、每批样本数和学习率。页面展示训练/验证 Loss 与 Accuracy、日志和 CSV。后台同一时间只允许一个训练任务。停止后保存本轮已学到的内容；若尚未开始任何一轮，则保留旧权重。
- **模型探索**：使用当前笔迹、测试样本或上传图片；查看 8/16/32 个卷积特征通道和完整网络结构。可选混淆矩阵使用每类 100 张，共 1,000 张官方测试样本，行是真实类别，列是预测类别。

新训练从新模型开始，并非接续旧权重，训练会覆盖 `mnist_cnn.pth`。训练结束后自动重新加载模型用于识别。
顶部准确率取自已保存权重的历史验证记录；不是用户笔迹的实际准确率。
CPU 版可完成全部功能，GPU 在兼容的 PyTorch 环境下自动启用。

## 文件

| 文件 | 作用 |
| --- | --- |
| `app.py` | FastAPI 接口、模型加载、后台训练与服务入口 |
| `static/index.html` | 三个工作区与页面结构 |
| `static/style.css` | 桌面与窄屏布局、视觉样式 |
| `static/app.js` | Canvas 笔迹、撤销、预测队列、训练曲线与模型分析 |
| `model.py` | CNN、设备检测、权重存取 |
| `train.py` | 训练生成器、评估与报告 |
| `utils.py` | 数据、图像处理、静态报告 |
| `tests/test_api.py` | API 与训练回归测试 |
| `mnist_cnn.pth` | 当前权重 |
| `data/` | MNIST 数据 |
| `outputs/` | 训练报告、CSV 和界面预览 |

路径基于 `app.py` 所在目录，启动服务不依赖当前命令行工作目录。

## 新环境安装

使用 Python 3.10–3.13，在项目根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r mnist_web\requirements.txt
```

GPU 需安装与显卡兼容的 PyTorch 与 torchvision。本机当前安装的是 CPU 版本。

## 命令行训练与测试

```powershell
cd mnist_web
..\.venv\Scripts\python.exe train.py --epochs 5 --batch-size 64 --lr 0.001
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试包含真实模型预测、空白与非法图片、特征和均衡混淆矩阵、重复训练拦截、停止信号，以及临时权重上的小规模真实训练。
测试不会改写现有模型权重或训练报告。

## 设计参考

研究了 [Numberly](https://github.com/soaravant/Numberly) 的工作区、Canvas 与候选结果展示思路，以及 [Handwritten Digit Recognition](https://github.com/itzdineshx/Handwritten-Digit-Recognition-system) 的独立网页输入方式。
本项目的布局和接口重新实现，继续使用现有 PyTorch 模型，不迁移其 TensorFlow.js 权重。

## 常见问题

- 缺少 torch：确认使用根目录 `.venv` 的 Python。
- 页面提示服务未连接：先运行 `start.bat`，再刷新网页。
- 端口被占用：关闭旧服务，或修改 `app.py` 中 `uvicorn.Config(..., port=7860)` 及自动打开的网页地址。
- 图片识别不理想：写清楚、占满画布中间区域；数字概率是模型评分。
- 无模型：先到模型训练页面开始训练。
