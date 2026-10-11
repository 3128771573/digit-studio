# 从一个 3 开始：Digit Studio 教材与实验

完整教材：[PDF](../output/pdf/Digit_Studio_Textbook.pdf)；[可编辑正文](book.md)。正文含 18 章、5 个附录、练习答案、学习路线、符号表、基线与修订记录。

项目基线是 `3128771573/digit-studio` 的 main，提交 `43b82e9eedae27ed9771f40df130e0e51ceed462`，2026-10-11 核对。所有应用行为引用该固定版本。教学包和应用的模型、输出分开；本次不改应用代码、权重或报告。

## 跑基础实验

在仓库根目录，Windows PowerShell：

```powershell
python -m venv .venv
$py = ".venv\Scripts\python.exe"
& $py -m pip install -r mnist_web/requirements.txt
& $py -m textbook.labs.inspect_data
& $py -m textbook.labs.template
& $py -m textbook.labs.checks
& $py -m textbook.labs.train --model linear --run my_linear
& $py -m textbook.labs.train --model mlp --run my_mlp
& $py -m textbook.labs.train --model cnn1 --run my_cnn1
& $py -m textbook.labs.train --model cnn2 --run my_cnn2
& $py -m textbook.labs.evaluate --run my_cnn2
& $py -m textbook.labs.shift --run my_cnn2
$img = "textbook/labs/examples/synthetic_three.png"
& $py -m textbook.labs.predict --run my_cnn2 --image $img
```

macOS/Linux 使用 `.venv/bin/python` 替换 `& $py`。每次新 PowerShell 窗口重新定义 `$py`。已有 .venv 时不用重建。MNIST 首次自动下载约 11 MB，独立缓存到 `textbook/labs/data`；已有原应用缓存可用 `inspect_data --source-cache mnist_web/data` 复制。

默认 CPU 4 线程、种子 42、训练/验证/测试子集 2048/512/1000、2 epochs、Adam lr=.001、无增强。`train --full` 使用 55000/5000；仍须明确选 epochs。测试只有在验证选择冻结后，用 `evaluate --run <name> --split test` 执行。不同模型用不同 run 名；目录已存在会拒绝覆盖。

`runs/<name>` 保存 best.pt、last.pt、配置、划分、历史与评价。基础 checkpoint 用于推理，未保存完整优化器/随机源状态，不能严格断点续训。仅加载可信权重。`evidence` 是本次真实记录与轻量图表，不包含下载数据。

`run_verified` 是本次自动快速验证脚本，依赖原项目缓存，且 verified_* 目录必须尚不存在。首次全新环境先按单阶段命令运行即可。`project_edges` 在独立临时模型与 4 张数据上检查基线停止边界，不覆盖项目权重。结果范围、环境和未完成项均见教材附录 E。

## PDF 重建

```powershell
python -m venv .pdfvenv
$pdfpy = ".pdfvenv\Scripts\python.exe"
& $pdfpy -m pip install -r textbook/pdf_requirements.txt
& $pdfpy textbook/build_book.py
& $pdfpy textbook/qa_pdf.py
```

重建只需要正文、生成器和已有 evidence，不需要 torch 或重新训练。中文使用可嵌入 TrueType 字体：Windows 默认 SimHei；其他系统设置 `TEXTBOOK_CJK_FONT` 为自己的中文 TTF。MathText 生成矢量公式，机制图使用 PDF 矢量路径。PDF 正文 11 pt、A4 镜像页边距、内部目录与书签；双面打印 100% 实际大小、长边翻页。

结构检查不代替视觉检查。可用 Poppler：`pdftoppm -png -r 100 output/pdf/Digit_Studio_Textbook.pdf tmp/pdfs/textbook/page`，逐页检查。若重新训练后想更新书中结果，先按同样的验证选择规则更新 evidence，再构建并重新检查；保留旧证据，别覆盖后假装没有发生变化。
