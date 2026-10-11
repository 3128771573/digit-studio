"""Digit Studio: local Canvas UI with a PyTorch/FastAPI backend."""
import base64
import copy
import io
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import count_parameters, get_device, load_model
from train import evaluate, fit, predict_batch
from utils import MNIST_MEAN, MNIST_STD, get_transforms, model_summary, preprocess_uploaded_image

ROOT = Path(__file__).resolve().parent
DEVICE = get_device()
MODEL = None
META = {}
MODEL_LOCK = threading.RLock()
JOB_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
STOP = threading.Event()
CLOSING = threading.Event()
LIFECYCLE_LOCK = threading.Lock()
TRAINING = {'running': False, 'status': '准备就绪', 'history': {}, 'log': '', 'epochs': 5, 'csv': False}
TEST_SET = None
DATA_LOCK = threading.Lock()


def reload_model():
    global MODEL, META
    model, meta = load_model(str(ROOT / 'mnist_cnn.pth'), DEVICE)
    with MODEL_LOCK:
        MODEL, META = model, meta


def test_set():
    global TEST_SET
    with DATA_LOCK:
        if TEST_SET is None:
            TEST_SET = MNIST(str(ROOT / 'data'), train=False, download=True, transform=get_transforms(False))
    return TEST_SET


@asynccontextmanager
async def lifespan(_app):
    # Small single-image CNN inference benefits from avoiding a large CPU thread pool.
    if DEVICE.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        reload_model()
    except Exception as exc:
        print(f'[模型] {exc}')
    yield
    STOP.set()


app = FastAPI(title='Digit Studio', lifespan=lifespan)
app.mount('/assets', StaticFiles(directory=ROOT / 'static'), name='assets')


class ImageRequest(BaseModel):
    image: str = Field(max_length=6_000_000)
    auto_center: bool = True


class AnalysisRequest(ImageRequest):
    channels: Literal[8, 16, 32] = 8
    confusion: bool = False


class TrainingRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=30)
    batch_size: Literal[32, 64, 128, 256] = 64
    lr: float = Field(default=.001, ge=.00001, le=.1)
    optimizer: Literal['adam', 'sgd', 'rmsprop'] = 'adam'
    scheduler: Literal['cosine', 'step', 'none'] = 'cosine'


def decode_image(data):
    try:
        if not data.startswith('data:image/') or ',' not in data:
            raise ValueError('不是图片数据')
        raw = base64.b64decode(data.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 4_000_000:
                raise ValueError('图片过大，请使用小于 400 万像素的图片')
            # Composite transparent uploads onto white, matching the drawing canvas.
            rgba = source.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            background.alpha_composite(rgba)
            return np.asarray(background.convert('L'))
    except (ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f'无法读取图片：{exc}') from exc


def png_data(array):
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def infer(array, auto_center=True):
    if float(np.ptp(array.astype(np.float32))) < 12.75:
        return {'blank': True, 'probabilities': [0.] * 10, 'prediction': None, 'input': None, 'elapsed_ms': 0}
    started = time.perf_counter()
    display, tensor = preprocess_uploaded_image(array, auto_center=auto_center)
    with MODEL_LOCK, torch.inference_mode():
        if MODEL is None:
            raise HTTPException(409, '没有可用模型，请先训练模型。')
        preds, probs = predict_batch(MODEL, tensor, DEVICE)
    probabilities = probs[0].cpu().tolist()
    return {'blank': False, 'prediction': int(preds.item()), 'probabilities': probabilities,
            'input': png_data(display), 'elapsed_ms': round((time.perf_counter() - started) * 1000, 1)}


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html')


@app.get('/api/status')
def status():
    with MODEL_LOCK:
        accuracy = META.get('best_acc')
        return {'ready': MODEL is not None, 'device': str(DEVICE).upper(),
                'parameters': count_parameters(MODEL) if MODEL is not None else None,
                'validation_accuracy': accuracy if isinstance(accuracy, (float, int)) else None,
                'epoch': META.get('epoch'), 'architecture': 'CNN · 2 个卷积层'}


@app.post('/api/predict')
def predict(data: ImageRequest):
    return infer(decode_image(data.image), data.auto_center)


@app.get('/api/sample')
def sample(digit: int | None = None):
    if digit is not None and digit not in range(10):
        raise HTTPException(400, '数字应在 0–9 之间')
    dataset = test_set()
    choices = np.where(np.asarray(dataset.targets) == digit)[0] if digit is not None else np.arange(len(dataset))
    idx = int(np.random.choice(choices))
    image, label = dataset[idx]
    pixels = (image.squeeze().numpy() * MNIST_STD + MNIST_MEAN).clip(0, 1)
    # Display sample as dark ink on white paper, just like handwriting.
    return {'image': png_data(1 - pixels), 'label': int(label), 'index': idx}


@app.post('/api/analyze')
def analyze(data: AnalysisRequest):
    array = decode_image(data.image)
    result = infer(array, data.auto_center)
    if result['blank']:
        raise HTTPException(400, '图片为空，请写一个数字或选择测试样本。')
    _, tensor = preprocess_uploaded_image(array, auto_center=data.auto_center)
    with MODEL_LOCK, torch.inference_mode():
        model = MODEL
        features = torch.relu(model.bn1(model.conv1(tensor.to(DEVICE))))[0].cpu().numpy()
        summary = model_summary(model, device=DEVICE)
    maps = []
    for m in features[:data.channels]:
        normalized = (m - m.min()) / (float(np.ptp(m)) + 1e-8)
        maps.append(png_data(normalized))
    result.update(features=maps, summary=summary, confusion=None)
    if data.confusion:
        # Use a balanced, deterministic subset; this is labelled explicitly in the UI.
        dataset = test_set()
        from torch.utils.data import Subset
        indices = []
        labels = np.asarray(dataset.targets)
        for digit in range(10):
            indices.extend(np.where(labels == digit)[0][:100].tolist())
        loader = DataLoader(Subset(dataset, indices), batch_size=256, shuffle=False)
        with MODEL_LOCK:
            metrics = evaluate(model, loader, torch.nn.CrossEntropyLoss(), DEVICE, collect_predictions=True)
        matrix = np.zeros((10, 10), dtype=int)
        for actual, predicted in zip(metrics['labels'], metrics['preds']):
            matrix[int(actual), int(predicted)] += 1
        result['confusion'] = {'matrix': matrix.tolist(), 'accuracy': metrics['acc'], 'samples': len(indices)}
    return result


def training_worker(config):
    try:
        for update in fit(epochs=config.epochs, batch_size=config.batch_size, lr=config.lr,
                          optimizer_name=config.optimizer, scheduler_name=config.scheduler,
                          num_workers=0, stop_event=STOP, model_path=str(ROOT / 'mnist_cnn.pth'),
                          data_dir=str(ROOT / 'data'), out_dir=str(ROOT / 'outputs')):
            with STATE_LOCK:
                TRAINING.update(status=update['status'], history=update['history'], log=update['log'],
                                csv=bool(update.get('csv')), stopped=update.get('stopped', False))
            if update.get('done') and (ROOT / 'mnist_cnn.pth').exists():
                reload_model()
    except Exception as exc:
        traceback.print_exc()
        with STATE_LOCK:
            TRAINING['status'] = f'训练失败：{exc}'
    finally:
        with STATE_LOCK:
            TRAINING['running'] = False
        STOP.clear()
        JOB_LOCK.release()


@app.post('/api/train/start')
def start_training(config: TrainingRequest):
    with LIFECYCLE_LOCK:
        if CLOSING.is_set():
            raise HTTPException(503, '服务正在关闭。')
        return launch_training(config)


def launch_training(config):
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(409, '已有训练正在运行，请等待结束或停止。')
    STOP.clear()
    with STATE_LOCK:
        TRAINING.update(running=True, status='正在准备训练数据…', history={}, log='',
                        epochs=config.epochs, csv=False, stopped=False)
    try:
        threading.Thread(target=training_worker, args=(config,), daemon=True).start()
    except Exception:
        JOB_LOCK.release()
        with STATE_LOCK:
            TRAINING['running'] = False
        raise
    return {'started': True}


@app.post('/api/train/stop')
def stop_training():
    with STATE_LOCK:
        if not TRAINING['running']:
            return {'status': '当前没有正在运行的训练任务。'}
        STOP.set()
        TRAINING['status'] = '正在停止，当前批次结束后保存模型…'
        return {'status': TRAINING['status']}


@app.get('/api/train/status')
def training_status():
    with STATE_LOCK:
        return copy.deepcopy(TRAINING)


@app.get('/api/train/csv')
def download_csv():
    path = ROOT / 'outputs' / 'training_log.csv'
    if not path.exists():
        raise HTTPException(404, '请先完成一次训练。')
    return FileResponse(path, filename='training_log.csv', media_type='text/csv')


def finish_shutdown(server):
    # Let the current training batch finish and persist weights before exiting.
    with JOB_LOCK:
        server.should_exit = True


@app.post('/api/shutdown')
def shutdown_service(request: Request, background_tasks: BackgroundTasks):
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        raise HTTPException(403, '请从本机服务页面关闭。')
    server = getattr(app.state, 'server', None)
    if server is None:
        raise HTTPException(503, '此启动方式不支持网页关闭，请在终端按 Ctrl+C。')
    with LIFECYCLE_LOCK:
        if not CLOSING.is_set():
            CLOSING.set()
            STOP.set()
            # Respond first; waiting for training must not block the browser.
            background_tasks.add_task(
                lambda: threading.Thread(target=finish_shutdown, args=(server,), daemon=True).start())
    return {'closing': True}


def open_browser_when_ready(server, url, timeout=60):
    import webbrowser
    deadline = time.monotonic() + timeout
    while not server.should_exit and time.monotonic() < deadline:
        if server.started:
            webbrowser.open(url)
            return
        time.sleep(.1)


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description='Digit Studio 本机服务')
    parser.add_argument('--open-browser', action='store_true', help='服务就绪后打开默认浏览器')
    args = parser.parse_args()
    CLOSING.clear()
    print('Digit Studio · http://127.0.0.1:7860 · Ctrl+C 停止服务')
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=7860, log_level='warning'))
    app.state.server = server
    if args.open_browser:
        threading.Thread(target=open_browser_when_ready,
                         args=(server, 'http://127.0.0.1:7860'), daemon=True).start()
    try:
        server.run()
    finally:
        server.should_exit = True
        print('服务已关闭。')


if __name__ == '__main__':
    main()
