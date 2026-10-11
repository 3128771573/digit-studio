import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'textbook' / 'labs' / 'data'
RUNS = ROOT / 'textbook' / 'labs' / 'runs'
MEAN, STD = 0.1307, 0.3081


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def datasets(seed=42, full=False):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train = MNIST(DATA, train=True, download=True, transform=transform)
    test = MNIST(DATA, train=False, download=True, transform=transform)
    indices = torch.randperm(60000,
                            generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = indices[:5000], indices[5000:]
    if not full:
        val_idx, train_idx = val_idx[:512], train_idx[:2048]
        test = Subset(test, list(range(1000)))
    return Subset(train, train_idx.tolist()), Subset(train, val_idx.tolist()), test


def loader(dataset, batch=64, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=0)


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    errors = []
    for x, y in batches:
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(
            logits, y, reduction='sum').item()
        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred == y).sum().item()
        for image, actual, guessed in zip(x, y, pred):
            matrix[actual, guessed] += 1
            if actual != guessed and len(errors) < 16:
                errors.append((image[0].numpy(), int(actual), int(guessed)))
    return {'loss': loss_sum / total, 'accuracy': correct / total,
            'n': total, 'matrix': matrix.tolist()}, errors


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding='utf-8')
