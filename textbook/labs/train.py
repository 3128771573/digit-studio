"""Stages 3-6. Each run owns its outputs; no application paths are written."""
import argparse
import copy
import hashlib
import platform
import time
import torch
import torchvision
import numpy as np
from PIL import __version__ as pillow_version
from .common import RUNS, datasets, loader, evaluate, seed_all, write_json
from .models import make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['linear', 'mlp', 'cnn1', 'cnn2'],
                   default='linear')
    p.add_argument('--run', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--full', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0:
        p.error('epochs, batch and lr must be positive')
    run = RUNS / args.run
    if run.parent.resolve() != RUNS.resolve():
        p.error('run must be a simple directory name')
    run.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    train, val, _ = datasets(args.seed, args.full)
    model = make_model(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_batches = loader(train, args.batch, True, args.seed)
    history, best, best_state, best_epoch = [], -1.0, None, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
        metrics, _ = evaluate(model, loader(val))
        row = {'epoch': epoch, 'train_loss': loss_sum / total,
               'train_accuracy': correct / total, 'val_loss': metrics['loss'],
               'val_accuracy': metrics['accuracy']}
        history.append(row)
        if metrics['accuracy'] > best:
            best, best_epoch = metrics['accuracy'], epoch
            best_state = copy.deepcopy(model.state_dict())
        print(row, flush=True)
    config = vars(args) | {'device': 'cpu', 'threads': 4,
        'optimizer': 'Adam', 'weight_decay': 0.0, 'augmentation': False,
        'normalization': {'mean': .1307, 'std': .3081},
        'input': [1, 28, 28], 'classes': list(range(10)),
        'preprocessing': 'white-paper-bbox20-v1',
        'parameters': sum(p.numel() for p in model.parameters()),
        'sizes': [len(train), len(val)], 'seconds': time.perf_counter() - started,
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
        'pillow': pillow_version}
    split = {'train': train.indices, 'validation': val.indices}
    write_json(run / 'split.json', split)
    config['split_sha256'] = hashlib.sha256(
        (run / 'split.json').read_bytes()).hexdigest()
    torch.save({'state_dict': best_state, 'config': config,
                'epoch': best_epoch, 'validation_accuracy': best}, run / 'best.pt')
    torch.save({'state_dict': model.state_dict(), 'config': config,
                'epoch': args.epochs}, run / 'last.pt')
    write_json(run / 'config.json', config)
    write_json(run / 'history.json', history)
    print('Saved best.pt and last.pt; test set has not been evaluated.')


if __name__ == '__main__':
    main()
