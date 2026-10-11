"""Controlled validation experiment: zero-filled shifts, no wraparound."""
import argparse
import torch
from .common import RUNS, datasets, loader, evaluate, write_json, MEAN, STD
from .evaluate import load_checkpoint


def shifted(batches, dx):
    for x, y in batches:
        out = torch.full_like(x, -MEAN / STD)
        if dx > 0:
            out[:, :, :, dx:] = x[:, :, :, :-dx]
        elif dx < 0:
            out[:, :, :, :dx] = x[:, :, :, -dx:]
        else:
            out = x
        yield out, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    run = RUNS / args.run
    model, ckpt = load_checkpoint(run / 'best.pt')
    _, val, _ = datasets(ckpt['config']['seed'], ckpt['config']['full'])
    result = []
    for dx in [-4, -2, 0, 2, 4]:
        metrics, _ = evaluate(model, shifted(loader(val), dx))
        result.append({'dx': dx, 'accuracy': metrics['accuracy'], 'n': metrics['n']})
    write_json(run / 'shift.json', result)
    print(result)


if __name__ == '__main__':
    main()
