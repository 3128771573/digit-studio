"""Evaluate a selected checkpoint once; produce auditable figures."""
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RUNS, datasets, loader, evaluate, write_json, seed_all
from .models import make_model


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(checkpoint['config']['model'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = p.parse_args()
    run = RUNS / args.run
    seed_all()
    model, ckpt = load_checkpoint(run / 'best.pt')
    _, val, test = datasets(ckpt['config']['seed'], ckpt['config']['full'])
    data = val if args.split == 'validation' else test
    metrics, errors = evaluate(model, loader(data))
    metrics.update(split=args.split, checkpoint_epoch=ckpt['epoch'])
    write_json(run / (args.split + '.json'), metrics)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(metrics['matrix'], cmap='Greys')
    for i, row in enumerate(metrics['matrix']):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha='center', va='center', fontsize=7,
                    color='white' if count > max(map(max, metrics['matrix'])) / 2
                    else 'black')
    ax.set(xlabel='predicted', ylabel='actual', xticks=range(10), yticks=range(10))
    fig.tight_layout()
    fig.savefig(run / (args.split + '_matrix.png'), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < len(errors):
            image, actual, pred = errors[i]
            ax.imshow(image * .3081 + .1307, cmap='gray')
            ax.set_title(f'{actual} -> {pred}', fontsize=10)
    fig.tight_layout()
    fig.savefig(run / (args.split + '_errors.png'), dpi=180)
    plt.close(fig)
    print(metrics)


if __name__ == '__main__':
    main()
