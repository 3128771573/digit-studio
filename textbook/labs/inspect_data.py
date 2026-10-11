"""Stage 1. Copy an existing cache if requested, then inspect real MNIST."""
import argparse
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import DATA, ROOT, datasets, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache', type=str)
    args = parser.parse_args()
    if args.source_cache:
        source = ROOT / args.source_cache / 'MNIST' / 'raw'
        shutil.copytree(source, DATA / 'MNIST' / 'raw', dirs_exist_ok=True)
    train, val, test = datasets()
    output = ROOT / 'textbook' / 'evidence'
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for i, ax in enumerate(axes.flat):
        x, y = train[i]
        ax.imshow(x[0] * .3081 + .1307, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'label={y}')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'mnist_samples.png', dpi=180)
    plt.close(fig)
    x, y = train[0]
    info = {'shape': list(x.shape), 'dtype': str(x.dtype), 'label': y,
            'normalized_min': float(x.min()), 'normalized_max': float(x.max()),
            'quick_sizes': [len(train), len(val), len(test)],
            'first_pixels_5x5': (x[0, :5, :5] * .3081 + .1307).tolist()}
    write_json(output / 'data.json', info)
    print(info)


if __name__ == '__main__':
    main()
