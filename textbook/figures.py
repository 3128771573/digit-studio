"""Rebuild charts solely from recorded evidence; no torch or training needed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
E = HERE / 'evidence'


def charts():
    comparison = json.loads((E / 'comparison.json').read_text('utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    markers = ['o', 's', '^', 'D']
    for row, marker in zip(comparison['runs'], markers):
        name = row['model']
        history = json.loads((E / (name + '_history.json')).read_text('utf-8'))
        epochs = [h['epoch'] for h in history]
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot(epochs, [h['train_' + metric] for h in history],
                    marker=marker, linestyle='-', label=name + ' train')
            ax.plot(epochs, [h['val_' + metric] for h in history],
                    marker=marker, linestyle='--', label=name + ' val')
            ax.set(xlabel='epoch', ylabel=metric, xticks=epochs)
            ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(E / 'real_curves.png', dpi=240)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for row, marker in zip(comparison['runs'], markers):
        name = row['model']
        shifts = json.loads((E / (name + '_shift.json')).read_text('utf-8'))
        ax.plot([s['dx'] for s in shifts], [s['accuracy'] for s in shifts],
                marker=marker, label=name)
    ax.set(xlabel='horizontal shift (pixels)', ylabel='validation accuracy',
           xticks=[-4, -2, 0, 2, 4])
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(E / 'real_shift.png', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    charts()
