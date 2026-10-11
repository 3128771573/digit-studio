"""Stage 2. Nearest class mean, in the same standardized space."""
import argparse
import torch
from torch import nn
from .common import ROOT, datasets, loader, evaluate, seed_all, write_json


class Template(nn.Module):
    def __init__(self, means):
        super().__init__()
        self.register_buffer('means', means)

    def forward(self, x):
        return -(x.flatten(1)[:, None] - self.means[None]).square().sum(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    seed_all()
    train, val, _ = datasets(full=args.full)
    sums, counts = torch.zeros(10, 784), torch.zeros(10)
    for x, y in loader(train):
        for c in range(10):
            sums[c] += x[y == c].flatten(1).sum(0)
            counts[c] += (y == c).sum()
    assert (counts > 0).all(), 'Every class needs an example.'
    model = Template(sums / counts[:, None])
    result, _ = evaluate(model, loader(val))
    # Squared distances are arbitrary scores; their CE is not a fair loss comparison.
    result.pop('loss')
    result.update(counts=counts.tolist(), split='validation', full=args.full)
    write_json(ROOT / 'textbook' / 'evidence' / 'template.json', result)
    print(result)


if __name__ == '__main__':
    main()
