"""Observe fixed project stop edges, with all writes confined to teaching runs."""
import sys
import threading
import uuid
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from .common import ROOT, RUNS, seed_all, write_json


def main():
    sys.path.insert(0, str(ROOT / 'mnist_web'))
    import train
    seed_all()
    original = ROOT / 'mnist_web' / 'mnist_cnn.pth'
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    output = RUNS / ('edges-' + uuid.uuid4().hex)
    output.mkdir(parents=True)
    x, y = torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    actual_train = train.train_one_epoch
    cases = []
    for mode in ['before_epoch', 'zero_batch', 'partial_epoch', 'report']:
        stop = threading.Event()
        path = output / (mode + '.pth')
        # A sentinel proves whether the early cancellation preserves old bytes.
        path.write_bytes(b'old-checkpoint-sentinel')
        captured = {}
        if mode == 'before_epoch':
            stop.set()

        def controlled(*args, **kwargs):
            if mode == 'zero_batch':
                stop.set()
            if mode == 'partial_epoch':
                kwargs['progress_cb'] = lambda *a: stop.set()
            metrics = actual_train(*args, **kwargs)
            captured.update(metrics)
            return metrics

        def report(*args, **kwargs):
            stop.set()
            captured['report_called'] = True
            return {}

        with patch.object(train, 'get_dataloaders', return_value=(loader, loader, loader)), \
             patch.object(train, 'show_sample_images', side_effect=RuntimeError('skip preview')), \
             patch.object(train, 'train_one_epoch', side_effect=controlled), \
             patch.object(train, 'generate_post_training_report', side_effect=report):
            states = list(train.fit(epochs=1,
                model_path=str(path), out_dir=str(output), stop_event=stop,
                post_report=(mode == 'report'), verbose=False))
        changed = path.read_bytes() != b'old-checkpoint-sentinel'
        cases.append({'case': mode, 'checkpoint_replaced': changed,
            'history_epochs': len(states[-1]['history']['train_loss']),
            'metrics': captured, 'final_done': states[-1]['done'],
            'final_stopped': states[-1].get('stopped', False)})
    after = hashlib.sha256(original.read_bytes()).hexdigest()
    assert before == after
    assert not cases[0]['checkpoint_replaced']
    assert cases[1]['metrics']['n'] == 0 and cases[1]['checkpoint_replaced']
    assert cases[2]['metrics']['n'] == 2 and cases[2]['checkpoint_replaced']
    assert cases[3]['metrics']['report_called']
    write_json(ROOT / 'textbook' / 'evidence' / 'project_edges.json',
        {'baseline': '43b82e9eedae27ed9771f40df130e0e51ceed462',
         'method': 'controlled small-data calls with patched preparation/report',
         'original_weight_sha256_unchanged': before, 'cases': cases})
    print(cases)


if __name__ == '__main__':
    main()
