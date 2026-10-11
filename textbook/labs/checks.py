"""Numerical mechanisms and complete train/save/load/predict path checks."""
import json
import shutil
import uuid
from contextlib import contextmanager
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .models import make_model
from .predict import prepare
from .common import ROOT, seed_all, write_json


@contextmanager
def scratch_directory():
    base = ROOT / 'textbook' / 'labs' / 'runs'
    path = base / ('check-' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    assert path.resolve().is_relative_to(base.resolve())
    try:
        yield path
    finally:
        assert path.resolve().is_relative_to(base.resolve())
        shutil.rmtree(path)


def numerical_examples():
    x = torch.tensor(2., dtype=torch.float64)
    w = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    u = torch.tensor([1., -1.], dtype=torch.float64, requires_grad=True)
    c = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    h = torch.relu(w * x + b)
    z = u * h + c
    loss = torch.nn.functional.cross_entropy(z[None], torch.tensor([1]))
    loss.backward()
    result = {'tiny': {'h': float(h.detach()), 'logits': z.detach().tolist(),
        'probabilities': z.detach().softmax(0).tolist(), 'loss': loss.item(),
        'dw': w.grad.item(), 'db': b.grad.item(), 'du': u.grad.tolist(),
        'dc': c.grad.tolist()}}
    with torch.no_grad():
        for p in [w, b, u, c]:
            p -= .1 * p.grad
        new_z = u * torch.relu(w * x + b) + c
        result['tiny'].update(new_w=w.item(), new_b=b.item(),
            new_u=u.tolist(), new_c=c.tolist(), new_logits=new_z.tolist(),
            new_loss=torch.nn.functional.cross_entropy(
                new_z[None], torch.tensor([1])).item())
    result['softmax_210'] = torch.tensor([2., 1., 0.],
                                       dtype=torch.float64).softmax(0).tolist()
    result['params'] = {name: sum(p.numel() for p in make_model(name).parameters())
                        for name in ['linear', 'mlp', 'cnn1', 'cnn2']}
    return result


class Checks(unittest.TestCase):
    def setUp(self):
        seed_all()

    def test_shapes_and_parameters(self):
        expected = {'linear': 7850, 'mlp': 50890, 'cnn1': 15770, 'cnn2': 9098}
        for name, count in expected.items():
            model = make_model(name)
            self.assertEqual(tuple(model(torch.zeros(3, 1, 28, 28)).shape), (3, 10))
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)

    def test_ce_gradient_and_finite_difference(self):
        z = torch.tensor([[2., 1., 0.], [0., .1, .2]],
                         dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 2])
        loss = torch.nn.functional.cross_entropy(z, y)
        loss.backward()
        expected = z.detach().softmax(1)
        expected[range(2), y] -= 1
        self.assertTrue(torch.allclose(z.grad, expected / 2))
        eps = 1e-6
        plus, minus = z.detach().clone(), z.detach().clone()
        plus[0, 1] += eps
        minus[0, 1] -= eps
        fd = (torch.nn.functional.cross_entropy(plus, y) -
              torch.nn.functional.cross_entropy(minus, y)) / (2 * eps)
        self.assertAlmostEqual(fd.item(), z.grad[0, 1].item(), places=8)

    def test_shared_convolution_gradient(self):
        conv = torch.nn.Conv1d(1, 1, 2, bias=False).double()
        x = torch.tensor([[[1., 2., 3.]]], dtype=torch.float64)
        conv(x).sum().backward()
        self.assertTrue(torch.equal(conv.weight.grad,
                                    torch.tensor([[[3., 5.]]], dtype=torch.float64)))

    def test_weighted_average(self):
        self.assertAlmostEqual((64 * 1.0 + 6 * 3.0) / 70, 82 / 70)
        self.assertNotAlmostEqual(82 / 70, (1 + 3) / 2)

    def test_tiny_update(self):
        e = numerical_examples()['tiny']
        self.assertAlmostEqual(e['loss'], 2.126928011, places=8)
        self.assertAlmostEqual(e['dw'], 3.523188312, places=8)
        self.assertLess(e['new_loss'], e['loss'])

    def test_train_save_load_predict(self):
        with scratch_directory() as temp:
            path = Path(temp)
            image = Image.new('RGB', (100, 100), 'white')
            ImageDraw.Draw(image).line([(30, 20), (70, 20), (45, 45),
                                       (70, 65), (30, 80)], fill='black', width=8)
            image.save(path / 'digit.png')
            display, x = prepare(path / 'digit.png')
            self.assertEqual(tuple(x.shape), (1, 1, 28, 28))
            self.assertEqual(display.size, (28, 28))
            model = make_model('cnn1')
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            before = model[0].weight.detach().clone()
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x), torch.tensor([3])).backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model[0].weight))
            torch.save(model.state_dict(), path / 'weights.pt')
            restored = make_model('cnn1')
            restored.load_state_dict(torch.load(path / 'weights.pt', weights_only=True))
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(x), restored(x)))
                self.assertAlmostEqual(restored(x).softmax(1).sum().item(), 1, places=6)
            Image.new('RGB', (28, 28), 'white').save(path / 'blank.png')
            with self.assertRaises(ValueError):
                prepare(path / 'blank.png')


if __name__ == '__main__':
    evidence = ROOT / 'textbook' / 'evidence'
    evidence.mkdir(exist_ok=True)
    write_json(evidence / 'math.json', numerical_examples())
    unittest.main()
