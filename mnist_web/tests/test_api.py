"""API regressions. Run from mnist_web: python -m unittest discover -s tests."""
import shutil
import uuid
from contextlib import contextmanager
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, Subset

import app
import train


@contextmanager
def test_directory():
    directory = app.ROOT / "tests" / ("run-" + uuid.uuid4().hex)
    directory.mkdir()
    assert directory.resolve().is_relative_to(app.ROOT.resolve())
    try:
        yield directory
    finally:
        assert directory.resolve().is_relative_to(app.ROOT.resolve())
        shutil.rmtree(directory)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_status_and_static_page(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertIn('drawing', self.client.get('/').text)
        info = self.client.get('/api/status').json()
        self.assertTrue(info['ready'])
        self.assertEqual(info['parameters'], 421834)

    def test_shutdown_waits_for_training_and_blocks_new_jobs(self):
        server = SimpleNamespace(should_exit=False)
        app.JOB_LOCK.acquire()
        try:
            with patch.object(app.app.state, 'server', server, create=True):
                self.assertEqual(self.client.post('/api/shutdown', headers={'origin':'https://other.example'}).status_code, 403)
                self.assertFalse(app.CLOSING.is_set())
                response = self.client.post('/api/shutdown', headers={'origin':'http://testserver'})
                self.assertEqual(response.json(), {'closing':True})
                self.assertTrue(app.STOP.is_set())
                self.assertFalse(server.should_exit)
                self.assertEqual(self.client.post('/api/train/start', json={}).status_code, 503)
                self.assertEqual(self.client.post('/api/shutdown').status_code, 200)
        finally:
            app.JOB_LOCK.release()
            deadline = time.monotonic() + 3
            while not server.should_exit and time.monotonic() < deadline:
                time.sleep(.01)
            app.CLOSING.clear()
            app.STOP.clear()
        self.assertTrue(server.should_exit)

    def test_browser_opens_only_after_server_ready(self):
        server = SimpleNamespace(should_exit=False, started=False)
        with patch('webbrowser.open') as browser, patch.object(app.time, 'sleep', side_effect=lambda _: setattr(server, 'started', True)):
            app.open_browser_when_ready(server, 'http://127.0.0.1:7860')
            browser.assert_called_once_with('http://127.0.0.1:7860')
        server.should_exit = True
        with patch('webbrowser.open') as browser:
            app.open_browser_when_ready(server, 'http://127.0.0.1:7860')
            browser.assert_not_called()

    def test_blank_and_bad_uploads(self):
        white = app.png_data(np.full((320, 320), 255, dtype=np.uint8))
        result = self.client.post('/api/predict', json={'image': white}).json()
        self.assertTrue(result['blank'])
        self.assertIsNone(result['prediction'])
        self.assertEqual(result['probabilities'], [0.] * 10)
        self.assertEqual(self.client.post('/api/predict', json={'image':'bad'}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', json={'image':'data:image/png;base64,!!!'}).status_code, 400)
        self.assertEqual(self.client.get('/api/sample?digit=12').status_code, 400)

    def test_predictions_use_real_model(self):
        correct = 0
        for digit in range(10):
            sample = self.client.get(f'/api/sample?digit={digit}').json()
            self.assertEqual(sample['label'], digit)
            result = self.client.post('/api/predict', json={'image':sample['image']}).json()
            self.assertFalse(result['blank'])
            self.assertEqual(len(result['probabilities']), 10)
            self.assertAlmostEqual(sum(result['probabilities']), 1, places=5)
            self.assertTrue(result['input'].startswith('data:image/png;base64,'))
            correct += result['prediction'] == digit
        self.assertGreaterEqual(correct, 8)

    def test_features_and_balanced_confusion(self):
        sample = self.client.get('/api/sample?digit=3').json()
        response = self.client.post('/api/analyze', json={'image':sample['image'],'confusion':True})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result['features']), 8)
        self.assertIn('MnistCNN', result['summary'])
        matrix = result['confusion']['matrix']
        self.assertEqual(len(matrix), 10)
        self.assertEqual([sum(row) for row in matrix], [100] * 10)

    def test_training_lock_and_stop(self):
        entered = threading.Event()

        def controlled_fit(**kwargs):
            entered.set()
            yield {'status':'正在训练', 'history':{}, 'log':'', 'done':False}
            if not kwargs['stop_event'].wait(3):
                raise AssertionError('Stop signal did not arrive')
            yield {'status':'已停止', 'history':{}, 'log':'', 'done':False, 'stopped':True}

        with patch.object(app, 'fit', controlled_fit):
            self.assertEqual(self.client.post('/api/train/start', json={'epochs':1}).status_code, 200)
            self.assertTrue(entered.wait(1))
            self.assertEqual(self.client.post('/api/train/start', json={'epochs':1}).status_code, 409)
            self.assertEqual(self.client.post('/api/train/stop').status_code, 200)
            deadline = time.monotonic() + 3
            while app.JOB_LOCK.locked() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertFalse(app.JOB_LOCK.locked())
            self.assertFalse(self.client.get('/api/train/status').json()['running'])
        self.assertEqual(self.client.post('/api/train/start', json={'epochs':0}).status_code, 422)

    def test_cancelling_before_first_epoch_keeps_weights(self):
        dataset = Subset(app.test_set(), range(16))
        loader = DataLoader(dataset, batch_size=8)
        stop = threading.Event()
        stop.set()
        with test_directory() as directory:
            assert Path(directory).resolve().is_relative_to(app.ROOT.resolve())
            model_path = Path(directory) / 'existing.pth'
            model_path.write_bytes(b'previous-model')
            with patch.object(train, 'get_dataloaders', return_value=(loader, loader, loader)):
                updates = list(train.fit(epochs=1, stop_event=stop, model_path=str(model_path),
                                         out_dir=directory, post_report=False, verbose=False))
            self.assertEqual(model_path.read_bytes(), b'previous-model')
            self.assertTrue(updates[-1]['stopped'])

    def test_real_training_on_small_subset(self):
        loader = DataLoader(Subset(app.test_set(), range(32)), batch_size=16)
        with test_directory() as directory:
            assert Path(directory).resolve().is_relative_to(app.ROOT.resolve())
            model_path = Path(directory) / 'test-model.pth'
            with patch.object(train, 'get_dataloaders', return_value=(loader, loader, loader)):
                updates = list(train.fit(epochs=1, model_path=str(model_path), out_dir=directory,
                                         post_report=False, verbose=False))
            self.assertTrue(model_path.exists())
            self.assertEqual(len(updates[-1]['history']['train_loss']), 1)
            self.assertTrue(updates[-1]['done'])


if __name__ == '__main__':
    unittest.main()
