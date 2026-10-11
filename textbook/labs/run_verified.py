"""Execute the quick CPU curriculum and collect genuine evidence."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from .common import ROOT, RUNS, write_json


def run(module, *args):
    result = subprocess.run([sys.executable, '-m', 'textbook.labs.' + module,
                             *args], cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    (ROOT / 'textbook' / 'evidence' / (module + '_' +
        ('_'.join(args).replace('/', '_') or 'default') + '.log')).write_text(
            result.stdout + result.stderr, encoding='utf-8')
    print(module, args, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def main():
    evidence = ROOT / 'textbook' / 'evidence'
    evidence.mkdir(exist_ok=True)
    run('inspect_data', '--source-cache', 'mnist_web/data')
    run('checks')
    run('template')
    rows = []
    for name in ['linear', 'mlp', 'cnn1', 'cnn2']:
        run_name = 'verified_' + name
        run('train', '--model', name, '--run', run_name, '--epochs', '2')
        run('evaluate', '--run', run_name)
        run('shift', '--run', run_name)
        folder = RUNS / run_name
        config = json.loads((folder / 'config.json').read_text('utf-8'))
        metrics = json.loads((folder / 'validation.json').read_text('utf-8'))
        rows.append({'model': name, 'parameters': config['parameters'],
                     'seconds': config['seconds'], 'validation': metrics})
        for file in folder.iterdir():
            if file.suffix in ['.json', '.png']:
                shutil.copy2(file, evidence / (name + '_' + file.name))
    # Freeze selection using validation only, then evaluate test once.
    chosen = max(rows, key=lambda row: row['validation']['accuracy'])['model']
    run('evaluate', '--run', 'verified_' + chosen, '--split', 'test')
    for file in (RUNS / ('verified_' + chosen)).glob('test*'):
        shutil.copy2(file, evidence / (chosen + '_' + file.name))
    example = ROOT / 'textbook' / 'labs' / 'examples'
    example.mkdir(exist_ok=True)
    image = Image.new('RGB', (160, 160), 'white')
    ImageDraw.Draw(image).line([(45, 30), (105, 30), (115, 55), (90, 80),
                               (115, 100), (100, 130), (45, 130)],
                              fill='black', width=12, joint='curve')
    image.save(example / 'synthetic_three.png')
    run('predict', '--run', 'verified_' + chosen,
        '--image', 'textbook/labs/examples/synthetic_three.png')
    for filename in ['prediction.json', 'processed_input.png']:
        shutil.copy2(RUNS / ('verified_' + chosen) / filename, evidence / filename)
    write_json(evidence / 'comparison.json', {'runs': rows,
        'selected_on_validation': chosen, 'test_evaluations': 1,
        'scope': 'quick CPU: 2048 train, 512 validation, 1000 test; 2 epochs'})


if __name__ == '__main__':
    main()
