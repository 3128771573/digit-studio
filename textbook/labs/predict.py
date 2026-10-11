"""Stage 7. One digit, dark ink on white paper; always save processed input."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from .common import RUNS, ROOT, MEAN, STD, write_json, seed_all
from .evaluate import load_checkpoint


def prepare(path):
    with Image.open(path) as source:
        rgba = source.convert('RGBA')
        paper = Image.new('RGBA', rgba.size, 'white')
        paper.alpha_composite(rgba)
        ink = 1.0 - np.asarray(ImageOps.grayscale(paper), dtype=np.float32) / 255
    ink[ink <= .12] = 0
    rows, cols = np.nonzero(ink)
    if not len(rows):
        raise ValueError('Blank image after thresholding')
    crop = ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = crop.shape
    size = (max(1, round(w * 20 / max(h, w))),
            max(1, round(h * 20 / max(h, w))))
    scaled = Image.fromarray((crop * 255).astype('uint8')).resize(
        size, Image.Resampling.LANCZOS)
    display = Image.new('L', (28, 28), 0)
    display.paste(scaled, ((28 - size[0]) // 2, (28 - size[1]) // 2))
    tensor = torch.from_numpy(np.asarray(display).copy()).float() / 255
    return display, ((tensor - MEAN) / STD)[None, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--image', required=True)
    args = p.parse_args()
    seed_all()
    run = RUNS / args.run
    model, ckpt = load_checkpoint(run / 'best.pt')
    assert ckpt['config']['preprocessing'] == 'white-paper-bbox20-v1'
    display, tensor = prepare(Path(args.image))
    display.save(run / 'processed_input.png')
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0]
    result = {'prediction': int(probabilities.argmax()),
              'probabilities': probabilities.tolist(),
              'input_shape': list(tensor.shape), 'source': Path(args.image).name}
    write_json(run / 'prediction.json', result)
    print(result)


if __name__ == '__main__':
    main()
