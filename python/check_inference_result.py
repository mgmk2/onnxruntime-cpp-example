"""
MIT License

Copyright (c) 2022 mgmk2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch
import torchvision

from output_resource import get_pytorch_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-l', '--label', type=str, required=True)
    args = parser.parse_args()

    image_filename = os.path.normpath(args.input)
    label_filename = os.path.normpath(args.label)

    # load image
    assert os.path.exists(image_filename)
    image = Image.open(image_filename)

    assert image.height == 224 and image.width == 224

    # load labels
    with open(label_filename, mode='r') as f:
        labels = [s.strip() for s in f.readlines()]

    # load model
    model = get_pytorch_model()
    model.eval()

    # preprocess image
    input_tensor = torchvision.transforms.functional.to_tensor(image)  # [0,1]に変換
    input_tensor = input_tensor.unsqueeze(0)  # CHW -> BCHW

    # run inference
    with torch.no_grad():
        results = model(input_tensor)
        result = results[0].numpy().copy()

    # show Top5
    indices = np.argsort(result)[::-1]
    for i in range(5):
        index = indices[i]
        print(f'{i + 1}: {labels[index]} {result[index]}')
