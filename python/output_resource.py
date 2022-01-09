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

import os
import argparse

import torch
import torchvision
from PIL import Image


def save_image_sample(output_dir: str):
    # donwload image
    filename = os.path.join(output_dir, 'dog.jpg')
    torch.hub.download_url_to_file(
        'https://github.com/pytorch/hub/raw/master/images/dog.jpg', filename)

    # save 224x224 resized image
    image = Image.open(filename)
    image = torchvision.transforms.functional.resize(image, size=256)
    image = torchvision.transforms.functional.center_crop(image, output_size=224)
    image.save(os.path.join(output_dir, 'dog_input.png'))


def save_imagenet_label(output_dir: str):
    filename = os.path.join(output_dir, 'imagenet_classes.txt')
    torch.hub.download_url_to_file(
        'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', filename)


def get_pytorch_model():
    # load pretrained ResNet50 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    # create new model by concat normalization, model and softmax.
    return torch.nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
        torch.nn.Softmax(-1)
    )


def save_model(output_dir: str):
    # get pytorch model
    model = get_pytorch_model()

    # define dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # convert and save ONNX format model
    filename = os.path.join(output_dir, 'resnet50.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=['input0'],
        output_names=['output0'],
        dynamic_axes={'input0': {0: 'batch_size'}}
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_image_sample(output_dir)
    save_imagenet_label(output_dir)
    save_model(output_dir)
