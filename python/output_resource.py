import os
import argparse

import torch
import torchvision
from PIL import Image


def save_image_sample(output_dir: str):
    # 画像をダウンロード
    filename = os.path.join(output_dir, 'dog.jpg')
    torch.hub.download_url_to_file(
        'https://github.com/pytorch/hub/raw/master/images/dog.jpg', filename)

    # 224x224サイズに編集して別途保存
    image = Image.open(filename)
    image = torchvision.transforms.functional.resize(image, size=256)
    image = torchvision.transforms.functional.center_crop(image, output_size=224)
    image.save(os.path.join(output_dir, 'dog_input.png'))


def save_imagenet_label(output_dir: str):
    filename = os.path.join(output_dir, 'imagenet_classes.txt')
    torch.hub.download_url_to_file(
        'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', filename)


def get_pytorch_model():
    # ResNet50の学習済みモデルをロード
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    # 正規化処理とSoftmaxをモデルに連結して新しいモデルを作成
    return torch.nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
        torch.nn.Softmax(-1)
    )


def save_model(output_dir: str):
    # 変換するモデルを取得
    model = get_pytorch_model()

    # ダミーの入力
    dummy_input = torch.randn(1, 3, 224, 224)

    # ONNXフォーマットに変換して出力
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
