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

    # 画像の読み込み
    assert os.path.exists(image_filename)
    image = Image.open(image_filename)

    assert image.height == 224 and image.width == 224

    # ラベルの読み込み
    with open(label_filename, mode='r') as f:
        labels = [s.strip() for s in f.readlines()]

    # モデルの読み込み
    model = get_pytorch_model()
    model.eval()

    # 画像の前処理
    input_tensor = torchvision.transforms.functional.to_tensor(image)  # [0,1]に変換
    input_tensor = input_tensor.unsqueeze(0)  # CHW -> BCHW

    # 推論を実行
    with torch.no_grad():
        results = model(input_tensor)
        result = results[0].numpy().copy()

    # Top5を表示
    indices = np.argsort(result)[::-1]
    for i in range(5):
        index = indices[i]
        print(f'{i + 1}: {labels[index]} {result[index]}')
