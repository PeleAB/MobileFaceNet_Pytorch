import os
import argparse
import glob
import random
import shutil
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import onnx
import onnxruntime
from onnxruntime import quantization
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnx import version_converter
import tensorflow as tf
from tqdm import tqdm


def change_opset(input_model: str, new_opset: int = 15) -> str:
    """Upgrade the ONNX opset of *input_model* in place."""
    model = onnx.load(input_model)
    current = model.opset_import[0].version
    if current == new_opset:
        return input_model
    converted = version_converter.convert_version(model, new_opset)
    tmp = input_model + '.tmp'
    onnx.save(converted, tmp)
    onnxruntime.InferenceSession(tmp)  # check
    os.replace(tmp, input_model)
    return input_model


def create_calibration_dataset(dataset_path: str, samples_per_class: int = 100) -> str:
    """Create a smaller dataset for calibration."""
    target = os.path.join(os.path.dirname(dataset_path), 'calibration_' + os.path.basename(dataset_path))
    if not os.path.exists(target):
        os.makedirs(target)
    for class_dir in tqdm(next(os.walk(dataset_path))[1]):
        images = (glob.glob(os.path.join(dataset_path, class_dir, '*.jpg')) +
                  glob.glob(os.path.join(dataset_path, class_dir, '*.png')) +
                  glob.glob(os.path.join(dataset_path, class_dir, '*.jpeg')))
        random.shuffle(images)
        for img in images[:samples_per_class]:
            shutil.copy2(img, target)
    return target


TORCH_MEANS = [0.485, 0.456, 0.406]
TORCH_STD = [0.224, 0.224, 0.224]


def preprocess_image_batch(folder: str, h: int, w: int, limit: int = 0) -> np.ndarray:
    files = os.listdir(folder)
    if limit > 0:
        files = files[:limit]
    batch = []
    for f in files:
        img_path = os.path.join(folder, f)
        img = tf.keras.utils.load_img(img_path, color_mode='rgb', target_size=(w, h), interpolation='nearest')
        arr = np.array([tf.keras.utils.img_to_array(img)])
        arr = -1 + arr / 127.5
        arr = arr.transpose((0, 3, 1, 2))
        batch.append(arr)
    return np.stack(batch, axis=0)


def preprocess_random_images(h: int, w: int, c: int, count: int = 400) -> np.ndarray:
    batch = []
    for _ in range(count):
        vals = np.random.uniform(0, 1, c*h*w).astype('float32')
        batch.append(vals.reshape(1, c, h, w))
    return np.concatenate(np.expand_dims(batch, axis=0), axis=0)


class ImageNetDataReader(CalibrationDataReader):
    def __init__(self, folder: Optional[str], model_path: str):
        session = onnxruntime.InferenceSession(model_path, None)
        (_, c, h, w) = session.get_inputs()[0].shape
        if folder:
            self.data = preprocess_image_batch(folder, h, w, limit=0)
        else:
            self.data = preprocess_random_images(h, w, c)
        self.input_name = session.get_inputs()[0].name
        self.enum_data = None

    def get_next(self) -> Optional[Dict[str, List[float]]]:
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: d} for d in self.data])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def quantize_model(input_model: str, dataset: Optional[str], output_model: str) -> None:
    change_opset(input_model, 15)
    if dataset:
        calib_dataset = create_calibration_dataset(dataset, samples_per_class=200)
    else:
        calib_dataset = None
    dr = ImageNetDataReader(calib_dataset, input_model)
    infer_model = os.path.splitext(input_model)[0] + '_infer.onnx'
    quantization.quant_pre_process(input_model_path=input_model, output_model_path=infer_model, skip_optimization=False)
    quantize_static(
        infer_model,
        output_model,
        dr,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        reduce_range=True,
        extra_options={'WeightSymmetric': True, 'ActivationSymmetric': False}
    )
    print(f'Quantized model saved to {output_model}')


def main():
    parser = argparse.ArgumentParser(description='Static int8 quantization for ONNX models')
    parser.add_argument('--input', required=True, help='Path to fp32 ONNX model')
    parser.add_argument('--dataset', help='Path to dataset for calibration')
    parser.add_argument('--output', default='model_int8.onnx', help='Output quantized model path')
    args = parser.parse_args()
    quantize_model(args.input, args.dataset, args.output)


if __name__ == '__main__':
    main()
