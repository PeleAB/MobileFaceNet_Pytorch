import argparse
import os
import numpy as np
import torch
from core import model
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import tempfile


def load_model(ckpt_path):
    net = model.MobileFacenet()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    return net


def export_onnx(net, onnx_path):
    dummy_input = torch.randn(1, 3, 112, 96)
    torch.onnx.export(net, dummy_input, onnx_path,
                      input_names=['input'], output_names=['embedding'],
                      opset_version=11)


def quantize_onnx(onnx_path, quant_path):
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)


def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, 112, 96, 3).astype(np.float32)]


def export_tflite(quant_onnx_path, tflite_path):
    onnx_model = onnx.load(quant_onnx_path)
    tf_rep = prepare(onnx_model)
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_rep.export_graph(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser(description='Convert checkpoint to quantized models')
    parser.add_argument('--checkpoint', required=True, help='Path to .ckpt file')
    parser.add_argument('--onnx_out', default='mobilefacenet_int8.onnx', help='Output quantized onnx path')
    parser.add_argument('--tflite_out', default='mobilefacenet_int8.tflite', help='Output quantized tflite path')
    args = parser.parse_args()

    net = load_model(args.checkpoint)
    onnx_fp32 = 'temp_model.onnx'
    export_onnx(net, onnx_fp32)
    quantize_onnx(onnx_fp32, args.onnx_out)
    export_tflite(args.onnx_out, args.tflite_out)
    os.remove(onnx_fp32)


if __name__ == '__main__':
    main()
