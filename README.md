# MobileFaceNet

## Introduction
* This repository is the pytorch implement of the paper: [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/pdf/1804.07573.pdf) and I almost follow the implement details of the paper.
* I train the model on CASIA-WebFace dataset, and evaluate on LFW dataset.

## Requirements

* Python 3.5
* pytorch 0.4+
* GPU memory

## Usage

### Part 1: Preprocessing

* All images of dataset are preprocessed following the [SphereFace](https://github.com/wy1iu/sphereface) and you can download the aligned images at [Align-CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1k3Cel2wSHQxHO9NkNi3rkg) and [Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg).

### Part 2: Train

  1. Change the **CAISIA_DATA_DIR** and **LFW_DATA_DIR** in `config.py` to your data path if needed.
     The repository now defaults `LFW_DATA_DIR` to `/data` so the evaluation
     scripts can find the dataset mounted at that location.
  
  2. Train the mobilefacenet model. 
  
        **Note:** The default settings set the batch size of 512, use 2 gpus and train the model on 70 epochs. You can change the settings in `config.py`
      ```
      python3 train.py
      ```
      
### Part 3: Test

  1. Test the model on LFW.
    
      **Note:** I have tested `lfw_eval.py` on the caffe model at [SphereFace](https://github.com/wy1iu/sphereface), it gets the same result.
    
      ```
      python3 lfw_eval.py --resume --feature_save_dir
      ```
      * `--resume:` path of saved model
      * `--feature_save_dir:` path to save the extracted features (must be .mat file)

### Part 4: Quantization

Two helper scripts are provided for converting the trained model to a fully
quantized format.

1. **export_quantized.py** – converts a training checkpoint to an int8 ONNX
   model and a fully quantized TFLite model:

   ```bash
   python3 export_quantized.py --checkpoint path/to/checkpoint.pth \
       --onnx_out mobilefacenet_int8.onnx --tflite_out mobilefacenet_int8.tflite
   ```

2. **quantize_onnx.py** – performs post training static quantization on an
   existing FP32 ONNX model. Pass `--dataset` to provide a calibration dataset
   (folder with sub‑directories for each class). If omitted, random data will be
   used for calibration.

   ```bash
   python3 quantize_onnx.py --input mobilefacenet.onnx \
       --dataset path/to/calibration_set --output mobilefacenet_int8.onnx
   ```

## Results

  * You can just run the `lfw_eval.py` to get the result, the accuracy on LFW like this:

  | Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | AVE(ours) | Paper(112x96) |
  | ------ |------|------|------|------|------|------|------|------|------|------| ------ | ------ |
  | ACC | 99.00 | 99.00 | 99.00 | 98.67 | 99.33 | 99.67 | 99.17 | 99.50 | 100.00 | 99.67| **99.30** | 99.18 |


## Reference resources

  * [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
  * [SphereFace](https://github.com/wy1iu/sphereface)
  * [Insightface](https://github.com/deepinsight/insightface)
