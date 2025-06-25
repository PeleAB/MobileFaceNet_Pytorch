#!/usr/bin/env python3
"""Prepare an NPZ file from the bundled LFW sample images.

This script loads all images under `data/lfw-deepfunneled/lfw-deepfunneled`
using Keras' ``ImageDataGenerator`` and saves them to ``lfw_test_data.npz``.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'data',
                            'lfw-deepfunneled', 'lfw-deepfunneled')
    nb_test_files = sum(len(files) for _, _, files in os.walk(base_dir))

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        base_dir,
        target_size=(224, 224),
        batch_size=nb_test_files,
        class_mode='categorical',
        shuffle=False,
    )
    x_test, y_test = next(generator)
    np.savez('lfw_test_data.npz', x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()
