BATCH_SIZE = 512
SAVE_FREQ = 1
TEST_FREQ = 1
TOTAL_EPOCH = 70

RESUME = ''
SAVE_DIR = './model'
MODEL_PRE = 'CASIA_B512_'


CASIA_DATA_DIR = '/home/xiaocc/Documents/caffe_project/sphereface/train/data'
# Directory containing the LFW evaluation data. By default use the mounted
# /data directory so the evaluation script can locate the dataset without
# additional configuration.
LFW_DATA_DIR = '/data'

GPU = 0, 1

