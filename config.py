BASE_DATASET_DIR = '/mnt/Windows/Users/taufi/MyFile/Projects/datasets/N-BaIoT'
BASE_GENERATED_IMAGE_DIR = 'generated_image'
BASE_RESULT_DIR = 'result'

N_IMAGE_CHANNEL = 3

# for height of image
CHUNK_SIZE_1 = 102
CHUNK_SIZE_2 = 51

BENIGN_FILENAME = '1.benign.csv'
MIRAI_FILENAME = '1.mirai.ack.csv'

# Training and validation config
BATCH_SIZE = 2
LABEL_MODE = 'binary'
IMG_HEIGHT = 1
IMG_WIDTH = 1
VALIDATION_SPLIT = 0.2
SEED = 111
NUM_EPOCHS = 100