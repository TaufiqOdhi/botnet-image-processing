import numpy as np
from PIL import Image
from config import N_IMAGE_CHANNEL, CHUNK_SIZE_1, CHUNK_SIZE_2


def convert_pict(list_pics, n_features, image_dir, pict_prefix):
    for i in range(len(list_pics)):
        a = list_pics[i].size // N_IMAGE_CHANNEL
        a = a // n_features
        Image.fromarray(list_pics[i].reshape(a, n_features, N_IMAGE_CHANNEL), mode='RGB').save(f'{image_dir}/{pict_prefix}_{i}.png')

def get_list_pics(df):
    list_pics_1, list_pics_2 = np.array_split(df.to_numpy(), 2)
    list_pics_1 = np.array_split(list_pics_1[:-(len(list_pics_1) % CHUNK_SIZE_1)], len(list_pics_1) // CHUNK_SIZE_1)
    list_pics_2 = np.array_split(list_pics_2[:-(len(list_pics_2) % CHUNK_SIZE_2)], len(list_pics_2) // CHUNK_SIZE_2)

    return [list_pics_1, list_pics_2]
