import os
import pandas as pd
from convert import convert_pict, get_list_pics
from config import BASE_DATASET_DIR, BASE_GENERATED_IMAGE_DIR


# untuk mengkoversi semua file dataset yang ada di folder
def convert_all(filter_column=None):
    list_filenames = os.listdir(BASE_DATASET_DIR)
    list_filenames.sort()
    for dataset_filename in list_filenames[:-4]:
        print(dataset_filename)
        convert(dataset_filename=dataset_filename, filter_column=filter_column)

def convert(dataset_filename, filter_column=None):
    df = pd.read_csv(f'{BASE_DATASET_DIR}/{dataset_filename}')
    if filter_column:
        image_dir = f'{BASE_GENERATED_IMAGE_DIR}/v1/{filter_column}/{dataset_filename[:-4]}'
        df_filtered = df[df.columns[df.columns.str.contains(filter_column)].to_list()]
    else:
        image_dir = f'{BASE_GENERATED_IMAGE_DIR}/v1/all/{dataset_filename[:-4]}'
        df_filtered = df
    n_features = df_filtered.shape[1]
    list_pics_1, list_pics_2 = get_list_pics(df_filtered)    
    os.system(f'mkdir -p {image_dir}')

    convert_pict(image_dir=image_dir, list_pics=list_pics_1, n_features=n_features, pict_prefix='pic1')
    convert_pict(image_dir=image_dir, list_pics=list_pics_2, n_features=n_features, pict_prefix='pic2')
    